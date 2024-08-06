# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

# Base
import itertools
from glob import glob
from tqdm import tqdm
import time
from contextlib import nullcontext
import shutil
from pathlib import Path
import math
import random
from tqdm import tqdm

# ML
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import wandb

# Local
from supervoice_flow.config import config
from supervoice_flow.model import AudioFlow
from supervoice_flow.tensors import count_parameters, probability_binary_mask, drop_using_mask, random_interval_masking
from training.dataset import create_loader

# Train parameters
train_experiment = "flow-01"
train_project="supervoice-flow-2"
train_snapshot_overwrite = True
train_datasets = "https://external_datasets.korshakov.com/librilight-large-processed/"
# train_datasets = "./external_datasets/librilight-large-processed/"
train_duration = 15 # seconds, 15s x 5 (batches) = 75s per GPU
train_source_experiment = None
train_auto_resume = True
train_batch_size = 5 # Per GPU
train_clean = True
train_target_gpus = 32 # 16x2 (double batch size) = 32 GPU to match paper
train_steps = 600000 # Directly matches paper
train_loader_workers = 64
train_log_every = 1
train_save_every = 1000
train_watch_every = 1000
train_lr_start = 1e-7
train_lr_max = 5e-5
train_warmup_steps = 5000
train_mixed_precision = "fp16" # "bf16" or "fp16" or None
train_clip_grad_norm = 0.2
train_sigma = 1e-5

# Train
def main():

    # Prepare accelerator
    train_grad_accum_every = train_target_gpus // torch.cuda.device_count()
    ddp_kwargs = DistributedDataParallelKwargs()
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps = train_grad_accum_every, mixed_precision=train_mixed_precision)
    accelerator.print(f"Using {torch.cuda.device_count()} GPUs with {train_grad_accum_every} gradient accumulation steps, with total batch size of {torch.cuda.device_count() * train_grad_accum_every}.")
    device = accelerator.device
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if train_mixed_precision == "fp16" else (torch.bfloat16 if train_mixed_precision == "bf16" else torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True 
    lr_start = train_lr_start * accelerator.num_processes
    lr_max = train_lr_max * accelerator.num_processes

    # Prepare dataset
    accelerator.print("Loading dataset...")
    train_loader = create_loader(datasets = train_datasets, duration = train_duration, num_workers = train_loader_workers, batch_size = train_batch_size)

    # Prepare model
    accelerator.print("Loading model...")
    step = 0
    raw_model = AudioFlow(config)
    model = raw_model
    wd_params, no_wd_params = [], []
    for param in model.parameters():
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    optim = torch.optim.AdamW([{'params': wd_params}, {'params': no_wd_params, 'weight_decay': 0}], lr_max, betas=[0.9, 0.99], weight_decay=0.01, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max = train_steps)

    # Accelerate
    model, optim = accelerator.prepare(model, optim)
    train_cycle = cycle(train_loader)
    hps = {
        "train_lr_start": train_lr_start, 
        "train_lr_max": train_lr_max, 
        "batch_size": train_batch_size, 
        "grad_accum_every": train_grad_accum_every,
        "steps": train_steps, 
        "warmup_steps": train_warmup_steps,
        "mixed_precision": train_mixed_precision
    }
    accelerator.init_trackers(train_project, config=hps)
    if accelerator.is_main_process:
        wandb.watch(model, log="all", log_freq=train_watch_every * train_grad_accum_every)

    # Save
    def save():
        if train_snapshot_overwrite:
            fname = str(output_dir / f"{train_experiment}.pt")
            fname_step = str(output_dir / f"{train_experiment}.{step}.pt")
            torch.save({

                # Model
                'model': raw_model.state_dict(), 

                # Optimizer
                'step': step,
                'optimizer': optim.state_dict(), 
                'scheduler': scheduler.state_dict(),

            },  fname_step)
            shutil.move(fname_step, fname)
        else:
        
            # Save step checkpoint
            fname = str(output_dir / f"{train_experiment}.pt")
            fname_step = str(output_dir / f"{train_experiment}.{step}.pt")
            torch.save({

                # Model
                'model': raw_model.state_dict(), 

                # Optimizer
                'step': step,
                'optimizer': optim.state_dict(), 
                'scheduler': scheduler.state_dict(),

            },  fname_step)

            # Copy to latest
            shutil.copyfile(fname_step, fname)
            

    # Load
    source = None
    if (output_dir / f"{train_experiment}.pt").exists():
        source = train_experiment
    elif train_source_experiment and (output_dir / f"{train_source_experiment}.pt").exists():
        source = train_source_experiment

    if train_auto_resume and source is not None:
        accelerator.print("Resuming training...")
        checkpoint = torch.load(str(output_dir / f"{source}.pt"), map_location="cpu")

        # Model
        raw_model.load_state_dict(checkpoint['model'])

        # Optimizer
        optim.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        step = checkpoint['step']

        accelerator.print(f'Loaded at #{step}')
        

    # Train step
    def train_step():
        model.train()

        # Update LR
        if step < train_warmup_steps:
            lr = (lr_start + ((lr_max - lr_start) * step) / train_warmup_steps)
            for param_group in optim.param_groups:
                param_group['lr'] = lr
            lr = lr / accelerator.num_processes
        else:
            scheduler.step()
            lr = scheduler.get_last_lr()[0] / accelerator.num_processes

        # Load batch
        successful_cycles = 0
        failed_steps = 0
        last_loss = 0
        while successful_cycles < train_grad_accum_every:
            with accelerator.accumulate(model):
                spec = next(train_cycle)

                # Prepare batch
                batch_size = spec.shape[0]
                seq_len = spec.shape[1]

                # Normalize spectograms
                spec = (spec - config.audio.norm_mean) / config.audio.norm_std

                # Prepare target flow (CFM)
                times = torch.rand((batch_size,), dtype = spec.dtype, device = spec.device)
                t = rearrange(times, 'b -> b 1 1')
                source_noise = torch.randn_like(spec, device = spec.device)
                noise = (1 - (1 - train_sigma) * t) * source_noise + t * spec
                flow = spec - (1 - train_sigma) * source_noise

                # Masking 
                # 70% - 100% of the sequence is masked, with segments of at least 10 frames
                mask = random_interval_masking(batch_size, seq_len, 
                                                min_size = 10, 
                                                min_count = int(seq_len * 0.7), 
                                                max_count = seq_len, 
                                                device = spec.device)

                # Drop everything for unconditional generation
                # 0.1 probability of full mask
                conditional_drop_mask = probability_binary_mask(shape = (batch_size,), true_prob = 0.1, device = spec.device)

                # Merge masks
                mask = drop_using_mask(source = mask, replacement = True, mask = conditional_drop_mask)

                # Prepare condition spec
                condition_spec = drop_using_mask(source = spec, replacement = 0, mask = mask)

                # Train step
                with accelerator.autocast():
                    _, loss = model(

                        # Audio
                        audio = condition_spec.to(device, non_blocking=True), 
                        noise = noise.to(device, non_blocking=True), 

                        # Time
                        times = times.to(device, non_blocking=True), 

                        # Loss
                        loss_mask = mask.to(device, non_blocking=True), 
                        target = flow.to(device, non_blocking=True),
                    )

                # Backprop
                optim.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_clip_grad_norm)
                optim.step()

                # Log skipping step
                if optim.step_was_skipped:
                    failed_steps = failed_steps + 1
                    if torch.isnan(loss).any():
                        accelerator.print("Step was skipped with NaN loss")
                    else:
                        accelerator.print("Step was skipped")
                    if failed_steps > 20:
                        raise Exception("Too many failed steps")
                else:
                    successful_cycles = successful_cycles + 1
                    failed_steps = 0

                # Save last loss
                last_loss = loss.detach()

                # Cleanup
                del loss

        return last_loss.item(), lr

    #
    # Start Training
    #

    accelerator.print("Training started at step", step)
    while step < train_steps:
        start = time.time()
        loss, lr = train_step()
        end = time.time()

        # Advance
        step = step + 1

        # Summary
        if step % train_log_every == 0 and accelerator.is_main_process:
            accelerator.log({
                "learning_rate": lr,
                "loss": loss,
            }, step=step)
            accelerator.print(f'Step {step}: loss={loss}, lr={lr}, time={end - start} sec')
        
        # Save
        if step % train_save_every == 0 and accelerator.is_main_process:
            save()

    # End training
    if accelerator.is_main_process:
        accelerator.print("Finishing training...")
        save()
    accelerator.end_training()
    accelerator.print('✨ Training complete!')

#
# Utility
#

def cycle(dl):
    while True:
        for data in dl:
            yield data    

if __name__ == "__main__":
    main()