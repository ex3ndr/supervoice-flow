# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

# Base
import itertools
from glob import glob
import textgrid
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
from model.config import config
from model.model_audio import AudioEffector
from model.tensors import count_parameters, probability_binary_mask, drop_using_mask, interval_mask
from training.dataset import load_distorted_loader

# Train parameters
train_experiment = "exp-02"
train_project="supervoice-effector"
train_datasets = [
    "./external_datasets/libritts-r-clean-100/", 
    "./external_datasets/libritts-r-clean-360/", 
    "./external_datasets/lj-speech-1.1/",
    "./external_datasets/vctk-corpus-0.92/"
]
train_eval_datasets = [
    "./external_datasets/libritts-r/test-clean/"
]
train_duration = 5 # seconds
train_source_experiment = None
train_auto_resume = True
train_batch_size = 16 # Per GPU
train_grad_accum_every = 8
train_steps = 1000000
train_loader_workers = 8
train_log_every = 1
train_save_every = 1000
train_watch_every = 1000
train_evaluate_every = 1
train_evaluate_batch_size = 10
train_max_segment_size = 500
train_lr_start = 1e-7
train_lr_max = 2e-5
train_warmup_steps = 5000
train_mixed_precision = "fp16" # "bf16" or "fp16" or None
train_clip_grad_norm = 0.2
train_sigma = 1e-5

# Train
def main():

    # Prepare accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps = train_grad_accum_every, mixed_precision=train_mixed_precision)
    device = accelerator.device
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if train_mixed_precision == "fp16" else (torch.bfloat16 if train_mixed_precision == "bf16" else torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True 
    set_seed(42)
    lr_start = train_lr_start * accelerator.num_processes
    lr_max = train_lr_max * accelerator.num_processes

    # Prepare dataset
    accelerator.print("Loading dataset...")
    train_loader = load_distorted_loader(datasets = train_datasets, duration = train_duration, num_workers = train_loader_workers, batch_size = train_batch_size)
    test_loader = load_distorted_loader(datasets = train_eval_datasets, duration = train_duration, num_workers = train_loader_workers, batch_size = train_batch_size)

    # Prepare model
    accelerator.print("Loading model...")
    step = 0
    raw_model = AudioEffector(config)
    model = raw_model
    wd_params, no_wd_params = [], []
    for param in model.parameters():
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    optim = torch.optim.AdamW([{'params': wd_params}, {'params': no_wd_params, 'weight_decay': 0}], lr_max, betas=[0.9, 0.99], weight_decay=0.01, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max = train_steps)

    # Accelerate
    model, optim, train_loader, test_loader = accelerator.prepare(model, optim, train_loader, test_loader)
    train_cycle = cycle(train_loader)
    test_cycle = cycle(test_loader)
    test_batch = next(test_cycle)
    hps = {
        "train_lr_start": train_lr_start, 
        "train_lr_max": train_lr_max, 
        "batch_size": train_batch_size, 
        "grad_accum_every": train_grad_accum_every,
        "steps": train_steps, 
        "warmup_steps": train_warmup_steps,
        "mixed_precision": train_mixed_precision,
        "clip_grad_norm": train_clip_grad_norm,
    }
    accelerator.init_trackers(train_project, config=hps)
    if accelerator.is_main_process:
        wandb.watch(model, log="all", log_freq=train_watch_every * train_grad_accum_every)

    # Save
    def save():
        
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

        # Overwrite main checkpoint
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
        for _ in range(train_grad_accum_every):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    spec, spec_aug, wav, wav_aug = next(train_cycle)
                    batch_size = spec.shape[0]
                    seq_len = spec.shape[1]

                    # Normalize spectograms
                    spec = (spec - config.audio.norm_mean) / config.audio.norm_std
                    spec_aug = (spec_aug - config.audio.norm_mean) / config.audio.norm_std

                    # Prepare target flow (CFM)
                    times = torch.rand((batch_size,), dtype = spec.dtype, device = device)
                    t = rearrange(times, 'b -> b 1 1')
                    source_noise = torch.randn_like(spec, device=device)
                    noise = (1 - (1 - train_sigma) * t) * source_noise + t * spec
                    flow = spec - (1 - train_sigma) * source_noise

                    # Prepare mask (segments that are needed to be predicted)
                    # 70% - 100% of sequence, and 30% probability of masking everything
                    mask = interval_mask(batch_size, seq_len, int(seq_len * 0.7), seq_len, 0.3, device)

                    # Mix augment and clean audio depending on mask
                    condition_spec = drop_using_mask(source = spec_aug, replacement = 0, mask = ~mask) + drop_using_mask(source = spec, replacement = 0, mask = mask)

                    # Drop everything for unconditional generation
                    # 0.2 probability of dropping everything
                    conditional_drop_mask = probability_binary_mask(shape = (batch_size,), true_prob = 0.15, device = device)
                    condition_spec = drop_using_mask(source = condition_spec, replacement = 0, mask = conditional_drop_mask)

                    # Train step
                    predicted, loss = model(

                        # Audio
                        audio = condition_spec, 
                        noise = noise, 

                        # Time
                        times = times, 

                        # Loss
                        mask = mask, 
                        target = flow
                    )

                    # # Check if loss is nan
                    # if torch.isnan(loss) and accelerator.is_main_process:
                    #     raise RuntimeError("Loss is NaN")

                    # Backprop
                    optim.zero_grad()
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), train_clip_grad_norm)
                    optim.step()

                    # Log skipping step
                    if optim.step_was_skipped:
                        accelerator.print("Step was skipped")

        return loss, predicted, flow, lr

    # def train_eval():
    #     model.eval()
    #     with torch.inference_mode():
    #         tokens, style, audio = test_batch
    #         audio = (audio - config.audio.norm_mean) / config.audio.norm_std
    #         mask = torch.tokens(audio, device = device)
    #         predicted = model.sample(tokens = tokens, tokens_style = style, audio = audio, mask = mask)
    #         score = evaluate_mos(predicted, config.audio.sample_rate)
    #         gathered_score = accelerator.gather(score).cpu()
    #         if len(gathered_score.shape) == 0:
    #             gathered_score = gathered_score.unsqueeze(0)
    #         return gathered_score.mean().item()

    #
    # Start Training
    #

    accelerator.print("Training started at step", step)
    while step < train_steps:
        start = time.time()
        loss, predicted, flow, lr = train_step()
        end = time.time()

        # Advance
        step = step + 1

        # Summary
        if step % train_log_every == 0 and accelerator.is_main_process:
            accelerator.log({
                "learning_rate": lr,
                "loss": loss,
                "predicted/mean": predicted.mean(),
                "predicted/max": predicted.max(),
                "predicted/min": predicted.min(),
                "target/mean": flow.mean(),
                "target/max": flow.max(),
                "target/min": flow.min()
            }, step=step)
            accelerator.print(f'Step {step}: loss={loss}, lr={lr}, time={end - start} sec')
        
        # Evaluate
        # if step % train_evaluate_every == 0:
        #     accelerator.print("Evaluating...")
        #     mos = train_eval()
        #     accelerator.print(f"Step {step}: MOS={mos}")
        #     accelerator.log({"eval/mos": mos}, step=step)
        
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