import math
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torchdiffeq import odeint

from .transformer import Transformer, ConvPositionEmbed
from .tensors import drop_using_mask, merge_mask

class AudioFlow(torch.nn.Module):
    def __init__(self, config, *, cache_alibi = False):
        super(AudioFlow, self).__init__()
        self.config = config.model

        # Transformer input
        self.transformer_input = torch.nn.Linear(2 * config.audio.n_mels, self.config.n_dim)

        # Sinusoidal positional embedding for time
        self.sinu_pos_emb = LearnedSinusoidalPosEmb(self.config.n_dim)

        # Convolutional positional encoder
        self.conv_embed = ConvPositionEmbed(n_dim = self.config.n_dim, kernel_size = 31)

        # Transformer
        self.transformer = Transformer(
            n_heads = self.config.n_heads,
            n_layers = self.config.n_layers,
            n_dim = self.config.n_dim,
            n_dim_head = self.config.n_dim_head,
            n_dim_ffn = self.config.n_dim_ffn,
            n_non_bias_tokens = 1, # Exclude time embedding from attention bias
            att_dropout = 0,
            ffn_dropout = 0.1,
            cache_alibi = cache_alibi
        )

        # Prediction
        self.prediction = torch.nn.Linear(self.config.n_dim, config.audio.n_mels)

    def sample(self, *, audio, mask = None, steps, alpha = None, return_trajectory = False):
        
        #
        # Prepare
        #

        # Mask out audio
        source_audio = audio
        if mask is not None:
            audio = drop_using_mask(source = audio, replacement = 0, mask = mask)

        # Create noise
        noise = torch.randn_like(audio)

        # Create time interpolation
        times = torch.linspace(0, 1, steps, device = audio.device)

        #
        # Solver
        # 

        # Overwrite audio segment with predicted audio according to mask
        def merge_predicted(predicted):
            if mask is None:
                return predicted
            return merge_mask(source = source_audio, replacement = predicted, mask = mask)

        def solver(t, z):

            # If alpha is not provided
            if alpha is None:
                return self.forward(audio = audio.unsqueeze(0), noise = z.unsqueeze(0), times = t.unsqueeze(0)).squeeze(0)

            # If alpha is provided - zero out tokens and audio and mix together
            audio_empty = torch.zeros_like(audio)

            # Mix together
            audio_t = torch.stack([audio_empty, audio], dim = 0)
            noise_t = torch.stack([z, z], dim = 0) # Just double it
            t_t = torch.stack([t, t], dim = 0) # Just double it

            # Inference
            predicted_mix = self.forward(
                audio = audio_t, 
                noise = noise_t, 
                times = t_t
            )
            predicted_conditioned = predicted_mix[1]
            predicted_unconditioned = predicted_mix[0]
            
            # CFG prediction

            # There are different ways to do CFG, this is my very naive version, which worked for me:
            # prediction = (1 + alpha) * predicted_conditioned - alpha * predicted_unconditioned

            # Original paper uses a different one, but i found that it simply creates overexposed values
            # prediction = predicted_unconditioned + (predicted_conditioned - predicted_unconditioned) * alpha

            # This is from the latest paper that rescales original formula (https://arxiv.org/abs/2305.08891):
            prediction = predicted_conditioned + (predicted_conditioned - predicted_unconditioned) * alpha
            prediction_rescaled = predicted_conditioned.std() * (prediction / prediction.std())

            return prediction


        trajectory = odeint(solver, noise, times, atol = 1e-5, rtol = 1e-5, method = 'midpoint')

        #
        # Output sample and full trajectory
        #

        return merge_predicted(trajectory[-1]), trajectory

    def forward(self, *,  
        
        # Audio
        audio, 
        noise, 

        # Extra conditioning for fine-tuning
        condition = None,
        
        # Time
        times, 

        # Training    
        mask = None,
        target = None,
        mask_loss = False
    ):
        
        #
        # Prepare
        #

        if mask is None and target is not None and mask_loss:
            raise ValueError('Mask is required when target is provided and mask_loss enabled')
        if target is None and mask is not None:
            raise ValueError('Mask is not required when target is not provided')
        if condition is not None:
            assert condition.shape[0] == audio.shape[0], 'Condition should have the same batch size as audio'
            assert condition.shape[1] == audio.shape[1], 'Condition should have the same sequence length as audio'
            assert condition.shape[2] == self.config.n_dim, 'Condition should have ' + self.config.n_dim + ' channels'

        # Check shapes
        assert audio.shape[0] == noise.shape[0] # Batch
        assert audio.shape[1] == noise.shape[1] # Sequence length
        assert audio.shape[2] == noise.shape[2] # Channels length
        if mask is not None:
            assert audio.shape[0] == mask.shape[0] # Batch
            assert audio.shape[1] == mask.shape[1] # Squence length

        #
        # Compute
        #

        # Combine phoneme embeddings, masked audio and noizy audio
        output = torch.cat([audio, noise], dim = -1)

        # Apply transformer input layer
        output = self.transformer_input(output)

        # Apply condition after transformer input
        if condition is not None:
            output = output + condition

        # Apply sinusoidal positional embedding
        sinu_times = self.sinu_pos_emb(times).unsqueeze(1)
        output = torch.cat([output, sinu_times], dim=1)

        # Apply convolutional positional encoder
        output = self.conv_embed(output) + output

        # Run through transformer
        output = self.transformer(output)

        # Predict durations
        output = self.prediction(output)

        # Cut to length
        output = output[:, :-1, :]

        #
        # Loss
        #

        if target is not None:
            
            # Compute MSE loss
            loss = F.mse_loss(output, target, reduction = 'none')

            # Mean for each frame
            loss = reduce(loss, 'b n d -> b n', 'mean')

            # Mask out non target frames
            if mask_loss:
                loss = loss.masked_fill(~mask, 0.)

                # Number of masked frames
                n_masked_frames = mask.sum(dim = -1).clamp(min = 1)

                # Mean loss of expectation over masked loss
                loss = loss.sum(dim = -1) / n_masked_frames
            else:
                # Mean loss of expectation over each frame
                loss = loss.sum(dim = -1) / target.shape[1]

            # Expectation over loss of batch
            loss = loss.mean()

            return output, loss
        else:
            return output


class LearnedSinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        self.weights = torch.nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        return fouriered