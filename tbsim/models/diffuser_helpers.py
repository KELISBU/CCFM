import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb

# import diffuser.utils as utils

#-----------------------------------------------------------------------------#
#---------------------------------- modules ----------------------------------#
#-----------------------------------------------------------------------------#

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=4):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

#-----------------------------------------------------------------------------#
#--------------------------------- attention ---------------------------------#
#-----------------------------------------------------------------------------#

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)

#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def apply_conditioning(x, conditions, action_dim):
    for t, val in conditions.items():
        x[:, t, action_dim:] = val.clone()
    return x


#-----------------------------------------------------------------------------#
#------------------------------ history encoders -----------------------------#
#-----------------------------------------------------------------------------#

class AgentHistoryEncoder(nn.Module):
    """Encode ego agent history (positions, yaws, speeds, extent) into a feature vector."""
    def __init__(self, num_steps, out_dim=128, use_norm=True,
                 norm_info=([0.0]*5, [1.0]*5)):
        super().__init__()
        self.num_steps = num_steps
        self.out_dim = out_dim
        self.use_norm = use_norm
        # norm_info: (mean_5, std_5) for [dx, dy, yaw, speed, extent_len]
        mean = torch.tensor(norm_info[0], dtype=torch.float32) if norm_info else torch.zeros(5)
        std  = torch.tensor(norm_info[1], dtype=torch.float32) if norm_info else torch.ones(5)
        self.register_buffer('norm_mean', mean)
        self.register_buffer('norm_std', std)
        # input per timestep: pos(2) + yaw(1) + speed(1) + extent_len(1) = 5
        self.lstm = nn.LSTM(5, out_dim, batch_first=True)

    def forward(self, positions, yaws, speeds, extents, availabilities):
        """
        Args:
            positions: [B, T, 2]
            yaws:      [B, T, 1] or [B, T]
            speeds:    [B, T, 1] or [B, T]
            extents:   [B, D] (D>=1, uses first dim as length)
            availabilities: [B, T]
        Returns:
            feature: [B, out_dim]
        """
        B, T, _ = positions.shape
        if yaws.dim() == 2:
            yaws = yaws.unsqueeze(-1)
        if speeds.dim() == 2:
            speeds = speeds.unsqueeze(-1)
        # extent length broadcast to all timesteps [B, T, 1]
        ext_len = extents[:, 0:1].unsqueeze(1).expand(B, T, 1)
        # [B, T, 5]
        x = torch.cat([positions, yaws, speeds, ext_len], dim=-1)
        mask = availabilities.unsqueeze(-1)  # [B, T, 1]
        if self.use_norm:
            x = (x - self.norm_mean) / (self.norm_std + 1e-6)
        x = x * mask
        _, (h, _) = self.lstm(x)
        return h.squeeze(0)  # [B, out_dim]


class NeighborHistoryEncoder(nn.Module):
    """Encode neighbor agents' histories into a single feature vector via max-pooling."""
    def __init__(self, num_steps, out_dim=128, use_norm=True,
                 norm_info=([0.0]*5, [1.0]*5)):
        super().__init__()
        self.agent_encoder = AgentHistoryEncoder(num_steps, out_dim, use_norm, norm_info)
        self.out_dim = out_dim

    def forward(self, positions, yaws, speeds, extents, availabilities):
        """
        Args:
            positions: [B, N, T, 2]
            yaws:      [B, N, T, 1] or [B, N, T]
            speeds:    [B, N, T, 1] or [B, N, T]
            extents:   [B, N, D]
            availabilities: [B, N, T]
        Returns:
            feature: [B, out_dim]
        """
        B, N, T, _ = positions.shape
        pos_flat = positions.reshape(B * N, T, -1)
        yaw_flat = yaws.reshape(B * N, T, -1)
        spd_flat = speeds.reshape(B * N, T, -1)
        ext_flat = extents.reshape(B * N, -1)
        avail_flat = availabilities.reshape(B * N, T)
        feats = self.agent_encoder(pos_flat, yaw_flat, spd_flat, ext_flat, avail_flat)
        feats = feats.reshape(B, N, -1)  # [B, N, out_dim]
        pooled = feats.max(dim=1)[0]     # [B, out_dim]
        return pooled


class MapEncoder(nn.Module):
    """CNN-based map encoder wrapping RasterizedMapEncoder."""
    def __init__(self, model_arch="resnet18", input_image_shape=(3, 224, 224),
                 global_feature_dim=256):
        super().__init__()
        from tbsim.models.base_models import RasterizedMapEncoder
        self.encoder = RasterizedMapEncoder(
            model_arch=model_arch,
            input_image_shape=input_image_shape,
            feature_dim=global_feature_dim,
        )

    def forward(self, image):
        """
        Args:
            image: [B, C, H, W]
        Returns:
            (global_feat, None): global_feat is [B, feature_dim]
        """
        global_feat = self.encoder(image)
        return global_feat, None


#-----------------------------------------------------------------------------#
#--------------------------- state transform utils ---------------------------#
#-----------------------------------------------------------------------------#


def angle_diff(theta1, theta2):
    '''
    :param theta1: angle 1 (..., 1)
    :param theta2: angle 2 (..., 1)
    :return diff: smallest angle difference between angles (..., 1)
    '''
    period = 2*np.pi
    diff = (theta1 - theta2 + period / 2) % period - period / 2
    diff[diff > np.pi] = diff[diff > np.pi] - (2 * np.pi)
    return diff

def convert_state_to_state_and_action(traj_state, vel_init, dt, data_type='torch'):
    """Infer vel and action (acc, yawvel) from state (x, y, yaw) based on Unicycle.

    Args:
        traj_state: (batch_size, [num_agents], num_steps, 3)
        vel_init: (batch_size, [num_agents],)
        dt: float
    Returns:
        traj_state_and_action: (batch_size, [num_agents], num_steps, 6)
            Format: (x, y, vel, yaw, acc, yawvel)
    """
    BM = traj_state.shape[:-2]
    if data_type == 'torch':
        sin = torch.sin
        cos = torch.cos
        device = traj_state.device
        pos_init = torch.zeros(*BM, 1, 2, device=device)
        yaw_init = torch.zeros(*BM, 1, 1, device=device)
        cat = lambda arr, dim: torch.cat(arr, dim=dim)
    elif data_type == 'numpy':
        sin = np.sin
        cos = np.cos
        pos_init = np.zeros((*BM, 1, 2))
        yaw_init = np.zeros((*BM, 1, 1))
        cat = lambda arr, dim: np.concatenate(arr, axis=dim)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    target_pos = traj_state[..., :2]
    traj_yaw = traj_state[..., 2:]

    pos = cat((pos_init, target_pos), dim=-2)
    yaw = cat((yaw_init, traj_yaw), dim=-2)

    vel_init = vel_init[..., None, None]
    vel = (pos[..., 1:, 0:1] - pos[..., :-1, 0:1]) / dt * cos(
        yaw[..., 1:, :]
    ) + (pos[..., 1:, 1:2] - pos[..., :-1, 1:2]) / dt * sin(
        yaw[..., 1:, :]
    )
    vel = cat((vel_init, vel), dim=-2)

    acc = (vel[..., 1:, :] - vel[..., :-1, :]) / dt
    yawdiff = angle_diff(yaw[..., 1:, :], yaw[..., :-1, :])
    yawvel = yawdiff / dt

    pos, yaw, vel = pos[..., 1:, :], yaw[..., 1:, :], vel[..., 1:, :]
    traj_state_and_action = cat((pos, vel, yaw, acc, yawvel), dim=-1)
    return traj_state_and_action


def state_grad_general_transform(x_guidance, data_batch, transform_params, bsize, num_samp=1):
    """Transform state (x,y,yaw) to state+action (x,y,vel,yaw,acc,yawvel) using curr_speed.

    Args:
        x_guidance: (B*N, T, 3)
    """
    expand_speed = data_batch['curr_speed'].unsqueeze(1).expand((bsize, num_samp)).reshape((bsize * num_samp))
    x_all = convert_state_to_state_and_action(x_guidance, expand_speed, dt=transform_params['dt'])
    return x_all


#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#
