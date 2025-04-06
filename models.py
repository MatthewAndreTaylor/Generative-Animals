import math
import torch
from torch import nn
import torch.nn.functional as F


# Future work: Use PhaseModulated Emb https://github.com/Roblox/cube/blob/main/cube3d/model/autoencoder/embedder.py#L7
class SinPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim >= 2, "dim must be >= 2 for SinPosEmb"
        emb = torch.exp(torch.arange(dim // 2) * -math.log(10000.0) / (dim // 2 - 1))
        self.register_buffer("emb", emb, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.emb * x.unsqueeze(-1)
        return torch.cat((m.sin(), m.cos()), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.scale = dim**0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.g * self.scale


class PreNormRes(nn.Module):
    def __init__(self, dim: int, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, classes_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(time_emb_dim + classes_emb_dim, dim_out * 2)
        )
        self.conv0, self.conv1 = nn.Conv2d(dim, dim_out, 3, padding=1), nn.Conv2d(
            dim_out, dim_out, 3, padding=1
        )
        self.norm0, self.norm1 = RMSNorm(dim_out), RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, cond_emb):
        scale, shift = self.mlp(cond_emb).view(x.shape[0], -1, 1, 1).chunk(2, 1)
        h = self.act(self.norm0(self.conv0(x)) * (scale + 1) + shift)
        return self.act(self.norm1(self.conv1(h))) + self.res_conv(x)


# Future attention work: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/unet.py#L200
class Attention(nn.Module):
    def __init__(self, dim: int, dim_head: int = 32, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(b, 3, self.heads, -1, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q = q * self.scale
        attn = torch.matmul(q.transpose(-2, -1), k)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1))
        out = out.transpose(-2, -1).reshape(b, -1, h, w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int = 32, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).reshape(b, 3, self.heads, -1, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = out.reshape(b, self.heads, -1, h, w)
        out = out.reshape(b, self.heads * out.shape[2], h, w)
        return self.to_out(out)


class UNet(nn.Module):
    def __init__(
        self,
        dim,
        num_classes,
        cond_drop_prob,
        channels=3,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()
        self.cond_drop_prob = cond_drop_prob
        self.channels = channels
        self.conv_in = nn.Conv2d(channels, dim, kernel_size=3, padding=1)
        dims = [dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        mid_dim = dims[-1]

        # time embeddings
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # class embeddings
        self.classes_emb = nn.Embedding(num_classes, dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))
        classes_dim = dim * 4
        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim), nn.GELU(), nn.Linear(classes_dim, classes_dim)
        )

        # layers
        self.down_block1s = nn.ModuleList([])
        self.down_block2s = nn.ModuleList([])
        self.down_attns = nn.ModuleList([])
        self.down_downsamples = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)

            self.down_block1s.append(ResnetBlock(dim_in, dim_in, time_dim, classes_dim))
            self.down_block2s.append(ResnetBlock(dim_in, dim_in, time_dim, classes_dim))
            self.down_attns.append(PreNormRes(dim_in, LinearAttention(dim_in)))
            self.down_downsamples.append(
                nn.Conv2d(dim_in, dim_out, 4, 2, padding=1)
                if not is_last
                else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            )

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_dim, classes_dim)
        self.mid_attn = PreNormRes(mid_dim, Attention(mid_dim))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_dim, classes_dim)

        self.up_block1s = nn.ModuleList([])
        self.up_block2s = nn.ModuleList([])
        self.up_attns = nn.ModuleList([])
        self.up_upsamples = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (len(in_out) - 1)

            self.up_block1s.append(
                ResnetBlock(dim_out + dim_in, dim_out, time_dim, classes_dim)
            )
            self.up_block2s.append(
                ResnetBlock(dim_out + dim_in, dim_out, time_dim, classes_dim)
            )
            self.up_attns.append(PreNormRes(dim_out, LinearAttention(dim_out)))
            self.up_upsamples.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(dim_out, dim_in, 3, padding=1),
                )
                if not is_last
                else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            )

        self.final_res_block = ResnetBlock(dim * 2, dim, time_dim, classes_dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        classes: torch.Tensor,
        cond_drop_prob: float,
    ) -> torch.Tensor:
        b = x.shape[0]
        classes_emb = self.classes_emb(classes)
        if cond_drop_prob > 0.0:
            keep_mask = torch.zeros((b, 1), device=x.device).uniform_(0.0, 1.0) < (
                1.0 - cond_drop_prob
            )
            null_classes_emb = self.null_classes_emb.expand(b, -1)
            classes_emb = torch.where(keep_mask, classes_emb, null_classes_emb)

        t, c = self.time_mlp(time), self.classes_mlp(classes_emb)
        cond_emb = torch.cat((t, c), dim=-1)
        x = self.conv_in(x)
        x_copy = x.clone()

        h = []
        for block1, block2, attn, downsample in zip(
            self.down_block1s, self.down_block2s, self.down_attns, self.down_downsamples
        ):
            x = block1(x, cond_emb)
            h.append(x)
            x = block2(x, cond_emb)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, cond_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond_emb)

        for block1, block2, attn, upsample in zip(
            self.up_block1s, self.up_block2s, self.up_attns, self.up_upsamples
        ):
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, cond_emb)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, cond_emb)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, x_copy), dim=1)
        x = self.final_res_block(x, cond_emb)
        return self.final_conv(x)

    def forward_with_cond_scale(
        self,
        x,
        t,
        classes,
        cond_scale: float = 1.0,
        rescaled_phi: float = 0.0,
    ) -> torch.Tensor:
        logits = self.forward(x, t, classes, cond_drop_prob=0.0)
        null_logits = self.forward(x, t, classes, cond_drop_prob=1.0)
        update = logits - null_logits
        update = orthogonal_component(update, logits)
        scaled_logits = logits + update * (cond_scale - 1.0)

        dims = list(range(1, scaled_logits.ndim))
        logits_std = torch.std(logits, dim=dims, keepdim=True)
        scaled_logits_std = torch.std(scaled_logits, dim=dims, keepdim=True)
        rescaled_logits = scaled_logits * (logits_std / scaled_logits_std)
        return rescaled_logits * rescaled_phi + scaled_logits * (1.0 - rescaled_phi)


def orthogonal_component(vector, basis):
    original_shape, original_dtype = vector.shape, vector.dtype
    vector = vector.view(original_shape[0], -1).double()
    basis = basis.view(original_shape[0], -1).double()
    unit_basis = F.normalize(basis, dim=-1)
    parallel_component = (vector * unit_basis).sum(dim=-1, keepdim=True) * unit_basis
    orthogonal_component = vector - parallel_component
    return orthogonal_component.view(original_shape).to(original_dtype)


def cosine_beta_schedule(timesteps: int):
    """
    cosine schedule from
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py#L36
    """
    x = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.double)
    alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def gather_timestep(a, t):
    return a.gather(0, t).view(t.shape[0], 1, 1, 1)


class GaussianDiffusion(nn.Module):
    def __init__(
        self, model, image_size: int, timesteps: int, weight_type=torch.float32
    ):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.num_timesteps = timesteps
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        register_buff = lambda name, val: self.register_buffer(
            name, val.to(weight_type)
        )
        register_buff("rt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buff("rt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buff("rt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buff("rt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        register_buff(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buff(
            "posterior_mean_c0",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buff(
            "posterior_mean_c1",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def forward(self, img, classes):
        t = torch.randint(0, self.num_timesteps, (img.shape[0],), device=img.device)
        noise = torch.randn_like(img, device=img.device)
        x = (
            gather_timestep(self.rt_alphas_cumprod, t) * img
            + gather_timestep(self.rt_one_minus_alphas_cumprod, t) * noise
        )
        model_out = self.model(x, t, classes, self.model.cond_drop_prob)
        loss = F.mse_loss(model_out, noise, reduction="none")
        return loss.mean()  # Future work use adversarial loss or lpips

    def p_sample(self, x, t: int, classes, cond_scale: float, rescaled_phi: float):
        b_t = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_output = self.model.forward_with_cond_scale(
            x, b_t, classes, cond_scale, rescaled_phi
        )
        x_start = (
            gather_timestep(self.rt_recip_alphas_cumprod, b_t) * x
            - gather_timestep(self.rt_recipm1_alphas_cumprod, b_t) * model_output
        ).clamp_(-1.0, 1.0)

        model_mean = (
            gather_timestep(self.posterior_mean_c0, b_t) * x_start
            + gather_timestep(self.posterior_mean_c1, b_t) * x
        )
        model_log_variance = gather_timestep(self.posterior_log_variance_clipped, b_t)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img

    @torch.jit.export
    def sample(self, classes, cond_scale: float, rescaled_phi: float = 0.7):
        img = torch.randn(
            (classes.shape[0], self.model.channels, self.image_size, self.image_size),
            device=classes.device,
            dtype=torch.float32,
        )
        for t in range(self.num_timesteps - 1, -1, -1):
            img = self.p_sample(img, t, classes, cond_scale, rescaled_phi)
        return img
