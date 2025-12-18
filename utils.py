import os
import torch
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage


def p_sample_loop_gif(model, classes, shape, cond_scale=6., rescaled_phi=0.7, save_gif=True, gif_dir='gifs'):
    model.eval()
    to_pil = ToPILImage()
    
    with torch.no_grad():
        img = torch.randn(shape, device=model.betas.device)

        if save_gif:
            os.makedirs(gif_dir, exist_ok=True)

        for t in tqdm(reversed(range(0, model.num_timesteps)), desc='sampling', total=model.num_timesteps):
            img, _ = model.p_sample(img, t, classes, cond_scale, rescaled_phi)
            if save_gif:
                frame = img.detach().cpu().clamp(-1, 1)
                frame = (frame + 1) / 2
                grid = make_grid(frame, padding=2)
                frame = to_pil(grid)
                frame.save(os.path.join(gif_dir, f'frame_{t:03d}.png'))

    return img