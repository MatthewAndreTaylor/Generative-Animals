# Generative Animals

https://github.com/user-attachments/assets/b013e85c-585c-43c5-8c96-34e3c42c4e32

## Overview

Using a lightweight diffusion model, the goal was to generate small animal images. The images could be upsampled using super-resolution and converted to SVG to make stickers or art assets for a game. The goal was to create animal breeds that do not exist in the real world but are realistic-looking.

| | | | |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
| <img width="64" height="64" alt="bird" src="https://github.com/user-attachments/assets/a9276326-b60a-4f45-b7ba-4e05f65d56c5" /> | <img width="64" height="64" alt="cat" src="https://github.com/user-attachments/assets/2ceee81d-ae6a-4a02-8b17-f23eb44a605f" /> | <img width="64" height="64" alt="dog" src="https://github.com/user-attachments/assets/4003b312-4eb1-4fa2-bd27-abe5955c5c3d" /> | <img width="64" height="64" alt="fish" src="https://github.com/user-attachments/assets/22b00336-2b80-4394-b552-a2cbcf14630c" /> |
<img width="64" height="64" alt="frog" src="https://github.com/user-attachments/assets/1037b0f8-2543-44b4-8854-830478358251" /> | <img width="64" height="64" alt="rabbit" src="https://github.com/user-attachments/assets/218c953a-94ab-4036-9cb9-f1b633811753" /> | <img width="64" height="64" alt="horse" src="https://github.com/user-attachments/assets/5727d19e-0c21-4fb1-86c5-c907e4515c4a" /> | <img width="64" height="64" alt="turtle" src="https://github.com/user-attachments/assets/c137baba-de43-47bb-b535-eb5d31957ccc" />


I decided not to use a VQVAE since in experiments it did speed up training time when pre-saving latents; however, reconstructions were not perfect, and my dataset was fairly small (8 classes, 512 images for training per class) in comparison to datasets like CIFAR-10 (10 classes and 5000 images for training per class). The latent space did not have the best distribution, so the new generation could have more unexpected artifacts.

I also decided on not using EMA because it slowed down training time and increased the code complexity. For this example, I wanted to demonstrate generating images with as little code as possible while still keeping a reasonable fidelity of generated images.

When gathering images for my datasets, I noticed a lack of diversity of tools for creating image datasets. I created my own Flask utility apps for ranking images in the dataset and merging new data points when the existing data did not satisfy my needs. Gathering the training data was a slow task. I hope to see more support for gathering data in the future. The dataset I used was not perfect, and there is definitely work that could be done to improve it in the future.

Some generations have artifacts (extra ears on rabbits, missing fins on turtles, smudged faces on dogs). Some of these artifacts seemed to come from my dataset, but other artifacts seem to be consequences of using a diffusion network.

Most of my training was done in the train.ipynb notebook. The dataloaders num_workers option seemed to have a significant slowdown when working in the notebook. An optimization I made was to apply the transforms ahead of time and then pin the memory. This optimization had a positive impact on training speed. Another approach that could be used would be to generate pseudo-random prompts, then use existing models to generate images, then mask the images and convert them to SVG.

Examples found at [my site](https://matthewandretaylor.github.io/Generative-Animals).

The classes I chose to use were

- bird 🐦
- cat🐈
- dog 🐕
- fish 🐟
- frog 🐸
- horse 🐎
- rabbit 🐇
- turtle 🐢

## BibTeX

```bibtex
@article{Ho2022ClassifierFreeDG,
    title   = {Classifier-Free Diffusion Guidance},
    author  = {Jonathan Ho},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2207.12598}
}
```

```bibtex
@InProceedings{pmlr-v139-nichol21a,
    title       = {Improved Denoising Diffusion Probabilistic Models},
    author      = {Nichol, Alexander Quinn and Dhariwal, Prafulla},
    booktitle   = {Proceedings of the 38th International Conference on Machine Learning},
    pages       = {8162--8171},
    year        = {2021},
    editor      = {Meila, Marina and Zhang, Tong},
    volume      = {139},
    series      = {Proceedings of Machine Learning Research},
    month       = {18--24 Jul},
    publisher   = {PMLR},
    pdf         = {http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf},
    url         = {https://proceedings.mlr.press/v139/nichol21a.html},
}
```

```bibtex
@inproceedings{NEURIPS2020_4c5bcfec,
    author      = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
    booktitle   = {Advances in Neural Information Processing Systems},
    editor      = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
    pages       = {6840--6851},
    publisher   = {Curran Associates, Inc.},
    title       = {Denoising Diffusion Probabilistic Models},
    url         = {https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf},
    volume      = {33},
    year        = {2020}
}
```

## Acknowledgments

My codebase was motivated from OpenAI's [improved diffusion repo](https://github.com/openai/improved-diffusion/tree/main)
