# Matthew Taylor 2025

import glob
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple


class EncodedDataset(Dataset):
    def __init__(self, encoded_outputs):
        self.images = []
        self.labels = []

        for i, tensors in tqdm(enumerate(encoded_outputs)):
            for tensor in tensors:
                self.images.append(tensor)
                self.labels.append(i)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class AnimalDataset(Dataset):
    label_map = {
        "bird": 0,
        "cat": 1,
        "dog": 2,
        "fish": 3,
        "frog": 4,
        "horse": 5,
        "rabbit": 6,
        "turtle": 7,
    }

    # mode: RGBA, RGB, L
    def __init__(
        self,
        root,
        split: str = "train",
        transform: Optional[Callable] = None,
        mode: str = "RGB",
    ):
        self.root = root
        self.split = split
        self.transform = transform
        assert mode in ["RGBA", "RGB", "L"], f"Unknown mode: {mode}"
        self.images = []
        self.labels = []

        for directory in tqdm(os.listdir(self.root)):
            assert directory in self.label_map, f"Unknown label: {directory}"
            files = glob.glob(os.path.join(self.root, directory, "*.png"))

            if self.split == "train":
                files = files[:512]
            elif self.split == "all":
                files = files
            elif self.split == "val":
                files = files[512:]
            else:
                raise ValueError(f"Unknown split: {self.split}")

            for file in files:
                with Image.open(file).convert(mode) as im:
                    self.images.append(im)

                label = self.label_map[directory]
                # self.labels.append(torch.nn.functional.one_hot(torch.tensor(label), num_classes=8))
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> Tuple[Any, int]:
        img = self.images[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[index]

    def __iadd__(self, other: "AnimalDataset") -> "AnimalDataset":
        self.images += other.images
        self.labels += other.labels
        return self

    @property
    def num_classes(self) -> int:
        return len(self.label_map)


class PreloadTransformAnimalDataset(AnimalDataset):
    def __init__(
        self,
        root,
        split: str = "train",
        transform: Optional[Callable] = None,
        mode: str = "RGB",
    ):
        super().__init__(root, split=split, transform=None, mode=mode)
        assert transform is not None, "Transform must be provided"
        self.images = [transform(im) for im in self.images]

    def __getitem__(self, index) -> Tuple[Any, int]:
        return self.images[index], self.labels[index]


if __name__ == "__main__":
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torchvision.utils import make_grid
    from torch.utils.data import DataLoader

    import matplotlib.pyplot as plt

    animal_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (1.0,)),
        ]
    )

    animal_data = AnimalDataset(root="data/animals", transform=animal_transform)
    animal_loader = DataLoader(
        animal_data, batch_size=32, shuffle=True, pin_memory=True
    )

    (animal_images, _) = next(iter(animal_loader))

    def show(img):
        npimg = img.numpy()
        fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation="nearest")
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.show()

    show(make_grid(animal_images + 0.5))
