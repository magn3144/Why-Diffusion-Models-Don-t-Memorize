import numpy as np
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from urllib.request import urlretrieve


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def im_normalize(im):
    imn = (im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn

def tens2image(im):
    tmp = np.squeeze(im.numpy())
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0))


def download_sprites_files(
    output_dir="./data",
    base_url="https://raw.githubusercontent.com/Ryota-Kawamura/How-Diffusion-Models-Work/main",
    overwrite=False,
):
    """Download sprites data and label files from the upstream repository.

    Returns:
        dict: Mapping with keys 'sprites' and 'labels' containing local file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    targets = {
        "sprites": "sprites_1788_16x16.npy",
        "labels": "sprite_labels_nc_1788_16x16.npy",
    }

    saved_paths = {}
    for key, filename in targets.items():
        destination = output_path / filename
        source = f"{base_url.rstrip('/')}/{filename}"
        if overwrite or not destination.exists():
            urlretrieve(source, destination)
        saved_paths[key] = str(destination)

    return saved_paths


# Dataset from: https://github.com/Ryota-Kawamura/How-Diffusion-Models-Work
class SpritesDataset(Dataset):
    def __init__(self, 
                 transform, 
                 img_file='./data/sprites.npy',
                 num_samples=40000,
                 seed=1
    ):
        self.images = np.load(img_file)
        self.num_samples = num_samples
        self.seed = seed
        # Reduce dataset size
        if num_samples:
            set_seed(seed=self.seed)
            sampled_indeces = random.sample(range(len(self.images)), self.num_samples)
            self.images = self.images[sampled_indeces]

        print(f"Dataset shape: {self.images.shape}")
        
        self.transform = transform
        self.images_shape = self.images.shape
                
    # Return the number of images in the dataset
    def __len__(self):
        return len(self.images)
    
    # Get the image and label at a given index
    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image