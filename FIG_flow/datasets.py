from glob import glob
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset_name: Dataset,
                   root: str,
                   batch_size: int, 
                   shuffle: bool,
                   num_workers: int, 
                   drop_last: bool):
    dataset = get_dataset(dataset_name, root)
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=shuffle, 
                            num_workers=num_workers, 
                            drop_last=drop_last)
    return dataloader

@register_dataset(name="celeba")
class celeba_hq_dataset(Dataset):
    """CelebA HQ dataset."""

    def __init__(self, root):
        self.fpaths = sorted(glob(root + '/**/*', recursive=True))
        self.transform = transforms.Compose([
            CenterCropLongEdge(),
            transforms.Resize(256),
            transforms.ToTensor(), 
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, idx):
        fpath = self.fpaths[idx]
        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)
        return img

@register_dataset(name="lsun_bedroom")
class lsun_bedroom_dataset(Dataset):
    """LSUN bedroom dataset."""

    def __init__(self, root):
        self.fpaths = sorted(glob(root + '/**/*', recursive=True))
        self.transform = transforms.Compose([
            CenterCropLongEdge(),
            transforms.Resize(256),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, idx):
        fpath = self.fpaths[idx]
        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)
        return img

@register_dataset(name="afhq_cat")
class afhq_cat_dataset(Dataset):
    """AFHQ cat dataset."""

    def __init__(self, root, transform=None):
        self.fpaths = sorted(glob(root + '/**/*', recursive=True))
        self.transform = transforms.Compose([
            CenterCropLongEdge(),
            transforms.Resize(256),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, idx):
        fpath = self.fpaths[idx]
        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)
        return img


class CenterCropLongEdge(object):
    """Crops the given PIL Image on the long edge.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__
