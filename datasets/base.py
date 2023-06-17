from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import nibabel as nib
import torch

class NiftyImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), flip=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.flip = flip
        self.to_normal = to_normal # 是否归一化到[-1, 1]

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length
    

    def __getitem__(self, index):
        # Keep the code that handles dataset indexing
        if index >= self._length:
            index = index - self._length

        # Replace the image loading code with code that loads NIfTI files
        nii = nib.load(self.image_paths[index])
        image = nii.get_fdata()
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)
        print(image.shape)
        # If your NIfTI data is not already in the range [-1, 1], you might need to normalize it
        if self.to_normal:
            image = (image - image.mean()) / (image.max() - image.min()) 

        # Keep the code that gets the image name
        image_name = Path(self.image_paths[index]).stem

        return image, image_name
    
class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), flip=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.flip = flip
        self.to_normal = to_normal # 是否归一化到[-1, 1]

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return image, image_name

