import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import random
from torchvision.io import read_image
from torchvision.transforms import transforms

from tqdm import tqdm


class LIDC_CROPS(Dataset):
    images = []
    labels = []

    def __init__(self, dataset_location, folder_type=True, transform=None, masks_count=4, img_size=128):
        print("Start loading")
        self.transform = transform
        self.masks_count = masks_count
        self.img_size = img_size
        if folder_type:
            self.load_from_folder(dataset_location)
        else:
            self.images = torch.load(dataset_location + "/images.pt")
            self.labels = torch.load(dataset_location + "/labels.pt")

        assert (len(self.images) == len(self.labels))
        assert torch.max(torch.stack(self.images)) <= 1 and torch.min(torch.stack(self.images)) >= 0
        assert torch.max(torch.stack(self.labels)) <= 1 and torch.min(torch.stack(self.labels)) >= 0
        assert torch.max(torch.stack(self.labels)) <= 1 and torch.min(torch.stack(self.labels)) >= 0

    def load_from_folder(self, dataset_location):
        images_path = dataset_location + "images/"
        gt_path = dataset_location + "gt/"

        resizer = transforms.Resize(self.img_size)

        for patient in tqdm(os.listdir(images_path)):
            patient_images_path = images_path + f"{patient}/"
            patient_gt_path = gt_path + f"{patient}/"
            for image in os.listdir(patient_images_path):
                entry_name = image[:-4]
                image = read_image(patient_images_path + image).type(torch.FloatTensor) / 255
                image = resizer(image)
                self.images.append(image.type(torch.float))
                masks = []
                for gt in os.listdir(patient_gt_path):
                    if gt.startswith(entry_name):
                        gt = read_image(patient_gt_path + gt).type(torch.FloatTensor) / 255
                        gt = resizer(gt)
                        masks.append(gt.type(torch.float))
                self.labels.append(torch.stack(masks))

    def __getitem__(self, index):
        image = self.images[index]

        # Randomly select one of the four labels for this image
        label = self.labels[index][random.randint(0, self.masks_count-1)]
        if self.transform is not None:
            image = self.transform(image)

        # Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        return image, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)
