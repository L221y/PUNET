import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import random
import cv2
from torchvision.io import read_image


class LIDC_CROPS(Dataset):
    images = []
    labels = []

    def __init__(self, dataset_location, transform=None, masks_count=4):
        print("Start loading")
        self.transform = transform
        self.masks_count = masks_count

        images_path = dataset_location + "/images/"
        gt_path = dataset_location + "/gt/"

        for patient in os.listdir(images_path):
            patient_images_path = images_path + f"{patient}/"
            patient_gt_path = gt_path + f"{patient}/"
            for image in os.listdir(patient_images_path):
                entry_name = image[:-4]
                image=read_image(patient_images_path + image).type(torch.float) / 255
                image=image.numpy()
                image=np.squeeze(image,axis=0)
                image = cv2.resize(image, dsize=(128,128),interpolation=cv2.INTER_LINEAR)
                #print(image.shape)
                #image = np.mean(image, axis=-1, keepdims=False)
                image=torch.from_numpy(image)
                self.images.append(image.type(torch.float))
                masks = []
                for gt in os.listdir(patient_gt_path):
                    if gt.startswith(entry_name):
                        gt=read_image(patient_gt_path + gt).type(torch.float) / 255
                        gt=gt.numpy()
                        gt=np.squeeze(gt,axis=0)
                        gt = cv2.resize(gt, dsize=(128,128),interpolation=cv2.INTER_LINEAR)
                        #gt = np.mean(gt, axis=-1, keepdims=False)
                        #Sprint(gt.shape)
                        gt=torch.from_numpy(gt)
                        masks.append(gt.type(torch.float))
                self.labels.append(torch.stack(masks))
            #self.images.append(image_patient)
            #self.labels.append(labels_patient)


        assert (len(self.images) == len(self.labels))
        assert torch.max(torch.stack(self.images)) <= 1 and torch.min(torch.stack(self.images)) >= 0
        assert torch.max(torch.stack(self.labels)) <= 1 and torch.min(torch.stack(self.labels)) >= 0


    def __getitem__(self, index):
        image = np.expand_dims(self.images[index], axis=0)

        # Randomly select one of the four labels for this image
        label = self.labels[index][random.randint(0, self.masks_count-1)].type(torch.float)
        if self.transform is not None:
            image = self.transform(image)

        # Convert image and label to torch tensors
        image = torch.from_numpy(image)
        #label = torch.from_numpy(label)

        # Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        #label=label[0]
        #image=image[3]


        return image, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)
