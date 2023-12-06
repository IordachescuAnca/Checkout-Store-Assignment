import os
import torch
import torchvision
import dataset
import numpy as np
from torch.utils.data import Dataset
import PIL.Image
from torchvision import transforms


def make_transform(is_train = True, is_inception = False):
    resnet_transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    resnet_transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])





class PersonReIdDataset(Dataset):

    def __init__(self, root_dir, dataset_type, transform=None):

        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.transform = transform

        bounding_box_train_file = os.path.join(self.root_dir, 'bounding_box_train')
        bounding_box_train_paths = [os.path.join(self.root_dir, 'bounding_box_train', x) for x in os.listdir(bounding_box_train_file)]
        bounding_box_test_file = os.path.join(self.root_dir, 'bounding_box_test')
        bounding_box_test_paths = [os.path.join(self.root_dir, 'bounding_box_test', x) for x in os.listdir(bounding_box_test_file)]
        self.total_image_paths = bounding_box_train_paths + bounding_box_test_paths


        if dataset_type == 'train':
            self.unique_labels = list(range(0, 1000))
        elif dataset_type == 'val':
            self.unique_labels = list(range(1000, 1501))

        self.total_labels = [image_path.split('/')[-1].split('_')[0] for image_path in self.total_image_paths]
        
        self.image_paths = []
        self.labels = []
        for (image_path, label) in zip(self.total_image_paths, self.total_labels):
            if int(label) in self.unique_labels:
                self.image_paths.append(image_path)
                self.labels.append(int(label))


    def get_unique_classes(self):
        return len(set(self.unique_labels))

    
    def __len__(self):
        return len(self.labels)
    

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = PIL.Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[index]

        return image, label
