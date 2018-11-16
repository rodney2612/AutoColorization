from torchvision import datasets, transforms
from skimage.color import rgb2lab, rgb2gray
import torch
import numpy as np
#import matplotlib.pyplot as plt

scale_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    #transforms.ToTensor()
])


'''
    Loads the images and their class labels from the given folder during training
'''
class TrainImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        '''
            Path will contain images and target will contain labels of the images.
        '''
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img_original = self.transform(img)
            img_original = np.asarray(img_original)
            img_lab = rgb2lab(img_original)
            '''
                Mapping a and b values to between 0 and 1
            '''
            img_lab = (img_lab + 128) / 255
            '''
                Removing the luminance parameter as it does not contain any color information.
            '''
            img_ab = img_lab[:, :, 1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1)))
            '''
                We create our input gray scale image from the colorised image in the dataset.
            '''
            img_original = rgb2gray(img_original)
            img_original = torch.from_numpy(img_original)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img_original, img_ab), target



'''
    Loads the images and their class labels from the given folder during validation
'''
class ValImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        img_scale = img.copy()
        img_original = img
        img_scale = scale_transform(img_scale)

        img_scale = np.asarray(img_scale)
        img_original = np.asarray(img_original)

        img_scale = rgb2gray(img_scale)
        img_scale = torch.from_numpy(img_scale)
        img_original = rgb2gray(img_original)
        img_original = torch.from_numpy(img_original)
        return (img_original, img_scale), target
