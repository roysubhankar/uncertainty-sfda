import torch
from torchvision import datasets
from torchvision import transforms as transform_lib
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid

from PIL import Image

class CIFAR9(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR9, self).__init__(root, train=train, download=download, transform=None, target_transform=None)

        self.transform = transform
        self.target_transform = target_transform

        # manual target transform and sample exclusion
        self.targets = np.asarray(self.targets)
        idxs_valid = np.where(self.targets != 6)[0] # find all the indices which are not the class 'frog'

        self.data = self.data[idxs_valid, :, :, :]
        self.targets = self.targets[idxs_valid]

        # shift one label index ahead due to exclusion of class 'frog'
        np.place(self.targets, self.targets == 7, 6)
        np.place(self.targets, self.targets == 8, 7)
        np.place(self.targets, self.targets == 9, 8)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, index

    def __len__(self):
        return len(self.data)

class STL9(datasets.STL10):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=True):
        super(STL9, self).__init__(root, split=split, download=download, transform=None, target_transform=None)

        self.transform = transform
        self.target_transform = target_transform

        # manual target transform and sample exlcusion
        self.labels = np.asarray(self.labels)
        idxs_valid = np.where(self.labels != 7)[0] # find all the indices which are not the class 'monkey'

        self.data = self.data[idxs_valid, :, :, :]
        self.labels = self.labels[idxs_valid]

        # shift one label index ahead due to exclusion of class 'monkey'
        np.place(self.labels, self.labels == 8, 7)
        np.place(self.labels, self.labels == 9, 8)

        # arrange the labels of stl corresponding to cifar10 labels
        # classes bird and car need to exchanged
        np.place(self.labels, self.labels == 1, 10) # temporarily change labels of class 1 (bird) to 10
        np.place(self.labels, self.labels == 2, 1) # change labels of class 2 (car) to 1
        np.place(self.labels, self.labels == 10, 2) # change back the labels from 10 to 2
    
    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, index
    
    def __len__(self):
        return self.data.shape[0]

def prepare_transform(dataset):
    # mean and std
    mean, std = {
        'CIFAR9': [(0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)],
        'STL9': [(0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)],
        'SVHN': [(0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)],
    }[dataset]

    # target transforms
    target_transform = None
    if dataset == 'CIFAR9':
        cifar10_to_cifar9 = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 7:6, 8:7, 9:8}
        target_transform = lambda t: cifar10_to_cifar9[t]
    elif dataset == 'STL9':
        stl10_to_stl9 = {-1:-1, 0:0, 1:2, 2:1, 3:3, 4:4, 5:5, 6:6, 8:7, 9:8}
        target_transform = lambda t: stl10_to_stl9[t]
    
    train_transform = {
        'CIFAR9': transform_lib.Compose([
            transform_lib.RandomResizedCrop(32, (0.6, 1.0)),
            transform_lib.RandomHorizontalFlip(),
            transform_lib.ToTensor(),
            transform_lib.Normalize(mean, std)
        ]),
        'STL9': transform_lib.Compose([
            transform_lib.RandomResizedCrop(32, (0.6, 1.0)),
            transform_lib.RandomHorizontalFlip(),
            transform_lib.ToTensor(),
            transform_lib.Normalize(mean, std)
        ]),
        'SVHN': transform_lib.Compose([
            transform_lib.RandomResizedCrop(32, (0.6, 1.0)),
            transform_lib.ToTensor(),
            transform_lib.Normalize(mean, std)
        ])
    }[dataset]

    val_transform = {
        'CIFAR9': transform_lib.Compose([
            transform_lib.ToTensor(),
            transform_lib.Normalize(mean, std)
        ]),
        'STL9': transform_lib.Compose([
            transform_lib.Resize(32),
            transform_lib.ToTensor(),
            transform_lib.Normalize(mean, std)
        ]),
        'SVHN': transform_lib.Compose([
            transform_lib.Resize(32),
            transform_lib.ToTensor(),
            transform_lib.Normalize(mean, std)
        ])
    }[dataset]

    return train_transform, val_transform, target_transform

def denorm(x):
	out = (x + 1) / 2
	return out.clamp_(0, 1)

def visualize_samples(imgs, nrow=8, padding=2):
    save_image(make_grid(denorm(imgs), nrow=nrow), './logs/demo.png')
    return

def trainval_loader(args, dset):
    train_transform, val_transform, _ = prepare_transform(dset)
    if dset == 'CIFAR9':
        # train transform for the train dataset
        tr_dset = CIFAR9(root=args.root_dir, train=True, transform=train_transform,
                      target_transform=None, download=True)
        # val transform for the laplace dataset
        lap_dset = CIFAR9(root=args.root_dir, train=True, transform=val_transform,
                      target_transform=None, download=True)
        # val transform for the validation dataset
        val_dset = CIFAR9(root=args.root_dir, train=False, transform=val_transform,
                      target_transform=None, download=True)
    elif dset == 'STL9':
        # train transform for the train dataset
        tr_dset = STL9(root=args.root_dir, split='train', transform=train_transform,
                      target_transform=None, download=True)
        # val transform for the laplace dataset
        lap_dset = STL9(root=args.root_dir, split='train', transform=val_transform,
                      target_transform=None, download=True)
        # val transform for the validation dataset
        val_dset = STL9(root=args.root_dir, split='test', transform=val_transform,
                      target_transform=None, download=True)

    
    train_loader = torch.utils.data.DataLoader(
                        dataset=tr_dset,
                        shuffle=True,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        pin_memory=False,
                        drop_last=False
                    )
    
    lap_loader = torch.utils.data.DataLoader(
                        dataset=lap_dset,
                        shuffle=False,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        pin_memory=False,
                        drop_last=False
                    )
    
    val_loader = torch.utils.data.DataLoader(
                        dataset=val_dset,
                        shuffle=False,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        pin_memory=False,
                        drop_last=False
                    )

    #iter_dset = iter(val_loader)
    #batch = iter_dset.next()
    #print(batch[0].shape)
    #print(batch[1])
    #print(len(tr_dset))
    #print(len(lap_dset))
    #print(len(val_dset))
    #visualize_samples(batch[0])

    return train_loader, lap_loader, val_loader
