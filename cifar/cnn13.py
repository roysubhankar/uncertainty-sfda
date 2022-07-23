from genericpath import exists
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, os
import random

class CNN13(nn.Module):
       
    def __init__(self, num_classes=10, bias=True, dropout=0.5):
        super(CNN13, self).__init__()

        #self.gn = GaussianNoise(0.15)
        self.feature_extractor = nn.Sequential(
            ## layer 1-a###
            nn.Conv2d(3, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            ## layer 1-b###
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),

            ## layer 1-c###
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, stride=2, padding=0),

            ## layer 2-a###
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            ## layer 2-b###
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            ## layer 2-c###
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, stride=2, padding=0),

            ## layer 3-a###
            nn.Conv2d(256, 512, 3, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            ## layer 3-b###
            nn.Conv2d(512, 256, 1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            ## layer 3-c###
            nn.Conv2d(256, 128, 1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.AvgPool2d(6, stride=2, padding=0)
        )
        self.fc = nn.Linear(128, num_classes, bias=bias)
        
    def forward(self, x):
        out = x
        out = self.feature_extractor(out)
        out = out.view(-1, 128)
        out = self.fc(out)

        return out
