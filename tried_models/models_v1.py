## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        #(1 x 224 x 224) -> (32 x 222 x 222)
        self.conv1 = nn.Conv2d(1, 32, 3)
        
        #(32 x 222 x 222) -> (32 x 111 x 111)
        self.pool_2_2 = nn.MaxPool2d(2,2)
        
        # batch normalization
        self.batchNorm1 = nn.BatchNorm2d(32)
        
        #(32 x 111 x 111) -> (64 x 109 x 109)
        self.conv2 = nn.Conv2d(32,64,3)
        
        #(64 x 111 x 111) -> (64 x 55 x 55)
        self.pool2_2 = nn.MaxPool2d(2,2)
        
        # batch normalization
        self.batchNorm2 = nn.BatchNorm2d(64)
        
        
        #(64 x 55 x 55) -> (128 x 53 x 53)
        self.conv3 = nn.Conv2d(64,128,3)
        
        #(128 x 53 x 53) -> (128 x 27 x 27)
        # self.pool2_2 = nn.MaxPool2d(2,2)
        # batch normalization
        self.batchNorm3 = nn.BatchNorm2d(128)
        
        #(128 x 27 x 27) -> (256 x 25 x 25)
        self.conv4 = nn.Conv2d(128,256,3)
        
        #(256 x 25 x 25) -> (256 x 13 x 13)
        # self.pool2_2 = nn.MaxPool2d(2,2)
        # batch normalization
        self.batchNorm4 = nn.BatchNorm2d(256)
        
        self.convDropOut = nn.Dropout(p=0.5)
     
        self.fc1 = nn.Linear(256*12*12,2048)
        self.fc1_dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(2048,1024)
        self.fc2_dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(1024, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.batchNorm1( self.pool_2_2(F.relu(self.conv1(x))) )
        x = self.convDropOut(x)
        x = self.batchNorm2( self.pool_2_2(F.relu(self.conv2(x))) )
        x = self.convDropOut(x)
        x = self.batchNorm3(  self.pool_2_2(F.relu(self.conv3(x))) )
        x = self.convDropOut(x)
        x = self.batchNorm4( self.pool_2_2(F.relu(self.conv4(x))) )
        x = self.convDropOut(x)
        
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc1_dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_dropout(x)
        x = self.fc3(x)        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
