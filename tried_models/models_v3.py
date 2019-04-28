## TODO: define the convolutional neural network architecture

import torch
import torch.cuda
#from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I



def gpu(x):
    return x.cuda() if torch.cuda.is_available() else x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # (3 x 224 x 224) --> (32 x 74 x 74)
        # 74 = (224-7+2)/3 +1
        self.conv1 = nn.Conv2d(1, 32, 7, 3, 1)
        
        # (32 x 74 x 74) --> (64 x 24 x 24)
        # 74 = (74-5)/3 +1
        self.conv2 = nn.Conv2d(32, 64, 5, 3, 0)
        
        # (64 x 24 x 24) --> (128 x 8 x 8)
        # 8 = (24-5+2)/3 +1
        self.conv3 = nn.Conv2d(64, 128, 5, 3, 1)
        
        # (128 x 8 x 8) --> (256 x 6 x 6)
        # 6 = (8-3)/1 +1        
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 0)
        
        # (256 x 6 x 6) --> (256 x 4 x 4)
        # 4 = (6-3)/1 +1
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 0)
        
        # (256 x 4 x 4) --> (512 x 4 x 4)
        # 4 = (4-1)/1 +1
        self.conv6 = nn.Conv2d(512, 512, 1, 1, 0)
        
        self.fc1 = nn.Linear(512*4*4, 1024)
        self.fc1_drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc2_drop = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(1024, 136)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # Apply convolutional layers
        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        x = F.selu(self.conv3(x))
        x = F.selu(self.conv4(x))
        x = F.selu(self.conv5(x))
        x = F.selu(self.conv6(x))

        # Flatten and continue with dense layers
        x = x.view(x.size(0), -1)
        x = F.selu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.selu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x