import torch
from torch import nn 
import numpy as np 
import torch.nn.functional as F

def weights_init(m): 
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear): 
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias, 0)

def new_size_conv(size, kernel, stride=1, padding=0): 
    return np.floor((size + 2*padding - (kernel -1)-1)/stride +1)
    
    
def new_size_max_pool(size, kernel, stride=None, padding=0): 
    if stride == None: 
        stride = kernel
    return np.floor((size + 2*padding - (kernel -1)-1)/stride +1)

def calc_mlleaks_cnn_size(size): 
    x = new_size_conv(size, 5,1,2)
    x = new_size_max_pool(x,2,2)
    x = new_size_conv(x,5,1,2)
    out = new_size_max_pool(x,2,2)
    
    return out

class mlleaks_cnn(nn.Module): 
    def __init__(self, n_in=3, n_out=10, n_hidden=64, size=32): 
        super(mlleaks_cnn, self).__init__()
        
        self.n_hidden = n_hidden 
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(n_in, n_hidden, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(n_hidden), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(n_hidden, 2*n_hidden, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(2*n_hidden), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) 

        fc_feature_size = calc_mlleaks_cnn_size(size)
        self.fc = nn.Linear(int(2*n_hidden * fc_feature_size * fc_feature_size), 128)
        self.output = nn.Linear(2*n_hidden, n_out)
        
    def forward(self, x): 
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.output(x)
        
        return out

class mlleaks_mlp(nn.Module): 
    def __init__(self, n_in=3, n_out=1, n_hidden=64): 
        super(mlleaks_mlp, self).__init__()
        
        self.hidden = nn.Linear(n_in, n_hidden)
        #self.bn = nn.BatchNorm1d(n_hidden)
        self.output = nn.Linear(n_hidden, n_out)
        
    def forward(self, x): 
        x = F.sigmoid(self.hidden(x))
        #x = self.bn(x)
        out = self.output(x)
        #out = F.sigmoid(self.output(x))
        
        return out