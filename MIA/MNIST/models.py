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
    def __init__(self):
        super(mlleaks_cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

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