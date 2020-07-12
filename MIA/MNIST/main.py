import models
from train import *
import torch
import torchvision 
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim


lr = 0.001
batch_size = 64
k=3
n_epochs = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

target_net_type = models.mlleaks_cnn
shadow_net_type = models.mlleaks_cnn

train_transform = torchvision.transforms.Compose([
    #torchvision.transforms.Pad(2),
    

    #torchvision.transforms.RandomRotation(10),
    #torchvision.transforms.RandomHorizontalFlip(),
    #torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = torchvision.transforms.Compose([
    #torchvision.transforms.Pad(2),
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    torchvision.transforms.Normalize((0.1307,), (0.3081,))

])

# load training set 

mnist_trainset = torchvision.datasets.MNIST('../../../Datasets/', train=True, transform=train_transform, download=True)
mnist_trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# load test set 

mnist_testset = torchvision.datasets.MNIST('../../../Datasets/', train=False, transform=test_transform, download=True)
mnist_testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=32, shuffle=False, num_workers=2)


total_size = len(mnist_trainset)
split1 = total_size // 4
split2 = split1*2
split3 = split1*3

indices = list(range(total_size))
shadow_train_idx = indices[:split1]
shadow_out_idx = indices[split1:split2]
target_train_idx = indices[split2:split3]
target_out_idx = indices[split3:]

shadow_train_sampler = SubsetRandomSampler(shadow_train_idx)
shadow_out_sampler = SubsetRandomSampler(shadow_out_idx)
target_train_sampler = SubsetRandomSampler(target_train_idx)
target_out_sampler = SubsetRandomSampler(target_out_idx)

shadow_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, sampler=shadow_train_sampler, num_workers=1)
shadow_out_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, sampler=shadow_out_sampler, num_workers=1)
target_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, sampler=target_train_sampler, num_workers=1)
target_out_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, sampler=target_out_sampler, num_workers=1)

target_net = target_net_type().to(device)
target_net.apply(models.weights_init)
target_loss = nn.CrossEntropyLoss()
target_optim = optim.Adam(target_net.parameters(), lr=lr)

shadow_net = shadow_net_type().to(device)
shadow_net.apply(models.weights_init)
shadow_loss = nn.CrossEntropyLoss()
shadow_optim = optim.Adam(shadow_net.parameters(), lr=lr)

attack_net = models.mlleaks_mlp(n_in=k).to(device)
attack_net.apply(models.weights_init)
attack_loss = nn.BCELoss()
attack_optim = optim.Adam(attack_net.parameters(), lr=lr)


train(shadow_net, shadow_train_loader, mnist_testloader, shadow_optim, shadow_loss, n_epochs)
train_attacker(attack_net, shadow_net, shadow_train_loader, shadow_out_loader, attack_optim, attack_loss, n_epochs=1, k=k)
train(target_net, target_train_loader, mnist_testloader, target_optim, target_loss, n_epochs)
eval_attack_net(attack_net, target_net, target_train_loader, target_out_loader, k)
