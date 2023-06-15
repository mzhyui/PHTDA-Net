from .model_analysis_nets import *
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
dataset_test = datasets.CIFAR10('data/cifar10/', train=False, download=True, transform=trans_cifar10_val)
test_loader = torch.utils.data.DataLoader(
                dataset_test, batch_size=32,
                num_workers=2, pin_memory=True, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def merge_sublists_with_shared_items(data):
    merged = True
    while merged:
        merged = False
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if set(data[i]).intersection(data[j]):
                    data[i] = list(set(data[i]).union(data[j]))
                    del data[j]
                    merged = True
                    break
            if merged:
                break
    return data
    
def initDataset(model : str = "lenet"):
    if not model:
        return {}
    if model == "lenet":
        model1 = LeNet().to(device)
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
        test_loader = torch.utils.data.DataLoader(
                        dataset_test, batch_size=32,
                        num_workers=2, pin_memory=True, shuffle=False)
    elif model == "vgg":
        model1 = VGG16().to(device)
        dataset_test = datasets.CIFAR10('data/cifar10/', train=False, download=True, transform=trans_cifar10_val)
        test_loader = torch.utils.data.DataLoader(
                        dataset_test, batch_size=32,
                        num_workers=2, pin_memory=True, shuffle=False)

    elif model == "resnet":
        model1 = ResNet20().to(device)
        dataset_test = datasets.CIFAR10('data/cifar10/', train=False, download=True, transform=trans_cifar10_val)
        test_loader = torch.utils.data.DataLoader(
                        dataset_test, batch_size=32,
                        num_workers=2, pin_memory=True, shuffle=False)
    return (model1, test_loader)

def getGradients(modelpth : str = "", modelAndDataloader : list = [], ):
    if not modelAndDataloader and modelpth:
        return {}
    
    model1, test_loader = modelAndDataloader
                        
    model1.load_state_dict(torch.load(modelpth))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model1.parameters(), lr=0.01, momentum=0.5)

    for X, Y in test_loader:
        X_test = X.to(device)
        Y_test = Y.to(device)
        # print(Y)
        break

    # Forward pass
    outputs = model1(X_test)

    # Compute the loss
    loss = criterion(outputs, Y_test)

    # Zero the gradients
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Access the gradients
    gradients1 = {}
    for name, param in model1.named_parameters():
        gradients1[name] = param.grad
    return gradients1

def get_total_length(lst):
    total_length = 0

    for item in lst:
        if isinstance(item, list):
            total_length += get_total_length(item)
        else:
            total_length += 1

    return total_length