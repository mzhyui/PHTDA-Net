from .model_analysis_nets import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import random
import math
import copy
from .topoloss import getTopoLoss
import numpy as np

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def mergeSublistsWithSharedItems(data):
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
    
def overlay(im, a, x):
    multi = 10
    N = im.shape[0]

    x_ = torch.sigmoid(x*multi)
    a_ = torch.sigmoid(a*multi)

    transform = []
    for _ in range(N):
        # scale
        sz = float(torch.Tensor(1).uniform_(0.8, 1.2))
        # smol rotation
        theta = float(torch.Tensor(1).uniform_(-3.14/6, 3.14/6))
        # 5% imsz offset
        pad = 0.3
        offset = (torch.Tensor(2).uniform_(-pad, pad)).tolist()
        transform.append(torch.Tensor([[sz*math.cos(theta), -sz*math.sin(
            theta), offset[0]], [sz*math.sin(theta), sz*math.cos(theta), offset[1]]]))

    transform = torch.stack(transform, dim=0)
    grid = F.affine_grid(transform, im.size(), align_corners=True).cuda()
    # Synthesize trigger
    a_ = F.grid_sample(a_.repeat(N, 1, 1, 1), grid, align_corners=True)
    x_ = F.grid_sample(x_.repeat(N, 1, 1, 1), grid, align_corners=True)

    im_edit = (1-a_)*im+a_*x_
    return im_edit

def initDataset(model : str = None):
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
    return model1, test_loader, dataset_test


def getTotalLength(lst):
    total_length = 0

    for item in lst:
        if isinstance(item, list):
            total_length += getTotalLength(item)
        else:
            total_length += 1

    return total_length


def getSamples(dataset_test = None):
    num_classes = len(dataset_test.classes)
    sample_list = [[] for _ in range(num_classes)]
    # print(len(sample_list))

    # Count the number of samples for each class
    class_count = [0] * 10
    if not dataset_test:
        raise Exception("dataset_test is none")
    # Iterate over the test dataset
    for data, target in dataset_test:
        # Check if we have collected enough samples for each class
        if all(count >= 5 for count in class_count):
            break
        
        # Get the class index
        class_index = target
        
        # Check if we have already collected 5 samples for this class
        if class_count[class_index] < 5:
            # Add the sample to the list
            sample_list[class_index].append(data)
            
            # Increment the count for this class
            class_count[class_index] += 1

    # Print the number of samples collected for each class
    # print(class_count)

    # Shuffle the sample list (optional)
    # import random
    # random.shuffle(sample_list)
    return sample_list

def getGradients(modelpth : str = "", model: torch.nn.Module = None, dataloader : torch.utils.data.DataLoader = None ):
    if not model and modelpth and dataloader:
        raise Exception("None input")
    
    model1, test_loader = copy.deepcopy(model), dataloader
    model1.load_state_dict(torch.load(modelpth))
    model1.eval()
    
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

def getTopofeature(modelpth : str = "", model: torch.nn.Module = None, dataloader : torch.utils.data.DataLoader = None, dataset = None):
    if not model and modelpth and dataloader and dataset:
        raise Exception("None input") 
    
    model, test_loader = copy.deepcopy(model), dataloader
    model.load_state_dict(torch.load(modelpth))
    sample_list = getSamples(dataset_test=dataset)

    scores_pred = torch.zeros(len(sample_list), len(sample_list))

    topo_vector = []
    for class_id, sample_class in enumerate(sample_list):
        input_sample = torch.stack(sample_class).to(device)
        scores = F.log_softmax(model(input_sample), dim=1).data.cpu()
        scores_pred[class_id, :] = scores.mean(0)
        

        triggers = []
        # print(input_sample.shape)
        h = input_sample.shape[2]
        w = input_sample.shape[3]

        a = torch.Tensor(1, 1, h, w).uniform_(-0.01, 0.01)-0.1
        content = torch.Tensor(1, 3, h, w).uniform_(-0.01, 0.01)
        a = a.cuda().requires_grad_()
        content = content.cuda().requires_grad_()
        opt = optim.Adam([a, content], lr=0.001, betas=(0.5, 0.7))

        origin = copy.deepcopy([a, content])
        for k in range(10):
            opt.zero_grad()
            im_edit = overlay(input_sample, a, content)

            scores = F.softmax(model(im_edit), dim=1)
            loss_target = scores[:, class_id].mean()

            mask = torch.sigmoid(a * 10)[0][0]
            loss_topo = getTopoLoss(-mask)

            # print(loss_topo)

            loss = loss_target + loss_topo
            # print(loss.item())

            loss.backward()
            opt.step()

            topo_vector.append(loss.item())

        # print(origin[0] - a, origin[1] - content)
    return np.array(topo_vector)

def extractWeights(model):
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            if hasattr(param, 'is_quantized') and param.is_quantized:
                # 提取整数表示
                weights.append(param.flatten().int_repr().tolist())
            else:
                # 对于非量化张量，可以直接添加原始张量
                # weights.append(param.flatten().tolist())
                pass
    return weights


def minimizeProduct(k):
    # Initialize minimum product as k (maximum possible product of 1 and k itself)
    min_product = k
    best_factors = (1, k)
    
    # Iterate over possible factors
    for i in range(1, int(math.sqrt(k)) + 1):
        if k % i == 0:  # If i is a factor
            j = k // i  # Other factor
            # As we are looking for the closest factor pair, they will be the optimal solution
            if abs(i-j) < abs(best_factors[0]-best_factors[1]):
                min_product = i * j
                best_factors = (i, j)
                
    return best_factors, min_product