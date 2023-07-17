import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.tensorboard import SummaryWriter

import random
import os
import datetime

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        # self.fc2 = nn.Linear(28*28, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # in_features = 28 * 28
        # x = x.view(-1, in_features)
        # x = self.fc2(x)

        # normal return:
        return F.log_softmax(x, dim=1)
        # soft max is used for generate SDT data
        # return F.softmax(x, dim=1)

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.fc14 = nn.Linear(512*4*4, 1024)
        self.drop1 = nn.Dropout()
        self.fc15 = nn.Linear(1024, 128)
        self.drop2 = nn.Dropout()
        self.fc16 = nn.Linear(128, 10)

        # self.weight_keys = [['fc3.weight', 'fc3.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc1.weight', 'fc1.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['conv1.weight', 'conv1.bias'],
        #                     ]

        # self.weight_keys = [['conv1.weight', 'conv1.bias'],
        #                     ['conv2.weight', 'conv2.bias'],
        #                     ['fc2.weight', 'fc2.bias'],
        #                     ['fc3.weight', 'fc3.bias'],
        #                     ['fc1.weight', 'fc1.bias'],
        #                     ]

        self.weight_keys = [['conv1.weight', 'conv1.bias'], ['conv2.weight', 'conv2.bias'], ['conv3.weight', 'conv3.bias'], ['conv4.weight', 'conv4.bias'], ['conv5.weight', 'conv5.bias'], ['conv6.weight', 'conv6.bias'], ['conv7.weight', 'conv7.bias'], ['conv8.weight', 'conv8.bias'], ['conv9.weight', 'conv9.bias'], ['conv10.weight', 'conv10.bias'], ['conv11.weight', 'conv11.bias'], ['conv12.weight', 'conv12.bias'], ['conv13.weight', 'conv13.bias'],
                            ['fc14.weight', 'fc14.bias'],
                            ['fc15.weight', 'fc15.bias'],
                            ['bn1.weight', 'bn1.bias'],
                            ['bn2.weight', 'bn2.bias'], ['bn3.weight', 'bn3.bias'], [
                                'bn4.weight', 'bn4.bias'], ['bn5.weight', 'bn5.bias'],
                            ['fc16.weight', 'fc16.bias']
                            ]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(-1, 512*4*4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return F.log_softmax(x, dim=1)


class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 25, kernel_size=3),
            torch.nn.BatchNorm2d(25),
            torch.nn.ReLU(inplace=True)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(25, 50, kernel_size=3),
            torch.nn.BatchNorm2d(50),
            torch.nn.ReLU(inplace=True)
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(50 * 5 * 5, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)



class LeNet_tail(nn.Module):
    def __init__(self):
        super(LeNet_tail, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

#%%
layerNo = 0

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        global layerNo

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        mid = self.shortcut(x)
        out += mid
        now = datetime.datetime.now()
        formatted_date_time = now.strftime("%Y-%m-%d %H:%M")
        layerNo += 1
        writer = SummaryWriter(os.path.join('./log', "shortcut", formatted_date_time))
        writer.add_histogram("shortcut"+str(layerNo % 9), mid)
        writer.close()
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    global layerNo
    layerNo=0

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16


        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet20(args=None):
    return ResNet(BasicBlock, [3, 3, 3])