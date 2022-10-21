import torch.cuda
from torch.utils.data import DataLoader, Dataset
import numpy as np
import csv
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

class simple_model(nn.Module):
    def __init__(self,init_width=4,dropout_rate=0.4):
        super(simple_model,self).__init__()

        self.init_width   = init_width
        self.dropout_rate = dropout_rate

        self.init_layers()


    def init_layers(self):
        #input size (28,28)
        self.conv1_1 = nn.Conv2d(in_channels=1,out_channels=self.init_width,kernel_size=3,stride=1,padding=1)
        #num weights= 3*4*1=12
        self.conv1_2 = nn.Conv2d(in_channels=self.init_width,out_channels=self.init_width,kernel_size=3,stride=2,
                                 padding=1)
        #num weights= 3*4*4=48, num biases = 4
        #size (14,14,4)

        #self.batch1 = nn.BatchNorm2d(num_features=self.init_width)
        #input size is halved (14,14)
        self.conv2_1 = nn.Conv2d(in_channels=self.init_width,out_channels=self.init_width*2,kernel_size=3,stride=2,
                                 padding=1)
        #num weights= 3*4*8=96 num biases = 8
        self.conv2_2 = nn.Conv2d(in_channels=self.init_width*2,out_channels=self.init_width*2,kernel_size=3,stride=1,
                                 padding=1)
        #num weights= 3*8*8=192 num biases = 8
        #size (14,14,8)
        #input size is halved (7,7)
        self.conv3_1 = nn.Conv2d(in_channels=self.init_width*2,out_channels=self.init_width*4,kernel_size=3,stride=2,
                                 padding=1)
        #num weights= 3*8*16=384 num biases = 16
        self.conv3_2 = nn.Conv2d(in_channels=self.init_width * 4,out_channels=self.init_width * 4,kernel_size=3,
                                 stride=1,
                                 padding=1)
        #size (7,7,16)
        #input size is halved (4,4)
        # self.conv4_1 = nn.Conv2d(in_channels=self.init_width * 4,out_channels=self.init_width*8 ,kernel_size=3,
        #                          stride=2,
        #                          padding=1)
        # self.conv4_2 = nn.Conv2d(in_channels=self.init_width * 8,out_channels=self.init_width * 4,kernel_size=3,
        #                          stride=1,
        #                          padding=1)
        #size (4,4,16)
        self.fc1  = nn.Linear(in_features=self.init_width * 4 * 4*4,out_features=self.init_width*4*4)
        self.fc2  = nn.Linear(in_features=self.init_width * 4 * 4 ,out_features=self.init_width * 4 )
        self.out  = nn.Linear(in_features=self.init_width * 4,out_features=10)

        self.dropout = nn.Dropout(self.dropout_rate)
        #self.fc5  = nn.Linear(in_features=)
        #size (7,7,1)

    def forward(self,z):
        z = F.relu(self.conv1_1(z))
        z = F.relu(self.conv1_2(z))

        z = F.relu(self.conv2_1(z))
        z = F.relu(self.conv2_2(z))

        z = F.relu(self.conv3_1(z))
        z = F.relu(self.conv3_2(z))

        # z = F.relu(self.conv4_1(z))
        # z = F.relu(self.conv4_2(z))
        #
        z = torch.flatten(z,1)
        z = F.relu(self.fc1(z))
        z = self.dropout(z)
        z = F.relu(self.fc2(z))
        #z = self.dropout(z)
        z = self.out(z)
        return z
        #z = F.re




class mnist_model(nn.Module):
    def __init__(self,init_width=4,dropout_rate=0.4):
        super(mnist_model,self).__init__()

        self.init_width   = init_width
        self.dropout_rate = dropout_rate

        self.init_layers()


    def init_layers(self):
        #input size (28,28)
        self.conv1_1 = nn.Conv2d(in_channels=1,out_channels=self.init_width,kernel_size=3,stride=1,padding=1)
        #num weights= 3*4*1=12
        self.conv1_2 = nn.Conv2d(in_channels=self.init_width,out_channels=self.init_width,kernel_size=3,stride=2,
                                 padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_features=self.init_width)
        #num weights= 3*4*4=48, num biases = 4
        #size (14,14,4)

        #self.batch1 = nn.BatchNorm2d(num_features=self.init_width)
        #input size is halved (14,14)
        self.conv2_1 = nn.Conv2d(in_channels=self.init_width,out_channels=self.init_width*2,kernel_size=3,stride=2,
                                 padding=1)
        #num weights= 3*4*8=96 num biases = 8
        self.conv2_2 = nn.Conv2d(in_channels=self.init_width*2,out_channels=self.init_width*2,kernel_size=3,stride=1,
                                 padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_features=self.init_width*2)
        #num weights= 3*8*8=192 num biases = 8
        #size (14,14,8)
        #input size is halved (7,7)
        self.conv3_1 = nn.Conv2d(in_channels=self.init_width*2,out_channels=self.init_width*4,kernel_size=3,stride=2,
                                 padding=1)
        #num weights= 3*8*16=384 num biases = 16
        self.conv3_2 = nn.Conv2d(in_channels=self.init_width * 4,out_channels=self.init_width * 4,kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.conv3_bn = nn.BatchNorm2d(num_features=self.init_width*4)
        #size (7,7,16)
        #input size is halved (4,4)
        # self.conv4_1 = nn.Conv2d(in_channels=self.init_width * 4,out_channels=self.init_width*8 ,kernel_size=3,
        #                          stride=2,
        #                          padding=1)
        # self.conv4_2 = nn.Conv2d(in_channels=self.init_width * 8,out_channels=self.init_width * 4,kernel_size=3,
        #                          stride=1,
        #                          padding=1)
        #size (4,4,16)
        self.fc1  = nn.Linear(in_features=self.init_width * 4 * 4*4,out_features=self.init_width*4*4)
        self.fc1_bn = nn.BatchNorm1d(num_features=self.init_width*4*4)
        self.fc2  = nn.Linear(in_features=self.init_width * 4 * 4 ,out_features=self.init_width * 4 )
        self.fc2_bn = nn.BatchNorm1d(num_features=self.init_width * 4 )
        self.out  = nn.Linear(in_features=self.init_width * 4,out_features=10)

        self.dropout = nn.Dropout(self.dropout_rate)
        #self.fc5  = nn.Linear(in_features=)
        #size (7,7,1)

    def forward(self,z):
        z = F.relu(self.conv1_1(z))
        z = F.relu(self.conv1_bn(self.conv1_2(z)))


        z = F.relu(self.conv2_1(z))
        z = F.relu(self.conv2_bn(self.conv2_2(z)))

        z = F.relu(self.conv3_1(z))
        z = F.relu(self.conv3_bn(self.conv3_2(z)))

        # z = F.relu(self.conv4_1(z))
        # z = F.relu(self.conv4_2(z))
        #
        z = torch.flatten(z,1)
        z = F.relu(self.fc1_bn(self.fc1(z)))
        z = self.dropout(z)
        z = F.relu(self.fc2_bn(self.fc2(z)))
        #z = self.dropout(z)
        z = self.out(z)
        return z


class mnist_model_pool(nn.Module):
    def __init__(self,init_width=4,dropout_rate=0.4):
        super(mnist_model_pool,self).__init__()

        self.init_width   = init_width
        self.dropout_rate = dropout_rate

        self.init_layers()


    def init_layers(self):
        #input size (28,28)
        self.conv1_1 = nn.Conv2d(in_channels=1,out_channels=self.init_width,kernel_size=3,stride=1,padding=1)
        #num weights= 3*4*1=12
        self.conv1_2 = nn.Conv2d(in_channels=self.init_width,out_channels=self.init_width,kernel_size=3,stride=1,
                                 padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_features=self.init_width)
        #num weights= 3*4*4=48, num biases = 4
        #size (14,14,4)

        #self.batch1 = nn.BatchNorm2d(num_features=self.init_width)
        #input size is halved (14,14)
        self.conv2_1 = nn.Conv2d(in_channels=self.init_width,out_channels=self.init_width*2,kernel_size=3,stride=1,
                                 padding=1)
        #num weights= 3*4*8=96 num biases = 8
        self.conv2_2 = nn.Conv2d(in_channels=self.init_width*2,out_channels=self.init_width*2,kernel_size=3,stride=1,
                                 padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_features=self.init_width*2)
        #num weights= 3*8*8=192 num biases = 8
        #size (14,14,8)
        #input size is halved (7,7)
        self.conv3_1 = nn.Conv2d(in_channels=self.init_width*2,out_channels=self.init_width*4,kernel_size=3,stride=1,
                                 padding=1)
        #num weights= 3*8*16=384 num biases = 16
        self.conv3_2 = nn.Conv2d(in_channels=self.init_width * 4,out_channels=self.init_width * 4,kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.conv3_bn = nn.BatchNorm2d(num_features=self.init_width*4)


        self.conv4_1 = nn.Conv2d(in_channels=self.init_width*4,out_channels=self.init_width*8,kernel_size=3,stride=1,
                                 padding=1)
        #num weights= 3*8*16=384 num biases = 16
        self.conv4_2 = nn.Conv2d(in_channels=self.init_width * 8,out_channels=self.init_width * 8,kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.conv4_bn = nn.BatchNorm2d(num_features=self.init_width*8)
        #size (7,7,16)
        #input size is halved (4,4)
        # self.conv4_1 = nn.Conv2d(in_channels=self.init_width * 4,out_channels=self.init_width*8 ,kernel_size=3,
        #                          stride=2,
        #                          padding=1)
        # self.conv4_2 = nn.Conv2d(in_channels=self.init_width * 8,out_channels=self.init_width * 4,kernel_size=3,
        #                          stride=1,
        #                          padding=1)
        #size (4,4,16)
        self.fc1  = nn.Linear(in_features=self.init_width * 4 * 4*8,out_features=self.init_width*4*4)
        self.fc1_bn = nn.BatchNorm1d(num_features=self.init_width*4*4)
        self.fc2  = nn.Linear(in_features=self.init_width * 4 * 4 ,out_features=self.init_width * 4 )
        self.fc2_bn = nn.BatchNorm1d(num_features=self.init_width * 4 )
        self.out  = nn.Linear(in_features=self.init_width * 4,out_features=10)

        self.dropout = nn.Dropout(self.dropout_rate)
        #self.fc5  = nn.Linear(in_features=)
        #size (7,7,1)

    def forward(self,z):
        z = F.relu(self.conv1_1(z))
        z = F.relu(self.conv1_bn(self.conv1_2(z)))

        z = F.max_pool2d(z,kernel_size=2)
        z = F.relu(self.conv2_1(z))
        z = F.relu(self.conv2_bn(self.conv2_2(z)))

        z = F.max_pool2d(z,kernel_size=2)
        z = F.relu(self.conv3_1(z))
        z = F.relu(self.conv3_bn(self.conv3_2(z)))

        z = F.max_pool2d(z,kernel_size=2,padding=1)
        z = F.relu(self.conv4_1(z))
        z = F.relu(self.conv4_bn(self.conv4_2(z)))
        #

        # z = F.relu(self.conv4_1(z))
        # z = F.relu(self.conv4_2(z))
        #
        z = torch.flatten(z,1)
        z = F.relu(self.fc1_bn(self.fc1(z)))
        z = self.dropout(z)
        z = F.relu(self.fc2_bn(self.fc2(z)))
        #z = self.dropout(z)
        z = self.out(z)
        return z


class mnist_model_pool_single_conv(nn.Module):
    def __init__(self,init_width=4,dropout_rate=0.4):
        super(mnist_model_pool_single_conv,self).__init__()

        self.init_width   = init_width
        self.dropout_rate = dropout_rate

        self.init_layers()


    def init_layers(self):
        #input size (28,28)
        self.conv1_1 = nn.Conv2d(in_channels=1,out_channels=self.init_width,kernel_size=3,stride=1,padding=1)
        #num weights= 3*4*1=12
        # self.conv1_2 = nn.Conv2d(in_channels=self.init_width,out_channels=self.init_width,kernel_size=3,stride=1,
        #                          padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_features=self.init_width)
        #num weights= 3*4*4=48, num biases = 4
        #size (14,14,4)

        #self.batch1 = nn.BatchNorm2d(num_features=self.init_width)
        #input size is halved (14,14)
        self.conv2_1 = nn.Conv2d(in_channels=self.init_width,out_channels=self.init_width*2,kernel_size=3,stride=1,
                                 padding=1)
        #num weights= 3*4*8=96 num biases = 8
        # self.conv2_2 = nn.Conv2d(in_channels=self.init_width*2,out_channels=self.init_width*2,kernel_size=3,stride=1,
        #                          padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_features=self.init_width*2)
        #num weights= 3*8*8=192 num biases = 8
        #size (14,14,8)
        #input size is halved (7,7)
        self.conv3_1 = nn.Conv2d(in_channels=self.init_width*2,out_channels=self.init_width*4,kernel_size=3,stride=1,
                                 padding=1)
        #num weights= 3*8*16=384 num biases = 16
        # self.conv3_2 = nn.Conv2d(in_channels=self.init_width * 4,out_channels=self.init_width * 4,kernel_size=3,
        #                          stride=1,
        #                          padding=1)
        self.conv3_bn = nn.BatchNorm2d(num_features=self.init_width*4)


        self.conv4_1 = nn.Conv2d(in_channels=self.init_width*4,out_channels=self.init_width*8,kernel_size=3,stride=1,
                                 padding=1)
        #num weights= 3*8*16=384 num biases = 16
        # self.conv4_2 = nn.Conv2d(in_channels=self.init_width * 8,out_channels=self.init_width * 8,kernel_size=3,
        #                          stride=1,
        #                          padding=1)
        self.conv4_bn = nn.BatchNorm2d(num_features=self.init_width*8)
        #size (7,7,16)
        #input size is halved (4,4)
        # self.conv4_1 = nn.Conv2d(in_channels=self.init_width * 4,out_channels=self.init_width*8 ,kernel_size=3,
        #                          stride=2,
        #                          padding=1)
        # self.conv4_2 = nn.Conv2d(in_channels=self.init_width * 8,out_channels=self.init_width * 4,kernel_size=3,
        #                          stride=1,
        #                          padding=1)
        #size (4,4,16)
        self.fc1  = nn.Linear(in_features=self.init_width * 4 * 4*8,out_features=self.init_width*4*4)
        self.fc1_bn = nn.BatchNorm1d(num_features=self.init_width*4*4)
        self.fc2  = nn.Linear(in_features=self.init_width * 4 * 4 ,out_features=self.init_width * 4 )
        self.fc2_bn = nn.BatchNorm1d(num_features=self.init_width * 4 )
        self.out  = nn.Linear(in_features=self.init_width * 4,out_features=10)

        self.dropout = nn.Dropout(self.dropout_rate)
        #self.fc5  = nn.Linear(in_features=)
        #size (7,7,1)

    def forward(self,z):
        #z = F.relu(self.conv1_1(z))
        z = F.relu(self.conv1_bn(self.conv1_1(z)))

        z = F.max_pool2d(z,kernel_size=2)
        #z = F.relu(self.conv2_1(z))
        z = F.relu(self.conv2_bn(self.conv2_1(z)))

        z = F.max_pool2d(z,kernel_size=2)
        #z = F.relu(self.conv3_1(z))
        z = F.relu(self.conv3_bn(self.conv3_1(z)))

        z = F.max_pool2d(z,kernel_size=2,padding=1)
        #z = F.relu(self.conv4_1(z))
        z = F.relu(self.conv4_bn(self.conv4_1(z)))
        #

        # z = F.relu(self.conv4_1(z))
        # z = F.relu(self.conv4_2(z))
        #
        z = torch.flatten(z,1)
        z = F.relu(self.fc1_bn(self.fc1(z)))
        z = self.dropout(z)
        z = F.relu(self.fc2_bn(self.fc2(z)))
        #z = self.dropout(z)
        z = self.out(z)
        return z


class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        # Image starts as a matrix of size (1, 28, 28)

        # The size of the image after each convolution or pooling layer can be obtained by:
        # output = ((input - kernel_size + (2 * padding)) / stride) + 1

        # Convolutions and batch normalisations
        # Batch norm reduces internal covariate shift
        # Normalises the input feature (subtract batch mean, divide by batch sd)
        # This speeds up neural network training times
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,stride=1,padding=2)  # conv1
        self.conv1_bn = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2)  # conv2
        self.conv2_bn = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)  # conv3
        self.conv3_bn = nn.BatchNorm2d(num_features=128)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=2,stride=1,padding=1)  # conv4
        self.conv4_bn = nn.BatchNorm2d(num_features=256)

        # Fully connected linear layers and batch normalisations
        self.fc1 = nn.Linear(in_features=256 * 4 * 4,out_features=1024)  # linear 1
        self.fc1_bn = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(in_features=1024,out_features=512)  # linear 2
        self.fc2_bn = nn.BatchNorm1d(num_features=512)
        self.fc3 = nn.Linear(in_features=512,out_features=256)  # linear 3
        self.fc3_bn = nn.BatchNorm1d(num_features=256)
        self.fc4 = nn.Linear(in_features=256,out_features=64)  # linear 4
        self.fc4_bn = nn.BatchNorm1d(num_features=64)

        # Final Layer
        self.out = nn.Linear(in_features=64,out_features=10)  # output

        # Dropout
        self.dropout = nn.Dropout(0.4)

    def forward(self,z):
        # Apply Relu then Max Pool function between each convolution layer
        z = F.relu(self.conv1_bn(self.conv1(z)))  # (1, 28, 28)
        z = F.max_pool2d(z,kernel_size=2,stride=2)  # (1, 14, 14)

        z = F.relu(self.conv2_bn(self.conv2(z)))  # (1, 14, 14)
        z = F.max_pool2d(z,kernel_size=2,stride=2)  # (1, 7, 7)

        z = F.relu(self.conv3_bn(self.conv3(z)))  # (1, 7, 7)
        z = F.max_pool2d(z,kernel_size=2,stride=1)  # (1, 6, 6)

        z = F.relu(self.conv4_bn(self.conv4(z)))  # (1, 7, 7)
        z = F.max_pool2d(z,kernel_size=4,stride=1)  # (1, 4, 4)

        # Apply Relu function between each fully connected layer
        #print(z.size())
        z = F.relu(self.fc1_bn(self.fc1(z.reshape(-1,256 * 4 * 4))))  # 256 4 4
        z = self.dropout(z)

        z = F.relu(self.fc2_bn(self.fc2(z)))
        z = self.dropout(z)

        z = F.relu(self.fc3_bn(self.fc3(z)))
        z = self.dropout(z)

        z = F.relu(self.fc4_bn(self.fc4(z)))
        z = self.out(z)

        return z

def count_nn_params(model):
    """
    counts the number of trainable weights+biases in the nn
    :param model:
    :return:
    """
    num_params = 0
    for param in list(model.parameters()):
        n = 1
        for s in list(param.size()):
            n = n*s
        num_params += n
    return num_params


def _test_model():
    try:
        from kannadamnistpackage.DataManipulation import MnistDataset
    except ModuleNotFoundError:
        from DataManipulation import MnistDataset

    file_path = "Kannada-MNIST/train.csv"
    #arr      =  load_csv(file_path)

    batch_size = 100
    data_set = MnistDataset(file_path,im_size=(28,28))
    train_loader = DataLoader(data_set,batch_size=100,shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    my_cnn = simple_model(init_width=4,dropout_rate=0.4).to(device)
    for i,data in enumerate(train_loader,0):
        if i == 0:
            inputs,labels = data[0].to(device,dtype=torch.float),data[1].to(device)

            out = my_cnn(inputs)
            print(out.shape)
            assert out.shape == (batch_size,10)


        else:
            break
    my_cnn = mnist_model(init_width=4,dropout_rate=0.4).to(device)
    for i,data in enumerate(train_loader,0):
        if i == 0:
            inputs,labels = data[0].to(device,dtype=torch.float),data[1].to(device)

            out = my_cnn(inputs)
            print(out.shape)
            assert out.shape == (batch_size,10)


        else:
            break

    my_cnn = mnist_model_pool_single_conv(init_width=8,dropout_rate=0.4).to(device)
    for i,data in enumerate(train_loader,0):
        if i == 0:
            inputs,labels = data[0].to(device,dtype=torch.float),data[1].to(device)

            out = my_cnn(inputs)
            print(out.shape)
            assert out.shape == (batch_size,10)


        else:
            break


if __name__=='__main__':
    _test_model()





    

