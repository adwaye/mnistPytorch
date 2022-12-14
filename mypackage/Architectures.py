import torch.cuda
from torch.utils.data import DataLoader, Dataset
import numpy as np
import csv
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

class simple_model(Module):
    """
    model with the following architecture
    conv2d(kern=3,stride=1,out=init_width)->relu->conv2d(kern=3,stride=2,out=init_width)->relu->
    conv2d(kern=3,stride=1,out=init_width*2)->relu->conv2d(kern=3,stride=2,out=init_width*2)->relu->
    conv2d(kern=3,stride=1,out=init_width*4)->relu->conv2d(kern=3,stride=2,out=init_width*4)->relu->
    fc(out=init_with*4*4)->relu->fc(out=init_with*4)->relu->fc(out=10)
    """


    def __init__(self,init_width=4,dropout_rate=0.4):
        """
        :param init_width: initial width of filters for first conv layer
        :type init_width: int
        :param dropout_rate: dropout rate to be used in fc layers between 0 and 1
        :type dropout_rate: float
        """

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
    """
    model with the following architecture
    conv2d(kern=3,stride=1,out=init_width*2)->relu->conv2d(kern=3,stride=2,out=init_width)->batch_norm->relu->
    conv2d(kern=3,stride=1,out=init_width*4)->relu->conv2d(kern=3,stride=2,out=init_width*2))->batch_norm->relu->
    conv2d(kern=3,stride=1,out=init_width*4)->relu->conv2d(kern=3,stride=2,out=init_width*4))->batch_norm->relu->
    conv2d(kern=3,stride=1,out=init_width*4)->relu->conv2d(kern=3,stride=2,out=init_width*8))->batch_norm->relu->
    fc(out=init_with*4*4)->bnorm->relu->fc(out=init_with*4*2)->bnorm->relu->fc(out=init_with*4)->bnorm->relu->fc(out=10)
    """

    def __init__(self,init_width=16,dropout_rate=0.4):
        """
        :param init_width: initial width of filters for first conv layer
        :type init_width: int
        :param dropout_rate: dropout rate to be used in fc layers between 0 and 1
        :type dropout_rate: float
        """

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
        #size (14,14,4)



        self.conv2_1 = nn.Conv2d(in_channels=self.init_width,out_channels=self.init_width*2,kernel_size=3,stride=1,
                                 padding=1)

        self.conv2_2 = nn.Conv2d(in_channels=self.init_width*2,out_channels=self.init_width*2,kernel_size=3,stride=2,
                                 padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_features=self.init_width*2)
        #size (7,7,8)

        self.conv3_1 = nn.Conv2d(in_channels=self.init_width*2,out_channels=self.init_width*4,kernel_size=3,stride=1,
                                 padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=self.init_width * 4,out_channels=self.init_width * 4,kernel_size=3,
                                 stride=2,
                                 padding=1)
        self.conv3_bn = nn.BatchNorm2d(num_features=self.init_width*4)
        #size (4,4,16)


        self.conv4_1 = nn.Conv2d(in_channels=self.init_width*4,out_channels=self.init_width*8,kernel_size=3,stride=1,
                                 padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=self.init_width * 8,out_channels=self.init_width * 8,kernel_size=3,
                                 stride=2,
                                 padding=1)
        self.conv4_bn = nn.BatchNorm2d(num_features=self.init_width*8)
        #size (2,2,16)


        self.fc1  = nn.Linear(in_features=self.init_width * 2 * 2*8,out_features=self.init_width*4*4)
        self.fc1_bn = nn.BatchNorm1d(num_features=self.init_width*4*4)
        self.fc2  = nn.Linear(in_features=self.init_width * 4 * 4 ,out_features=self.init_width * 4*2 )
        self.fc2_bn = nn.BatchNorm1d(num_features=self.init_width * 4 *2)

        self.fc3 = nn.Linear(in_features=self.init_width * 4*2 ,out_features=self.init_width * 4)
        self.fc3_bn = nn.BatchNorm1d(num_features=self.init_width * 4)

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


        z = F.relu(self.conv4_1(z))
        z = F.relu(self.conv4_bn(self.conv4_2(z)))


        z = torch.flatten(z,1)
        z = F.relu(self.fc1_bn(self.fc1(z)))
        z = self.dropout(z)
        z = F.relu(self.fc2_bn(self.fc2(z)))
        z = self.dropout(z)
        z = F.relu(self.fc3_bn(self.fc3(z)))
        z = self.dropout(z)
        z = self.out(z)
        return z


class mnist_model_pool(nn.Module):
    """
    model with the following architecture
    conv2d(kern=3,stride=1,out=init_width*2)->relu->conv2d(kern=3,stride=1,out=init_width)->batch_norm->relu->max_pool2d
    conv2d(kern=3,stride=1,out=init_width*4)->relu->conv2d(kern=3,stride=1,out=init_width*2))->batch_norm->relu->max_pool2d
    conv2d(kern=3,stride=1,out=init_width*4)->relu->conv2d(kern=3,stride=1,out=init_width*4))->batch_norm->relu->max_pool2d
    conv2d(kern=3,stride=1,out=init_width*4)->relu->conv2d(kern=3,stride=1,out=init_width*8))->batch_norm->relu->
    fc(out=init_with*4*4)->bnorm->relu->fc(out=init_with*4)->bnorm->relu->fc(out=10)
    """
    def __init__(self,init_width=4,dropout_rate=0.4):
        """
        :param init_width: initial width of filters for first conv layer
        :type init_width: int
        :param dropout_rate: dropout rate to be used in fc layers between 0 and 1
        :type dropout_rate: float
        """
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
        # self.conv4_2 = nn.Conv2d(in_channels=self.init_width * 8,out_cha=self.init_width * 4,kernel_size=3,
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



class mnist_model_pool3fc(nn.Module):
    """
    model with the following architecture
    conv2d(kern=3,stride=1,out=init_width*2)->relu->conv2d(kern=3,stride=1,out=init_width)->batch_norm->relu->max_pool2d
    conv2d(kern=3,stride=1,out=init_width*4)->relu->conv2d(kern=3,stride=1,out=init_width*2))->batch_norm->relu->max_pool2d
    conv2d(kern=3,stride=1,out=init_width*4)->relu->conv2d(kern=3,stride=1,out=init_width*4))->batch_norm->relu->max_pool2d
    conv2d(kern=3,stride=1,out=init_width*4)->relu->conv2d(kern=3,stride=1,out=init_width*8))->batch_norm->relu->
    fc(out=init_with*4*4)->bnorm->relu->fc(out=init_with*4*2)->bnorm->relu->fc(out=init_with*4)->bnorm->relu->fc(out=10)
    """

    def __init__(self,init_width=4,dropout_rate=0.4):
        """
        :param init_width: initial width of filters for first conv layer
        :type init_width: int
        :param dropout_rate: dropout rate to be used in fc layers between 0 and 1
        :type dropout_rate: float
        """
        super(mnist_model_pool3fc,self).__init__()

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
        # self.conv4_2 = nn.Conv2d(in_channels=self.init_width * 8,out_cha=self.init_width * 4,kernel_size=3,
        #                          stride=1,
        #                          padding=1)
        #size (4,4,16)


        self.fc1  = nn.Linear(in_features=self.init_width * 4 * 4*8,out_features=self.init_width*4*4)
        self.fc1_bn = nn.BatchNorm1d(num_features=self.init_width*4*4)
        self.fc2  = nn.Linear(in_features=self.init_width * 4 * 4 ,out_features=self.init_width * 4*2 )
        self.fc2_bn = nn.BatchNorm1d(num_features=self.init_width * 4 *2)

        self.fc3 = nn.Linear(in_features=self.init_width * 4*2 ,out_features=self.init_width * 4)
        self.fc3_bn = nn.BatchNorm1d(num_features=self.init_width * 4)


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
        z = F.relu(self.fc3_bn(self.fc3(z)))
        #z = self.dropout(z)
        z = self.out(z)
        return z

class mnist_model_pool_leaky(nn.Module):
    """
    model with the following architecture
    conv2d(kern=3,stride=1,out=init_width*2)->leaky_relu->conv2d(kern=3,stride=1,out=init_width)->batch_norm->leaky_relu->max_pool2d
    conv2d(kern=3,stride=1,out=init_width*4)->leaky_relu->conv2d(kern=3,stride=1,out=init_width*2))->batch_norm->leaky_relu->max_pool2d
    conv2d(kern=3,stride=1,out=init_width*4)->leaky_relu->conv2d(kern=3,stride=1,out=init_width*4))->batch_norm->leaky_relu->max_pool2d
    conv2d(kern=3,stride=1,out=init_width*4)->leaky_relu->conv2d(kern=3,stride=1,out=init_width*8))->batch_norm->leaky_relu->
    fc(out=init_with*4*4)->bnorm->leaky_relu->fc(out=init_with*4)->bnorm->leaky_relu->fc(out=10)
    """

    def __init__(self,init_width=4,dropout_rate=0.4,negative_slope=0.1):
        """
        :param init_width: initial width of filters for first conv layer
        :type init_width: int
        :param dropout_rate: dropout rate to be used in fc layers between 0 and 1
        :type dropout_rate: float
        """
        super(mnist_model_pool_leaky,self).__init__()

        self.init_width   = init_width
        self.dropout_rate = dropout_rate
        self.negative_slope = negative_slope

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
        # self.conv4_2 = nn.Conv2d(in_channels=self.init_width * 8,out_cha=self.init_width * 4,kernel_size=3,
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
        z = F.leaky_relu(self.conv1_1(z),negative_slope=self.negative_slope)
        z = F.leaky_relu(self.conv1_bn(self.conv1_2(z)),negative_slope=self.negative_slope)

        z = F.max_pool2d(z,kernel_size=2)
        z = F.leaky_relu(self.conv2_1(z),negative_slope=self.negative_slope)
        z = F.leaky_relu(self.conv2_bn(self.conv2_2(z)),negative_slope=self.negative_slope)

        z = F.max_pool2d(z,kernel_size=2)
        z = F.leaky_relu(self.conv3_1(z),negative_slope=self.negative_slope)
        z = F.leaky_relu(self.conv3_bn(self.conv3_2(z)),negative_slope=self.negative_slope)

        z = F.max_pool2d(z,kernel_size=2,padding=1)
        z = F.leaky_relu(self.conv4_1(z),negative_slope=self.negative_slope)
        z = F.leaky_relu(self.conv4_bn(self.conv4_2(z)),negative_slope=self.negative_slope)
        #

        # z = F.relu(self.conv4_1(z))
        # z = F.relu(self.conv4_2(z))
        #
        z = torch.flatten(z,1)
        z = F.leaky_relu(self.fc1_bn(self.fc1(z)),negative_slope=self.negative_slope)
        z = self.dropout(z)
        z = F.leaky_relu(self.fc2_bn(self.fc2(z)),negative_slope=self.negative_slope)
        #z = self.dropout(z)
        z = self.out(z)
        return z





class mnist_model_pool_kern5(nn.Module):
    """
    model with the following architecture
    conv2d(kern=5,stride=1,out=init_width*2)->relu->conv2d(kern=5,stride=1,out=init_width)->batch_norm->relu->max_pool2d
    conv2d(kern=5,stride=1,out=init_width*4)->relu->conv2d(kern=5,stride=1,out=init_width*2))->batch_norm->relu->max_pool2d
    conv2d(kern=5,stride=1,out=init_width*4)->relu->conv2d(kern=5,stride=1,out=init_width*4))->batch_norm->relu->max_pool2d
    conv2d(kern=5,stride=1,out=init_width*4)->relu->conv2d(kern=5,stride=1,out=init_width*8))->batch_norm->relu->
    fc(out=init_with*4*4)->bnorm->relu->fc(out=init_with*4)->bnorm->relu->fc(out=10)
    """
    def __init__(self,init_width=4,dropout_rate=0.4):
        """

        :param init_width: initial width of filters for first conv layer
        :type init_width: int
        :param dropout_rate: dropout rate to be used in fc layers between 0 and 1
        :type dropout_rate: float
        """
        super(mnist_model_pool_kern5,self).__init__()

        self.init_width   = init_width
        self.dropout_rate = dropout_rate

        self.init_layers()


    def init_layers(self):
        #input size (28,28)
        self.conv1_1 = nn.Conv2d(in_channels=1,out_channels=self.init_width,kernel_size=5,stride=1,padding=2)
        #num weights= 3*4*1=12
        self.conv1_2 = nn.Conv2d(in_channels=self.init_width,out_channels=self.init_width,kernel_size=5,stride=1,
                                 padding=2)
        self.conv1_bn = nn.BatchNorm2d(num_features=self.init_width)
        #num weights= 3*4*4=48, num biases = 4
        #size (14,14,4)

        #self.batch1 = nn.BatchNorm2d(num_features=self.init_width)
        #input size is halved (14,14)
        self.conv2_1 = nn.Conv2d(in_channels=self.init_width,out_channels=self.init_width*2,kernel_size=5,stride=1,
                                 padding=2)
        #num weights= 3*4*8=96 num biases = 8
        self.conv2_2 = nn.Conv2d(in_channels=self.init_width*2,out_channels=self.init_width*2,kernel_size=5,stride=1,
                                 padding=2)
        self.conv2_bn = nn.BatchNorm2d(num_features=self.init_width*2)
        #num weights= 3*8*8=192 num biases = 8
        #size (14,14,8)
        #input size is halved (7,7)
        self.conv3_1 = nn.Conv2d(in_channels=self.init_width*2,out_channels=self.init_width*4,kernel_size=5,stride=1,
                                 padding=2)
        #num weights= 3*8*16=384 num biases = 16
        self.conv3_2 = nn.Conv2d(in_channels=self.init_width * 4,out_channels=self.init_width * 4,kernel_size=5,
                                 stride=1,
                                 padding=2)
        self.conv3_bn = nn.BatchNorm2d(num_features=self.init_width*4)


        self.conv4_1 = nn.Conv2d(in_channels=self.init_width*4,out_channels=self.init_width*8,kernel_size=5,stride=1,
                                 padding=2)
        #num weights= 3*8*16=384 num biases = 16
        self.conv4_2 = nn.Conv2d(in_channels=self.init_width * 8,out_channels=self.init_width * 8,kernel_size=5,
                                 stride=1,
                                 padding=2)
        self.conv4_bn = nn.BatchNorm2d(num_features=self.init_width*8)
        #size (7,7,16)
        #input size is halved (4,4)
        # self.conv4_1 = nn.Conv2d(in_channels=self.init_width * 4,out_channels=self.init_width*8 ,kernel_size=3,
        #                          stride=2,
        #                          padding=1)
        # self.conv4_2 = nn.Conv2d(in_channels=self.init_width * 8,out_cha=self.init_width * 4,kernel_size=3,
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

class mnist_model_pool_inddim3(nn.Module):
    """
    model with the following architecture
    conv2d(kern=(1,3),stride=1,out=init_width*2)->relu->conv2d(kern=(3,1),stride=1,out=init_width)->batch_norm->relu->max_pool2d
    conv2d(kern=(1,3),stride=1,out=init_width*4)->relu->conv2d(kern=(3,1),stride=1,out=init_width*2))->batch_norm->relu->max_pool2d
    conv2d(kern=(1,3),stride=1,out=init_width*4)->relu->conv2d(kern=(3,1),stride=1,out=init_width*4))->batch_norm->relu->max_pool2d
    conv2d(kern=(1,3),stride=1,out=init_width*4)->relu->conv2d(kern=(3,1),stride=1,out=init_width*8))->batch_norm->relu->
    fc(out=init_with*4*4)->bnorm->relu->fc(out=init_with*4)->bnorm->relu->fc(out=10)
    """
    def __init__(self,init_width=4,dropout_rate=0.4):
        """

        :param init_width: initial width of filters for first conv layer
        :type init_width: int
        :param dropout_rate: dropout rate to be used in fc layers between 0 and 1
        :type dropout_rate: float
        """
        super(mnist_model_pool_inddim3,self).__init__()

        self.init_width   = init_width
        self.dropout_rate = dropout_rate

        self.init_layers()


    def init_layers(self):
        #input size (28,28)
        self.conv1_1 = nn.Conv2d(in_channels=1,out_channels=self.init_width,kernel_size=(3,1),stride=(1,1),
                                 padding=(1,0))
        #num weights= 3*4*1=12
        self.conv1_2 = nn.Conv2d(in_channels=self.init_width,out_channels=self.init_width,kernel_size=(1,3),
                                 stride=(1,1),
                                 padding=(0,1))
        self.conv1_bn = nn.BatchNorm2d(num_features=self.init_width)
        #num weights= 3*4*4=48, num biases = 4
        #size (14,14,4)

        #self.batch1 = nn.BatchNorm2d(num_features=self.init_width)
        #input size is halved (14,14)
        self.conv2_1 = nn.Conv2d(in_channels=self.init_width,out_channels=self.init_width*2,kernel_size=(3,1),
                                 stride=(1,1),
                                 padding=(1,0))
        #num weights= 3*4*8=96 num biases = 8
        self.conv2_2 = nn.Conv2d(in_channels=self.init_width*2,out_channels=self.init_width*2,kernel_size=(1,3),
                                 stride=(1,1),
                                 padding=(0,1))
        self.conv2_bn = nn.BatchNorm2d(num_features=self.init_width*2)
        #num weights= 3*8*8=192 num biases = 8
        #size (14,14,8)
        #input size is halved (7,7)
        self.conv3_1 = nn.Conv2d(in_channels=self.init_width*2,out_channels=self.init_width*4,kernel_size=(3,1),
                                 stride=(1,1),
                                 padding=(1,0))
        #num weights= 3*8*16=384 num biases = 16
        self.conv3_2 = nn.Conv2d(in_channels=self.init_width * 4,out_channels=self.init_width * 4,kernel_size=(1,3),
                                 stride=(1,1),
                                 padding=(0,1))
        self.conv3_bn = nn.BatchNorm2d(num_features=self.init_width*4)


        self.conv4_1 = nn.Conv2d(in_channels=self.init_width*4,out_channels=self.init_width*8,kernel_size=(3,1),
                                 stride=(1,1),
                                 padding=(1,0))
        #num weights= 3*8*16=384 num biases = 16
        self.conv4_2 = nn.Conv2d(in_channels=self.init_width * 8,out_channels=self.init_width * 8,kernel_size=(3,1),
                                 stride=(1,1),
                                 padding=(1,0))
        self.conv4_bn = nn.BatchNorm2d(num_features=self.init_width*8)
        #size (7,7,16)

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



class mnist_model_pool_inddim5(nn.Module):
    """
    model with the following architecture
    conv2d(kern=(1,5),stride=1,out=init_width*2)->relu->conv2d(kern=(5,1),stride=1,out=init_width)->batch_norm->relu->max_pool2d
    conv2d(kern=(1,5),stride=1,out=init_width*4)->relu->conv2d(kern=(5,1),stride=1,out=init_width*2))->batch_norm->relu->max_pool2d
    conv2d(kern=(1,5),stride=1,out=init_width*4)->relu->conv2d(kern=(5,1),stride=1,out=init_width*4))->batch_norm->relu->max_pool2d
    conv2d(kern=(1,5),stride=1,out=init_width*4)->relu->conv2d(kern=(5,1),stride=1,out=init_width*8))->batch_norm->relu->
    fc(out=init_with*4*4)->bnorm->relu->fc(out=init_with*4)->bnorm->relu->fc(out=10)
    """
    def __init__(self,init_width=4,dropout_rate=0.4):
        """

        :param init_width: initial width of filters for first conv layer
        :type init_width: int
        :param dropout_rate: dropout rate to be used in fc layers between 0 and 1
        :type dropout_rate: float
        """
        super(mnist_model_pool_inddim5,self).__init__()

        self.init_width   = init_width
        self.dropout_rate = dropout_rate

        self.init_layers()


    def init_layers(self):
        #input size (28,28)
        self.conv1_1 = nn.Conv2d(in_channels=1,out_channels=self.init_width,kernel_size=(5,1),stride=(1,1),
                                 padding=(2,0))
        #num weights= 3*4*1=12
        self.conv1_2 = nn.Conv2d(in_channels=self.init_width,out_channels=self.init_width,kernel_size=(1,5),
                                 stride=(1,1),
                                 padding=(0,2))
        self.conv1_bn = nn.BatchNorm2d(num_features=self.init_width)
        #num weights= 3*4*4=48, num biases = 4
        #size (14,14,4)

        #self.batch1 = nn.BatchNorm2d(num_features=self.init_width)
        #input size is halved (14,14)
        self.conv2_1 = nn.Conv2d(in_channels=self.init_width,out_channels=self.init_width*2,kernel_size=(5,1),stride=(1,1),
                                 padding=(2,0))
        #num weights= 3*4*8=96 num biases = 8
        self.conv2_2 = nn.Conv2d(in_channels=self.init_width*2,out_channels=self.init_width*2,kernel_size=(1,5),
                                 stride=(1,1),
                                 padding=(0,2))
        self.conv2_bn = nn.BatchNorm2d(num_features=self.init_width*2)
        #num weights= 3*8*8=192 num biases = 8
        #size (14,14,8)
        #input size is halved (7,7)
        self.conv3_1 = nn.Conv2d(in_channels=self.init_width*2,out_channels=self.init_width*4,kernel_size=(5,1),stride=(1,1),
                                 padding=(2,0))
        #num weights= 3*8*16=384 num biases = 16
        self.conv3_2 = nn.Conv2d(in_channels=self.init_width * 4,out_channels=self.init_width * 4,kernel_size=(1,5),
                                 stride=(1,1),
                                 padding=(0,2))
        self.conv3_bn = nn.BatchNorm2d(num_features=self.init_width*4)


        self.conv4_1 = nn.Conv2d(in_channels=self.init_width*4,out_channels=self.init_width*8,kernel_size=(5,1),stride=(1,1),
                                 padding=(2,0))
        #num weights= 3*8*16=384 num biases = 16
        self.conv4_2 = nn.Conv2d(in_channels=self.init_width * 8,out_channels=self.init_width * 8,kernel_size=(1,5),
                                 stride=(1,1),
                                 padding=(0,2))
        self.conv4_bn = nn.BatchNorm2d(num_features=self.init_width*8)
        #size (7,7,16)
        #input size is halved (4,4)
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
    """
    model with the following architecture
    conv2d(kern=3,stride=1,out=init_width*2)->relu->max_pool2d
    conv2d(kern=3,stride=1,out=init_width*4)->relu->max_pool2d
    conv2d(kern=3,stride=1,out=init_width*4)->relu->max_pool2d
    conv2d(kern=3,stride=1,out=init_width*4)->relu->
    fc(out=init_with*4*4)->bnorm->relu->fc(out=init_with*4)->bnorm->relu->fc(out=10)
    """

    def __init__(self,init_width=4,dropout_rate=0.4):
        """

        :param init_width: initial width of filters for first conv layer
        :type init_width: int
        :param dropout_rate: dropout rate to be used in fc layers between 0 and 1
        :type dropout_rate: float
        """
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


class mnist_model_pool_bn_leaky(nn.Module):
    """
    model with the following architecture
    conv2d(kern=3,stride=1,out=init_width*2)->batch_norm->leaky_relu->conv2d(kern=3,stride=1,out=init_width)->batch_norm->leaky_relu->max_pool2d
    conv2d(kern=3,stride=1,out=init_width*4)->batch_norm->leaky_relu->conv2d(kern=3,stride=1,out=init_width*2))->batch_norm->leaky_relu->max_pool2d
    conv2d(kern=3,stride=1,out=init_width*4)->batch_norm->leaky_relu->conv2d(kern=3,stride=1,out=init_width*4))->batch_norm->leaky_relu->max_pool2d
    conv2d(kern=3,stride=1,out=init_width*4)->batch_norm->leaky_relu->conv2d(kern=3,stride=1,out=init_width*8))->batch_norm->leaky_relu->
    fc(out=init_with*4*4)->bnorm->relu->fc(out=init_with*4)->bnorm->relu->fc(out=10)
    """
    def __init__(self,init_width=4,dropout_rate=0.4,negative_slope=0.1):
        """

        :param init_width: initial width of filters for first conv layer
        :type init_width: int
        :param dropout_rate: dropout rate to be used in fc layers between 0 and 1
        :type dropout_rate: float
        """

        super(mnist_model_pool_bn_leaky,self).__init__()

        self.init_width     = init_width
        self.dropout_rate   = dropout_rate
        self.negative_slope = negative_slope

        self.init_layers()

    def init_layers(self):
        #input size (28,28)
        self.conv1_1 = nn.Conv2d(in_channels=1,out_channels=self.init_width,kernel_size=3,stride=1,padding=1)
        #num weights= 3*4*1=12
        self.conv1_2 = nn.Conv2d(in_channels=self.init_width,out_channels=self.init_width,kernel_size=3,stride=1,
                                 padding=1)

        self.conv11_bn = nn.BatchNorm2d(num_features=self.init_width)
        self.conv12_bn = nn.BatchNorm2d(num_features=self.init_width)

        #num weights= 3*4*4=48, num biases = 4
        #size (14,14,4)

        #self.batch1 = nn.BatchNorm2d(num_features=self.init_width)
        #input size is halved (14,14)
        self.conv2_1 = nn.Conv2d(in_channels=self.init_width,out_channels=self.init_width * 2,kernel_size=3,stride=1,
                                 padding=1)
        #num weights= 3*4*8=96 num biases = 8
        self.conv2_2 = nn.Conv2d(in_channels=self.init_width * 2,out_channels=self.init_width * 2,kernel_size=3,
                                 stride=1,
                                 padding=1)

        self.conv21_bn = nn.BatchNorm2d(num_features=self.init_width * 2)
        self.conv22_bn = nn.BatchNorm2d(num_features=self.init_width * 2)
        #num weights= 3*8*8=192 num biases = 8
        #size (14,14,8)
        #input size is halved (7,7)
        self.conv3_1 = nn.Conv2d(in_channels=self.init_width * 2,out_channels=self.init_width * 4,kernel_size=3,
                                 stride=1,
                                 padding=1)
        #num weights= 3*8*16=384 num biases = 16
        self.conv3_2 = nn.Conv2d(in_channels=self.init_width * 4,out_channels=self.init_width * 4,kernel_size=3,
                                 stride=1,
                                 padding=1)

        self.conv31_bn = nn.BatchNorm2d(num_features=self.init_width * 4)
        self.conv32_bn = nn.BatchNorm2d(num_features=self.init_width * 4)

        self.conv4_1 = nn.Conv2d(in_channels=self.init_width * 4,out_channels=self.init_width * 8,kernel_size=3,
                                 stride=1,
                                 padding=1)
        #num weights= 3*8*16=384 num biases = 16
        self.conv4_2 = nn.Conv2d(in_channels=self.init_width * 8,out_channels=self.init_width * 8,kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.conv41_bn = nn.BatchNorm2d(num_features=self.init_width * 8)
        self.conv42_bn = nn.BatchNorm2d(num_features=self.init_width * 8)
        #size (7,7,16)
        #input size is halved (4,4)
        # self.conv4_1 = nn.Conv2d(in_channels=self.init_width * 4,out_channels=self.init_width*8 ,kernel_size=3,
        #                          stride=2,
        #                          padding=1)
        # self.conv4_2 = nn.Conv2d(in_channels=self.init_width * 8,out_channels=self.init_width * 4,kernel_size=3,
        #                          stride=1,
        #                          padding=1)
        #size (4,4,16)
        self.fc1 = nn.Linear(in_features=self.init_width * 4 * 4 * 8,out_features=self.init_width * 4 * 4)
        self.fc1_bn = nn.BatchNorm1d(num_features=self.init_width * 4 * 4)
        self.fc2 = nn.Linear(in_features=self.init_width * 4 * 4,out_features=self.init_width * 4)
        self.fc2_bn = nn.BatchNorm1d(num_features=self.init_width * 4)
        self.out = nn.Linear(in_features=self.init_width * 4,out_features=10)

        self.dropout = nn.Dropout(self.dropout_rate)
        #self.fc5  = nn.Linear(in_features=)
        #size (7,7,1)

    def forward(self,z):
        z = F.leaky_relu(self.conv11_bn(self.conv1_1(z)),negative_slope=self.negative_slope)
        z = F.leaky_relu(self.conv12_bn(self.conv1_2(z)),negative_slope=self.negative_slope)

        z = F.max_pool2d(z,kernel_size=2)
        z = F.leaky_relu(self.conv21_bn(self.conv2_1(z)),negative_slope=self.negative_slope)
        z = F.leaky_relu(self.conv22_bn(self.conv2_2(z)),negative_slope=self.negative_slope)

        z = F.max_pool2d(z,kernel_size=2)
        z = F.leaky_relu(self.conv31_bn(self.conv3_1(z)),negative_slope=self.negative_slope)
        z = F.leaky_relu(self.conv32_bn(self.conv3_2(z)),negative_slope=self.negative_slope)

        z = F.max_pool2d(z,kernel_size=2,padding=1)
        z = F.leaky_relu(self.conv41_bn(self.conv4_1(z)),negative_slope=self.negative_slope)
        z = F.leaky_relu(self.conv42_bn(self.conv4_2(z)),negative_slope=self.negative_slope)
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
    """
    model with the following architecture
    conv2d(kern=3,stride=1,out=32)->batch_norm->leaky_relu->max_pool2d
    conv2d(kern=3,stride=1,out=64)->batch_norm->leaky_relu->max_pool2d
    conv2d(kern=3,stride=1,out=128)->batch_norm->leaky_relu->max_pool2d
    conv2d(kern=3,stride=1,out=256)->batch_norm->leaky_relu->max_pool2d
    fc(out=10244)->bnorm->relu->fc(out=512)->bnorm->relu->fc(out=256)->bnorm->relu->fc(out=64)->bnorm->relu->fc(out=10)
    """
    def __init__(self):
        super(Network,self).__init__()
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
    counts the number of trainable weights and biases in the neural network given by model

    :param model: CNN whose weights and biases are to be counted
    :type model:  torch.nn.Module
    :return:      number of neural network weights and parameters
    :rtype:       int

   :caption: Using the function



    .. code-block::

       from Architectures import simple_model
       my_model = simple_model()
       n_params = count_nn_params(my_model)

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
        from mypackage.DataManipulation import MnistDataset,load_train_csv
    except ModuleNotFoundError:
        from DataManipulation import MnistDataset,load_train_csv
    # from torchvision import transforms
    file_path = load_train_csv()
    #arr      =  load_csv(file_path)

    batch_size = 100
    # transform = transforms.Compose([transforms.ToPILImage(),
    #                                 transforms.ToTensor()
    #                                 ])
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
    del my_cnn
    torch.cuda.empty_cache()

    my_cnn = mnist_model(init_width=16,dropout_rate=0.4).to(device)
    for i,data in enumerate(train_loader,0):
        if i == 0:
            inputs,labels = data[0].to(device,dtype=torch.float),data[1].to(device)

            out = my_cnn(inputs)
            print(out.shape)
            assert out.shape == (batch_size,10)


        else:
            break
    del my_cnn
    torch.cuda.empty_cache()
    my_cnn = mnist_model_pool_single_conv(init_width=8,dropout_rate=0.4).to(device)
    for i,data in enumerate(train_loader,0):
        if i == 0:
            inputs,labels = data[0].to(device,dtype=torch.float),data[1].to(device)

            out = my_cnn(inputs)
            print(out.shape)
            assert out.shape == (batch_size,10)


        else:
            break
    del my_cnn
    torch.cuda.empty_cache()
    my_cnn = mnist_model_pool_leaky(init_width=8,dropout_rate=0.4).to(device)
    for i,data in enumerate(train_loader,0):
        if i == 0:
            inputs,labels = data[0].to(device,dtype=torch.float),data[1].to(device)

            out = my_cnn(inputs)
            print(out.shape)
            assert out.shape == (batch_size,10)


        else:
            break
    del my_cnn
    torch.cuda.empty_cache()
    my_cnn = mnist_model_pool_kern5(init_width=8,dropout_rate=0.4).to(device)
    for i,data in enumerate(train_loader,0):
        if i == 0:
            inputs,labels = data[0].to(device,dtype=torch.float),data[1].to(device)

            out = my_cnn(inputs)
            print(out.shape)
            assert out.shape == (batch_size,10)


        else:
            break
    del my_cnn
    torch.cuda.empty_cache()
    my_cnn = mnist_model_pool3fc(init_width=8,dropout_rate=0.4).to(device)

    for i,data in enumerate(train_loader,0):
        if i == 0:
            inputs,labels = data[0].to(device,dtype=torch.float),data[1].to(device)

            out = my_cnn(inputs)
            print(out.shape)
            assert out.shape == (batch_size,10)


        else:
            break
    del my_cnn
    torch.cuda.empty_cache()
    my_cnn = mnist_model_pool_inddim5(init_width=8,dropout_rate=0.4).to(device)
    for i,data in enumerate(train_loader,0):
        if i == 0:
            inputs,labels = data[0].to(device,dtype=torch.float),data[1].to(device)

            out = my_cnn(inputs)
            print(out.shape)
            assert out.shape == (batch_size,10)


        else:
            break
    del my_cnn
    torch.cuda.empty_cache()
    my_cnn = mnist_model_pool_inddim3(init_width=8,dropout_rate=0.4).to(device)
    for i,data in enumerate(train_loader,0):
        if i == 0:
            inputs,labels = data[0].to(device,dtype=torch.float),data[1].to(device)

            out = my_cnn(inputs)
            print(out.shape)
            assert out.shape == (batch_size,10)


        else:
            break

    del my_cnn
    torch.cuda.empty_cache()
    my_cnn = mnist_model_pool_bn_leaky(init_width=8,dropout_rate=0.4).to(device)
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





    

