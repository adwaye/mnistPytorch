from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

import pkg_resources
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms as trns
# from pkgutil import get_data
# import os, sys
#from Augmentor.Operations import Distort

label_dict = {'0' : 'shunya',
              '1' :'omdu',
              '2' :'eradu',
              '3' :'muru',
              '4' :'nalku',
              '5' :'aidu',
              '6' :'aru',
              '7' :'elu',
              '8' :'emtu',
              '9' :'ombattu',
              }




class MnistDataset(torch.utils.data.Dataset):
    """returns a dataset object that returns a tuple im, lab from the mnist format csv

    :param csvfile: pandas dataframe or location pointing to csv file, csv must be in the mnist
                    format where col[0] contains the labels and col[1:] contain flattened pixel intensities
    :type csvfile: str or pandas.dataframe
    :param im_size: tuple (width, height) of the image contained in col[1:]
    :type im_size: tuple of ints
    :param transforms: transformations to be applied to the image before being fed into the nn,
                       last transformation needs to be torchvision.transforms.ToTensor() to work with architectures
    :type transforms: torchvision.transforms


    :caption: Using the function



    .. code-block::

       from mypackage.DataManipulation import *
       import torchvision.transforms as trns
       trans_train = trns.Compose([
                                trns.ToPILImage(),
                                trns.RandomAffine(translate=(0.1,0.1),scale=(0.9,1.1),degrees=(9,9)),
                                trans.ToTensor()
                                ])
       data_set = MnistDataset(file_path,im_size=(28,28),transforms=trans_train)
       train_loader = DataLoader(data_set,batch_size=100,shuffle=True)
    """


    def __init__(self,csvfile,im_size,transforms=None):
        super(MnistDataset,self).__init__()
        if type(csvfile) is str:
            self.dataframe = pd.read_csv(csvfile)
        elif csvfile.__module__ == 'pandas.core.frame':
            # self.csvfile    = csvfile
            self.dataframe = csvfile

        self.im_size    = im_size
        if transforms is None:
            self.transforms = trns.Compose([trns.ToPILImage(),
                                    trns.ToTensor()
                                    ])
        else:
            self.transforms = transforms



    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self,i):

        row = self.dataframe.iloc[i,:]
        im   = np.array(row[1:]).astype(np.uint8).reshape(self.im_size[0],self.im_size[1],1)
        #im   = im.astype(np.float32)
        #image = (im - np.min(im)) / (np.max(im) - np.min(im))
        if self.transforms:
            im = self.transforms(im)
        lab   = row[0]
        return im, lab



def load_train_csv():
    """loads the sample submission file to be filled in at inference time

    :return: dataframe containing the training data see https://www.kaggle.com/competitions/Kannada-MNIST/data
    :rtype: pd.dataframe
    """

    # data = get_data('<kannadamnistpackage','Data/train.csv')
    # d = os.path.dirname(sys.modules['kannadamnistpackage'].__file__)
    #d = os.path.dirname(sys.modules[__name__].__file__)
    # d = os.listdir('./')
    stream = pkg_resources.resource_filename(__name__,'Data/train.csv')
    print(stream)

    # print(d)

    return pd.read_csv(stream,encoding='latin-1')

def load_test_csv():
    """loads the sample submission file to be filled in at inference time

    :return: dataframe containing the test data see https://www.kaggle.com/competitions/Kannada-MNIST/data
    :rtype: pd.dataframe
    """
    stream = pkg_resources.resource_filename(__name__,'Data/test.csv')
    return pd.read_csv(stream,encoding='latin-1')

def load_dig_csv():
    """loads the sample submission file to be filled in at inference time

    :return: dataframe containgin the Dig-MNIST data see https://www.kaggle.com/competitions/Kannada-MNIST/data
    :rtype: pd.dataframe
    """
    stream = pkg_resources.resource_filename(__name__,'Data/Dig-MNIST.csv')
    return pd.read_csv(stream,encoding='latin-1')

def load_sample_submission():
    """loads the sample submission file to be filled in at inference time

    :return: empty pandas dataframe to store predicted values
    :rtype: pd.dataframe
    """

    stream = pkg_resources.resource_filename(__name__,'Data/sample_submission.csv')
    return pd.read_csv(stream,encoding='latin-1')




def _plot_digits():
    """plots samples from the MNISTDataset class

    :return: plot of a sample from mnist generated from the mnistdataset class
    :rtype: matplotlib.figure
    """

    #arr      =  load_csv(file_path)

    df = load_train_csv()

    # p.rotate(probability=0.7,max_left_rotation=10,max_right_rotation=10)
    # p.zoom(probability=0.5,min_factor=1.1,max_factor=1.3)
    # p.random_distortion(grid_width=5,grid_height=5,magnitude=2,probability=0.5)
    # trans = tr.Compose([
    #     transforms.ToPILImage(),
    #     transforms.ToTensor()
    # ])

    data_gen = MnistDataset(df,im_size=(28,28))


    fix,ax = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            sampled_index = np.random.randint(0,df.shape[0])
            im,lab = data_gen.__getitem__(i=sampled_index)

            ax[i,j].imshow(im[0],cmap='Greys_r')
            ax[i,j].set_title(f'label = {lab} {label_dict[str(lab)]}')
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
    plt.tight_layout()
    plt.show()
    return fig





def _test_data():
    """Tests the shape of the output data from MnistDataset

    :return: None
    :rtype:  None
    """
    csvfile   = load_test_csv()
    out_shape = (1,28,28)
    data_gen  = MnistDataset(csvfile,im_size=out_shape[1:])
    im,lab    = data_gen.__getitem__(i=0)
    assert im.shape == out_shape
    assert type(lab)==np.int64




if __name__=='__main__':
    _plot_digits()
    _test_data()
    # import torch
    # csvfile   = "Kannada-MNIST/train.csv"
    # out_shape = (1,28,28)
    # data_gen  = MnistDataset(csvfile,im_size=out_shape[1:])
    # im,lab    = data_gen.__getitem__(i=0)
    # assert im.shape == out_shape
    # #assert type(im[0,0]) == np.float
    # assert type(lab)==np.int64
    #
    # p = Augmentor.Pipeline()
    # p.rotate(probability=0.3,max_left_rotation=10,max_right_rotation=10)
    # p.zoom(probability=0.3,min_factor=1.05,max_factor=1.1)
    # p.random_distortion(grid_width=5,grid_height=5,magnitude=2,probability=0.3)
    # trans = transforms.Compose([
    #     p.torch_transform(),
    #     transforms.ToTensor()
    # ])
    # data_gen = MnistDataset(csvfile,im_size=out_shape[1:],transforms=trans)
    # data_loader = DataLoader(data_gen)
    # for i,data in enumerate(data_loader):
    #     if i==0:
    #         inputs,labels = data[0].to('cuda',dtype=torch.float),data[1].to('cuda')
    #         print("max_va={:},minval={:}".format(np.max(data[0].to('cpu').numpy()),np.min(data[0].to('cpu').numpy())))
    # import Augmentor
    # p = Augmentor.Pipeline()
    # p.rotate(probability=0.7,max_left_rotation=10,max_right_rotation=10)
    # p.zoom(probability=0.5,min_factor=1.1,max_factor=1.3)
    # p.random_distortion(grid_width=5,grid_height=5,magnitude=2,probability=0.5)
    # trans = transforms.Compose([
    #         p.torch_transform(),
    #         transforms.ToTensor()
    # ])
    #
    # file_path = "Kannada-MNIST/train.csv"
    # #arr      =  load_csv(file_path)
    #
    # df = pd.read_csv(file_path)
    #
    # data_gen = MnistDataset(file_path,im_size=(28,28),transforms=trans)
    #
    #
    # fix,ax = plt.subplots(3,3)
    # for i in range(3):
    #     for j in range(3):
    #         sampled_index = np.random.randint(0,df.shape[0])
    #         im,lab = data_gen.__getitem__(i=sampled_index)
    #
    #         ax[i,j].imshow(im[0],cmap='Greys_r')
    #         ax[i,j].set_title('label = {:} {:}'.format(lab,label_dict[str(lab)]))
    #         ax[i,j].set_xticks([])
    #         ax[i,j].set_yticks([])
    # plt.tight_layout()


