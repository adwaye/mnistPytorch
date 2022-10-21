import torch.nn
import torchvision.transforms.functional as TF
import random
import torchvision.transforms as transforms
import Augmentor,os
import numpy as np
from Augmentor.Operations import Distort


class RandomDistortion:
    def __init__(self,probability, grid_width, grid_height, magnitude):
        """
        Applies elastic distortion with prob probability see https://augmentor.readthedocs.io/en/stable/ for info on
        grdi and magnitiude params
        :param probability:
        :param grid_width:
        :param grid_height:
        :param magnitude:
        """
        self.probability = probability
        self.grid_width  = grid_width
        self.grid_height = grid_height
        self.magnitude   = magnitude
        self.op          = Distort(probability, grid_width, grid_height, magnitude)

    def __call__(self,x):
        #x =
        #print(x[0].size)
        out = self.op.perform_operation([x])



        return out[0]

    def __repr__(self):
        return self.__class__.__name__+'(probability={:},grid_width={:},grid_height={:},magnitude={:})'.format(
            self.probability,
                                                                                        self.grid_width,
                                                                       self.grid_height,
                                                                   self.magnitude)

class SaltPepperNoise:
    def __init__(self, mean=0,std=1):
        """
        Applies Gaussian noise to an image
        :param mean:
        :param std:
        """
        self.mean = mean
        self.std  = std

    def __call__(self, x):

        return x + np.random.randn(*x.size) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__+'(mean={:},std={:})'.format(self.mean,self.std)

class MapToInterval:
    """
    transforms image intensity to [0,1] interval
    """
    def __init__(self):
        self.min=0
        self.max=1


    def __call__(self,x):#image = (im - np.min(im)) / (np.max(im) - np.min(im))
        #https://stackoverflow.com/questions/72440228/how-to-rescale-a-pytorch-tensor-to-interval-0-1
        flattened_outmap = x.view(x.shape[0],-1,1,
                                       1)  # Use 1's to preserve the number of dimensions for broadcasting later,
        # as explained
        outmap_min,_ = torch.min(flattened_outmap,dim=1,keepdim=True)
        outmap_max,_ = torch.max(flattened_outmap,dim=1,keepdim=True)
        outmap = (x - outmap_min) / (outmap_max - outmap_min)  # Broadcasting rules apply

        return outmap

    def __repr__(self):
        return self.__class__.__name__+'(min={:},max={:})'.format(self.min,self.max)



#rotation_transform = MyRotationTransform(angles=[-30, -15, 0, 15, 30])



def create_train_test_transform(log_dir,write_params=True,kwargs={}):
    """
    Creates transforms for training set and validation set. Validation trans has no augmentation, just toPIL and
    ToTensor
    :param log_dir: string path to where log files need to be saved logfile contain info on params used for each
                    transformation
    :param write_params: whether to save logs about parameters
    :param kwargs: optional kwargs conataining params for each transform. see body
                    probability       = kwargs.get("probability",1)
                    max_rotation_left = kwargs.get("max_rotation_left",10.0)
                    max_rotation_right = kwargs.get("max_rotation_right",10.0)

                    max_factor = kwargs.get("max_factor",1.1)
                    min_factor = kwargs.get("min_factor",1.05)

                    grid_height = kwargs.get("grid_height",15)
                    grid_width = kwargs.get("grid_width",15)
                    magnitude  = kwargs.get("magnitude",1)
                    brightness = kwargs.get("brightness",2)
                    contrast = kwargs.get("contrast",2)
                    translate = kwargs.get("translate",0.1)
                    blur_max = kwargs.get("blur_max",2)
                    blur_min = kwargs.get("blur_min",0.1)
                    blur_kernel = kwargs.get("blur_kernel",5)
                    noise_sd    = kwargs.get("noise_sd",1/784)
    :return: tuple tran_trans, test_trans
    """

    probability       = kwargs.get("probability",1)
    max_rotation_left = kwargs.get("max_rotation_left",10.0)
    max_rotation_right = kwargs.get("max_rotation_right",10.0)

    max_factor = kwargs.get("max_factor",1.1)
    min_factor = kwargs.get("min_factor",1.05)

    grid_height = kwargs.get("grid_height",15)
    grid_width = kwargs.get("grid_width",15)
    magnitude  = kwargs.get("magnitude",1)
    brightness = kwargs.get("brightness",2)
    contrast = kwargs.get("contrast",2)
    translate = kwargs.get("translate",0.1)
    blur_max = kwargs.get("blur_max",2)
    blur_min = kwargs.get("blur_min",0.1)
    blur_kernel = kwargs.get("blur_kernel",5)
    noise_sd    = kwargs.get("noise_sd",1/784)

    # p = Augmentor.Pipeline()
    # # p.rotate(probability=1,max_left_rotation=max_rotation_left,max_right_rotation=max_rotation_right)
    # # p.zoom(probability=1,min_factor=min_factor,max_factor=max_factor)
    # p.random_distortion(grid_width=grid_width,grid_height=grid_height,magnitude=magnitude,probability=1)
    #
    #elastic_trans = Distort(probability, grid_width, grid_height, magnitude)


    trans_aug = transforms.RandomApply([RandomDistortion(1, grid_width, grid_height, magnitude),
                                        transforms.ColorJitter(brightness=brightness,contrast=contrast),
                                        transforms.RandomAffine(translate=(translate,translate),scale=(min_factor,max_factor),
                                                                degrees=(max_rotation_left,max_rotation_right)),
                                        #transforms.ToTensor(),
                                        transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(blur_min, blur_max)),
                                        SaltPepperNoise(mean=0,std=noise_sd)
                                                        ],p=probability)

    trans_train = transforms.Compose([
                                    transforms.ToPILImage(),
                                    trans_aug,
                                    transforms.ToTensor()
                                    #MapToInterval()
                                    ])
    trans_test = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor()
                                     ])


    # trans = transforms.Compose([
    #     p.torch_transform(),
    #     transforms.ColorJitter(brightness=brightness,contrast=contrast),
    #
    #     transforms.ToTensor()
    # ])
    if write_params:
        out_dict = dict(probability=probability,
                        max_rotation_right=max_rotation_right,
                        max_rotation_left=max_rotation_left,
                        max_factor = max_factor,
                        min_factor = min_factor,
                        grid_height = grid_height,
                        grid_width = grid_width,
                        magnitude = magnitude
                        )
        with open(os.path.join(log_dir,"transform_params.txt"),'w') as f:
            # for op in p.operations:
            #     f.write(str(op.__class__)+ '\n')
            for op in trans_aug.transforms:
                f.write(str(op.__repr__) + '\n')
            # for key,val in out_dict.items():
            #     f.write('%s:%s\n' % (key, val))



    return trans_train,trans_test


def test():
    try:
        from mypackage.DataManipulation import MnistDataset,DataLoader
    except ModuleNotFoundError:
        from DataManipulation import MnistDataset,DataLoader

    trans_train,trans_test = create_train_test_transform(log_dir='./')
    csvfile_train   = "Data/train.csv"
    csvfile_test  = "Data/Dig-MNIST.csv"
    out_shape = (1,28,28)
    data_gen_tr  = MnistDataset(csvfile_train,im_size=out_shape[1:],transforms=trans_train)
    data_gen_tst = MnistDataset(csvfile_train,im_size=out_shape[1:],transforms=trans_test)
    #im,lab = data_gen.__getitem__(0)
    train_data_loader = DataLoader(data_gen_tr,batch_size=10)
    test_data_loader = DataLoader(data_gen_tst,batch_size=10)


    for i,data in enumerate(train_data_loader):
        if i ==0:
            im = data[0]

            print(im.shape)
            assert im.to('cpu').numpy().shape==(10,1,28,28)
        else:
            break
    import matplotlib.pyplot as plt

    plt.imshow(im.to('cpu').numpy()[0,0,:,:])
    plt.show()
    for i,data in enumerate(test_data_loader):
        if i ==0:
            im = data[0]
            print(im.shape)
            assert im.to('cpu').numpy().shape==(10,1,28,28)
        else:
            break
    import matplotlib.pyplot as plt

    plt.imshow(im.to('cpu').numpy()[0,0,:,:])
    plt.show()


if __name__=='__main__':
    test()




    # kwargs={}
    # probability       = kwargs.get("probability",1)
    # max_rotation_left = kwargs.get("max_rotation_left",10.0)
    # max_rotation_right = kwargs.get("max_rotation_right",10.0)
    #
    # max_factor = kwargs.get("max_factor",1.1)
    # min_factor = kwargs.get("min_factor",1.05)
    #
    # grid_height = kwargs.get("grid_height",15)
    # grid_width = kwargs.get("grid_width",15)
    # magnitude  = kwargs.get("magnitude",1)
    # brightness = kwargs.get("brightness",2)
    # contrast = kwargs.get("contrast",2)
    #
    #
    # p = Augmentor.Pipeline('/media/adwaye/2tb/data/xray_data/joints')
    # p.rotate(probability=1,max_left_rotation=max_rotation_left,max_right_rotation=max_rotation_right)
    # p.zoom(probability=1,min_factor=min_factor,max_factor=max_factor)
    # p.random_distortion(grid_width=grid_width,grid_height=grid_height,magnitude=magnitude,probability=1)
    #
    #
    # sample = p.sample(10)
    # data_gen = MnistDataset(csvfile,im_size=out_shape[1:])
    # data_gen
    #
    #
    #
