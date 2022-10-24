import torch.nn
import torchvision.transforms.functional as TF
import random
import torchvision.transforms as transforms
import Augmentor,os
import numpy as np
from Augmentor.Operations import Distort


class RandomDistortion:
    """Applies elastic distortion with prob probability see https://augmentor.readthedocs.io/en/stable/ for info on
    grdi and magnitiude params

    :param probability: 0-1 probability of operation being applied to image
    :type probability: float
    :param grid_width:
    :type grid_width: int
    :param grid_height:
    :type grid_height: int
    :param magnitude:
    :type magnitude: float
    """
    def __init__(self,probability, grid_width, grid_height, magnitude):


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
        return self.__class__.__name__+f'(probability={self.probability},grid_width={self.grid_width},grid_height=' \
                                       f'{self.grid_height},' \
                                       f'magnitude={self.magnitude})'


class SaltPepperNoise:
    """Applies Gaussian noise to an image

    :param mean: mean of gaussian noise
    :type mean: float
    :param std: standard deviation of gaussian noise
    :type std: float
    """
    def __init__(self, mean=0,std=1):


        self.mean = mean
        self.std  = std

    def __call__(self, x):

        return x + np.random.randn(*x.size) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__+f'(mean={self.mean},std={self.std})'

class MapToInterval:
    """transforms image intensity to [0,1] interval

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
        return self.__class__.__name__+f'(min={self.min},max={self.max})'



#rotation_transform = MyRotationTransform(angles=[-30, -15, 0, 15, 30])



def create_train_test_transform(log_dir,write_params=True,kwargs={}):
    """

    :param log_dir: string path to where log files need to be saved logfile contain info on params used for each
                    transformation
    :type log_dir: string
    :param write_params: whether to save logs about parameters
    :type write_params: bool
    :param kwargs: optional kwargs conataining params for each transform
                    list of keys that will be used:
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
    :type kwargs: dict
    :return: trains_trans and test_trans: transformations to be used on trainining data and validation data respectively
    :rtype: tuple


    .. code-block::

       opt_kwargs = dict(
              probability=trans_probability, #probability of applying data augmentation to \n
              max_rotation_right=max_rotation_right, #max rotation for affine transform in data aug \n
              max_rotation_left=max_rotation_left, #min rotation for affine transform in data aug \n
              max_factor=max_factor, #max scale for affine transform in data aug \n
              min_factor=min_factor, #min scale for affine transform in data aug \n
              grid_height=grid_height, #grid height for elastic transform \n
              grid_width=grid_width, #grid width for elastic transform \n
              magnitude=magnitude, #magnitude of elastic transform \n
              init_width=init_width, #number of stacks in first conv layer of nn \n
              brightness=brightness, #brightness of color jitter transform \n
              contrast = contrast, #contrast in color jitter transform see https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html \n
              translate=translate, #fraction by which to translate in affine transform see https://pytorch.org/vision/main/generated/torchvision.transforms.RandomAffine.html \n
              blur_kernel = blur_kernel,  #paramters of gaussian blur see https://pytorch.org/vision/main/generated/torchvision.transforms.GaussianBlur.html \n
              blur_min = blur_min, \n
              blur_max = blur_max, \n
              noise_sd = noise_sd #sd of gaussian noise transform see custom_transform.SaltPepperNoise \n
              )
       train_trans, test_trans = create_train_test_transform(kwargs=opt_kwarsgs)


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


def _test():
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
    _test()



