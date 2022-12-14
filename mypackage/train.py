import matplotlib.pyplot as plt
import torch, os, shutil
from pathlib import Path
import torch.optim as optim



from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix
import seaborn as sns

try:
    from mypackage.util import make_train_test_log_dir, make_log_dir
    from mypackage.custom_transforms import create_train_test_transform
    from mypackage.Architectures import simple_model,count_nn_params,nn
    from mypackage.DataManipulation import *
except ModuleNotFoundError:
    from util import make_train_test_log_dir, make_log_dir
    from custom_transforms import create_train_test_transform
    from Architectures import simple_model,count_nn_params,nn
    from DataManipulation import *




class Trainer(object):
    """

    :param model: model of class nn.Module, please use classes present in mypackage.Architectures as
                  the model
    :param optimizer: choice of 'adam' or 'momentum'
    :param device: 'cuda' or 'cpu': device on which to run training and inference
    :param transform: list of transform needs to be output of torchvision.transforms.Compose and needs to have
                    the form
                                               transforms = trns.Compose([trns.ToPILImage(),
                                                  transform1,
                                                  transform2,
                                                  .
                                                  .
                                                  .,
                                                  transformN,
                                                  trns.ToTensor()
                                ])

                       )
    :param train_file: pd.dataframe (result of load_csv) or path to csb containing training data, needs to be same
                        format as mnist dataset
    :param val_file: redundant
    :param log_dir:  string where to save the tensorboard output and the model, and params info
    :param create_save_loc: bool whether to create new directory withing log_dir or to save withing log_dir when saving
    :param write_logs: bool whether to save logs
    :param kwargs: dict, extra arguments to be passed
    example:




    .. code-block::

       from mypackage.Architectures import simple_model
       from mypackage.train import Trainer
       my_model = simple_model()
       TEST_FILE = "./data/test.csv"
       TRAIN_FILE = "./data/train.csv"
       log_loc   = "./logs"
       opt_kwargs = dict(batch_size=batch_size,#data batch size \n
              l_rate=l_rate, #initial learning rate \n
              lr_gamma=lr_gamma, #decay rate for lr scheduler \n
              momentum=momentum, #momentum for sgd, if sgd is selected \n
              weight_decay=weight_decay, #weight decay parameter for optimizer \n
              dampening=dampening, #dampening for adam and sgd \n
              nesterov=nesterov, #whether to use nesterov in sgd \n
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
       my_trainer = Trainer(model,kwargs=opt_kwargs,log_dir=log_loc,create_save_loc=True,transform=True,
                     train_file=TRAIN_FILE,val_file=TEST_FILE,write_logs=True)

    """
    def __init__(self,model,optimizer='Adam',device='cuda',transform=False,train_file =
    "Data/train.csv",val_file = "Data/Dig-MNIST.csv",log_dir='./logs',create_save_loc=True,write_logs=True,kwargs={}):


        self.write_logs = write_logs
        self.model = model
        self.kwargs = kwargs
        if create_save_loc:
            self._create_save_locs(log_dir)
        else:
            self._create_train_test_locs(log_dir)
        self._get_optimizer(optimizer)
        self._get_scheduler()
        if device=='cuda':
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)

        self._get_loss_function()
        if transform:
            self._init_transform(kwargs)
        else:
            self.transform = None
        self._init_data_pipeline(train_file,val_file)



        self._init_writers()


    def _init_transform(self,kwargs):
        #self.transform = create_transform(log_dir=self.model_outdir,write_params=True,kwargs=kwargs)
        self.train_trans,self.val_trans = create_train_test_transform(log_dir=self.model_outdir,write_params=True,
                                                             kwargs=kwargs)


    def _get_optimizer(self,optimizer):
        l_rate = self.kwargs.get("l_rate",0.01)
        weight_decay = self.kwargs.get("weight_decay",0.0)
        if optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(),lr=l_rate,weight_decay=weight_decay)
        elif optimizer.lower()=='momentum':
            momentum     = self.kwargs.get("momentum",0.0)

            dampening = self.kwargs.get("dampening",0.0)
            nesterov = bool(self.kwargs.get("nesterov",0))

            self.optimizer = optim.SGD(self.model.parameters(),lr=l_rate,momentum=momentum,weight_decay=weight_decay,
                                  dampening=dampening,nesterov=nesterov)
        else:
            print('optimizer class with name '+optimizer+' not implemented yet')
            self.optimizer=None

    def _get_scheduler(self):
        lr_gamma = self.kwargs.get("lr_gamma",1.0)
        if self.optimizer is not None:
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=lr_gamma)

    def _get_loss_function(self):
        """
        Returns the loss function to be used during training.
        Currently only returns the crossentropy loss, but can be extended to take a custom loss_fn or a different one
        :param loss_fn:
        :return:
        """
        self.loss_fn= nn.CrossEntropyLoss()

    def _init_data_pipeline(self,train_file,test_file):
        # if split_train_file:
        try:
            if os.path.isfile(train_file):
                df = pd.read_csv(train_file)
        except:
            print('trainfile is not a string, so it is probably a dataframe')
            df = train_file
        train_csv, val_csv = train_test_split(df.iloc[:,0:], test_size=0.2)
        train_csv.reset_index(drop=True,inplace=True)
        val_csv.reset_index(drop=True,inplace=True)



        train_dataset = MnistDataset(train_csv,im_size=(28,28),transforms=self.train_trans)

        test_dataset = MnistDataset(val_csv,im_size=(28,28),transforms=self.val_trans)


        # train_dataset = MnistDataset(train_file,im_size=(28,28),transforms=self.train_trans)
        # test_dataset = MnistDataset(test_file,im_size=(28,28),transforms=self.val_trans)

        batch_size = self.kwargs.get("batch_size",100)
        self.train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        self.test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    def _create_save_locs(self,log_dir):
        self.train_log_dir,self.test_log_dir,self.model_outdir = make_train_test_log_dir(model=self.model,
                                                                                        log_loc=log_dir,
                                                                                         create_dirs=self.write_logs)

    def _create_train_test_locs(self,log_dir):
        self.model_outdir = log_dir
        self.train_log_dir = os.path.join(log_dir,'train')

        self.test_log_dir = os.path.join(log_dir,'test')
        if self.write_logs:
            if not os.path.isdir(self.train_log_dir): os.makedirs(self.train_log_dir)
            if not os.path.isdir(self.test_log_dir): os.makedirs(self.test_log_dir)

    def _init_writers(self):
        if self.write_logs:
            self.train_writer = SummaryWriter(log_dir=self.train_log_dir)
            self.test_writer = SummaryWriter(log_dir=self.test_log_dir)
        else:
            self.train_writer = None
            self.test_writer  = None


    def _train_block(self,global_step):
        global_step = train_block(device=self.device,model=self.model,optimizer=self.optimizer,scheduler=self.scheduler,
                    data_loader=self.train_loader,writer=self.train_writer,global_step=global_step,
                                  loss_fn=self.loss_fn,write_logs = self.write_logs)
        return global_step
    def _eval_block(self,global_step):
        eval_block(device=self.device,model=self.model,data_loader=self.test_loader,writer=self.test_writer,
                   global_step=global_step,
                   loss_fn=self.loss_fn,write_logs = self.write_logs)

    def _save_checkpt(self,epoch):
        save_checkpt(model=self.model,optimizer=self.optimizer,scheduler=self.scheduler,epoch=epoch,
                     out_dir=self.model_outdir)

    def train(self,epochs=2,resume_training=False):
        """trains the model

        :param epochs: number of epochs
        :type epochs: int
        :param resume_training: currently not being used
        :type resume_training: bool
        :return:
        :rtype:
        """


        global_step = 0
        self._create_log_files()
        for epoch in range(epochs):
            print(f"======epoch={epoch}")
            global_step = self._train_block(global_step=global_step)
            self._eval_block(global_step=global_step)
            if self.write_logs:
                self._save_checkpt(epoch=epoch+1)


    def _create_log_files(self):

        with open(os.path.join(self.model_outdir,'training_setup.txt'),'w') as f:
            for key,val in self.kwargs.items():
                f.write('%s:%s\n' % (key,val))










def train_block(device,model,optimizer,scheduler,data_loader,writer,global_step,loss_fn=nn.CrossEntropyLoss(),
                write_logs=True):
    """Perform forward backward passes for one epoch, can be used to train torch.nn.module type models

    :param device:
    :type device: torch.device
    :param model:
    :type model: torch.nn.module
    :param optimizer: optimizer to be used
    :type optimizer: torch.optim.optimizer
    :param scheduler: learning rate scheduler
    :type scheduler: torch.optim
    :param data_loader: dataloader to be used to generate training data batches
    :type data_loader: torch.utils.data.DataLoader
    :param writer: logging writer to be used
    :type writer: torch.utils.data.SummaryWriter
    :param global_step: current step of training algorithm
    :type global_step: int
    :param loss_fn:
    :type loss_fn: torch.nn.losses see https://pytorch.org/docs/stable/nn.html#loss-functions
    :param write_logs: default True, if False, writer does not write any log files
    :type write_logs: bool
    :return: global_step, number of forward-backward passes is added to input global_step
    :rtype: int
    """


    model.train()
    scheduler.step()
    for i,data in enumerate(data_loader,0):
        global_step+=1
        inputs,labels = data[0].to(device,dtype=torch.float),data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs,labels)
        loss.backward()
        optimizer.step()


        #training_loss += loss.item()

        if global_step % 300 == 99:  # print every 2000 mini-batches
            # pred_score,pred_class = outputs.max(dim=1)
            # training_accuracy = (pred_class == labels).sum().item()
            # training_accuracy /= inputs.shape[0]
            # training_loss = loss.item()
            # print(f'[{epoch + 1}, {global_step:5d}] loss: {training_loss :.3f} accuracy: {training_accuracy}')

            # train_writer.add_scalar('accuracy',training_accuracy,global_step=global_step)
            # train_writer.add_scalar('loss',training_loss,global_step=global_step)
            add_to_writer(writer,loss_fn,inputs,outputs,labels,iter=global_step,write_logs=write_logs)

    return global_step



def eval_block(device,model,data_loader,writer,global_step,loss_fn=nn.CrossEntropyLoss(),write_logs=True):
    """Performs inference on the given dataset

    :param device:
    :type device: torch.device
    :param model:
    :type model: torch.nn.module
    :param data_loader: dataloader to be used to generate training data batches
    :type data_loader: torch.utils.data.DataLoader
    :param writer: logging writer to be used to store accuracy and losses on data_loader
    :type writer: torch.utils.data.SummaryWriter
    :param global_step: current step of training algorithm
    :type global_step: int
    :param loss_fn:
    :type loss_fn: torch.nn.losses see https://pytorch.org/docs/stable/nn.html#loss-functions
    :param write_logs: default True, if False, writer does not write any log files
    :type write_logs: bool
    :return: global_step, number of forward-backward passes is added to input global_step
    :rtype: int
    """


    eval_loss     = 0.0
    eval_accuracy = 0.0
    model.eval()
    with torch.no_grad():
        input_list = []
        label_list = []
        output_list = []
        for i,data in enumerate(data_loader,0):

            inputs,labels = data[0].to(device,dtype=torch.float),data[1].to(device)

            outputs =model(inputs)
            input_list +=[inputs]
            output_list += [outputs]
            label_list += [labels]


            #loss = loss_fn(outputs,labels)

            #pred_score,pred_class = outputs.max(dim=1)
            # eval_accuracy += (pred_class == labels).sum().item()
            # eval_loss     += loss.item()*inputs.shape[0]

        merged_inputs  = torch.cat(input_list,axis=0)
        merged_outputs = torch.cat(output_list,axis=0)
        merged_labels  = torch.cat(label_list,axis=0)
        add_to_writer(writer,loss_fn,merged_inputs,merged_outputs,merged_labels,iter=global_step,write_logs=write_logs)





def make_prediction_fig(inputs,pred_class,labels,iter):
    """plots images given in inputs with their corresponding predicted class and true class

    :param inputs: input datat to the model that produces outputs
    :type inputs: pytorch.Tensor
    :param outputs: outputs produced by modelbeing logged, should have shape [batch_size,1]
    :type outputs: pytorch.Tensor
    :param labels: true labels  should have shape [batch_size,1]
    :type labels: pytorch.Tensor
    :param iter: step at which logs are being logged in training loop
    :type iter: int
    :return:
    :rtype:
    """

    fig = plt.figure(figsize=(12, 16))
    for idx in np.arange(16):
        ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
        ax.imshow(inputs[idx,0,:,:].cpu(),cmap='Greys_r')
        ax.set_title(f"Predicted Class {pred_class[idx].cpu()},\n true label={labels[idx].cpu()}")
    plt.tight_layout()
        # ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(

        #     classes[preds[idx]],
        #     probs[idx] * 100.0,
        #     classes[labels[idx]]),
        #             color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


def make_confusion_matrix(inputs,pred_class,labels,iter):
    """makes a confusion matrix from the predicted_classes produced by some model being evaluated on inputs and true
    labels

    :param inputs: input datat to the model that produces outputs
    :type inputs: pytorch.Tensor
    :param outputs: outputs produced by modelbeing logged, should have shape [batch_size,1]
    :type outputs: pytorch.Tensor
    :param labels: true labels  should have shape [batch_size,1]
    :type labels: pytorch.Tensor
    :param iter: step at which logs are being logged in training loop
    :type iter: int
    :return:
    :rtype:
    """
    #TODO: THIS IS NOT DOING WHAT i WANT IT TO DO
    print("input to confusion matrix")
    print(labels.shape)
    print(pred_class.shape)
    cf_matrix = confusion_matrix(labels.cpu().detach().numpy(),pred_class.cpu().detach().numpy())
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix,axis=0,keepdims=True),index=[i for i in range(10)],
                         columns=[i for i in range(10)])

    # Plot the dataframe using seaborn
    fig = plt.figure(figsize=(12,7))
    sns.heatmap(df_cm,annot=True)
    plt.title(f"Neural Network Confusion Matrix step{iter}",fontsize=20)
    return fig

def add_to_writer(writer,loss_fn,inputs,outputs,labels,iter,write_logs):
    """Function that adds performance metrics ro a tensorboard summary writer at training glbal step=iter
    also prints the accuracy and loss metrics


    :param writer:
    :type writer: torch.utils.data.SummaryWriter
    :param loss_fn: loss function used to evaluate model loss
    :type loss_fn: torch.nn.losses see https://pytorch.org/docs/stable/nn.html#loss-functions
    :param inputs: input datat to the model that produces outputs
    :type inputs: pytorch.Tensor
    :param outputs: outputs produced by modelbeing logged, should have shape [batch_size,1]
    :type outputs: pytorch.Tensor
    :param labels: true labels  should have shape [batch_size,1]
    :type labels: pytorch.Tensor
    :param iter: step at which logs are being logged in training loop
    :type iter: int
    :param write_logs: if True, writer saves logging info
    :type write_logs: bool
    :return:
    :rtype:



    """

    pred_score,pred_class = outputs.max(dim=1)
    accuracy = (pred_class == labels).sum().item()
    accuracy /= inputs.shape[0]
    loss = loss_fn(outputs,labels)
    loss_val = loss.item()
    print(f'global_step{iter:5d} loss: {loss_val :.3f} accuracy: {accuracy}')
    if write_logs:
        writer.add_scalar('accuracy',accuracy,global_step=iter)
        writer.add_scalar('loss',loss_val,global_step=iter)
        fig = make_prediction_fig(inputs,pred_class,labels,iter)
        writer.add_figure('predictions',fig,global_step=iter)
    # fig = make_confusion_matrix(inputs,pred_class,labels,iter)

    # writer.add_figure('Confusion matrix',fig,global_step=iter)


def save_checkpt(model,optimizer,scheduler,epoch,out_dir,max_keep=10):
    """saves the state of the model, optimizer and scheduler in a training loop to output_loc
    where output_loc=out_dir/model.__get_name()/checkpoint_epoch_[epoch].pt

    :param model:
    :type model: torch.nn.module
    :param optimizer: optimizer to be used
    :type optimizer: torch.optim.optimizer
    :param scheduler: learning rate scheduler
    :type scheduler: torch.optim
    :param epoch:
    :type epoch: int
    :param out_dir: location where model should be saved
    :type out_dir: string
    :param max_keep: how many checkpoints to keep>1
    :type max_keep: int
    :return:
    :rtype:
    """
    checkpoints = [f for f in os.listdir(out_dir) if 'checkpoint' in f]
    if len(checkpoints)>max_keep:
        for f in checkpoints:
            os.remove(os.path.join(out_dir,f))

    file_path = os.path.join(out_dir,f'checkpoint_epoch_{epoch}.pt')
    torch.save({'epoch'     : epoch,
                'net'       : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()},file_path
               )



def load_checkpt(model,optimizer,scheduler,epoch,out_dir):
    """loads the state of the model, optimizer and scheduler saved at outdir
    will try to access the checkpoint at out_dir/model.__get_name()/checkpoint_epoch_[epoch].pt

    :param model:
    :type model: torch.nn.module
    :param optimizer: optimizer to be used
    :type optimizer: torch.optim.optimizer
    :param scheduler: learning rate scheduler
    :type scheduler: torch.optim
    :param epoch:
    :type epoch: int
    :param out_dir: location where model should be saved
    :type out_dir: string
    :return: model,optimizer,scheduler: model, optimizer and scheduler whose states have been loaded
    :rtype: tuple
    """
    file_path = os.path.join(out_dir,f'checkpoint_epoch_{epoch}.pt')
    print('loading '+file_path)
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model,optimizer,scheduler





def _test_trainer():
    """unittest for trainer

    :return:
    :rtype:
    """


    model = simple_model()




    batch_size = 50
    l_rate = 0.2
    momentum = 0.9
    weight_decay = 0.75
    dampening    = 0.0
    nesterov = 0
    lr_gamma = 0.75
    epochs =2
    #transforms params
    trans_probability = 0.4
    max_rotation_right = 9
    max_rotation_left = 9
    max_factor = 1.11
    min_factor = 1.05
    grid_height = 6
    grid_width = 6
    magnitude  = 3
    brightness = 3
    contrast = 3
    opt_kwargs = dict(batch_size = batch_size,
                      l_rate=l_rate,
                      lr_gamma =lr_gamma,
                      momentum=momentum,
                      weight_decay=weight_decay,
                      dampening = dampening,
                      nesterov = nesterov,
                      probability = trans_probability,
                      max_rotation_right= max_rotation_right,
                      max_rotation_left = max_rotation_left,
                      max_factor = max_factor,
                      min_factor = min_factor,
                      grid_height=grid_height,
                      grid_width=grid_width,
                      magnitude=magnitude,
                      brightness=brightness,
                      contrast=contrast,


                      )

    #checking that the save_loc is now different
    log_loc = './test_logs'
    if os.path.isdir(log_loc):
        shutil.rmtree(log_loc)

    train_file = load_train_csv()
    val_file   = load_dig_csv()
    my_trainer = Trainer(model,kwargs=opt_kwargs,log_dir=log_loc,create_save_loc=False,transform=True,
                         train_file=train_file,val_file=val_file)
    assert my_trainer.train_log_dir.split('/')[-1] == 'train'
    assert my_trainer.test_log_dir.split('/')[-1] == 'test'
    assert my_trainer.model_outdir == log_loc


    log_loc = './test_logs'
    if os.path.isdir(log_loc):
        shutil.rmtree(log_loc)

    my_trainer = Trainer(model,kwargs=opt_kwargs,log_dir=log_loc,create_save_loc=True,transform=True,train_file=train_file,val_file=val_file)
    assert os.path.isfile(os.path.join(my_trainer.model_outdir,'transform_params.txt'))
    with open(os.path.join(my_trainer.model_outdir,'transform_params.txt'),'r') as f:
        i = 0
        for line in f.readlines():
            if i>2:
                print(line)

                # param,val = line.split(':')
                # assert opt_kwargs[param]==np.float(val)
            i+=1




    #my_trainer._get_optimizer('adam')

    #test optimizer and scheduler options
    assert my_trainer.optimizer.param_groups[-1]['lr']==l_rate
    assert my_trainer.optimizer.param_groups[-1]['weight_decay'] == weight_decay
    my_trainer._get_optimizer('momentum')
    assert my_trainer.optimizer.param_groups[-1]['momentum']==momentum
    assert my_trainer.optimizer.param_groups[-1]['weight_decay'] == weight_decay
    assert my_trainer.optimizer.param_groups[-1]['nesterov'] == nesterov
    assert my_trainer.optimizer.param_groups[-1]['dampening'] == dampening


    assert my_trainer.scheduler.gamma == lr_gamma

    #check if being saved in the correct place
    assert my_trainer.train_log_dir.split('/')[-1] == 'train'
    assert my_trainer.test_log_dir.split('/')[-1] == 'test'
    assert my_trainer.model_outdir.split(('/'))[-1] == '1'
    assert my_trainer.model_outdir.split('/')[-2] == model._get_name()
    assert os.path.dirname(my_trainer.model_outdir) == os.path.join(log_loc,model._get_name())

    my_trainer.train(epochs=epochs)
    with open(os.path.join(my_trainer.model_outdir,'training_setup.txt'),'r') as f:
        i = 0
        for line in f.readlines():
            print(line)
            param,val = line.split(':')

            assert opt_kwargs[param]==np.float(val)


            i+=1
    #check if learning rate is being updated
    assert my_trainer.scheduler.get_last_lr()[-1] < l_rate
    assert np.abs(my_trainer.scheduler.get_last_lr()[-1] - l_rate * lr_gamma ** epochs) < 10e-6


    model_loaded,optimizer_loaded,scheduler_loaded = load_checkpt(model=my_trainer.model,optimizer=my_trainer.optimizer,
                                                                  scheduler=my_trainer.scheduler,
                                                      out_dir=my_trainer.model_outdir,epoch=1)

    assert optimizer_loaded.state_dict().keys() == my_trainer.optimizer.state_dict().keys()

    assert np.abs(scheduler_loaded.get_last_lr()[-1] - l_rate * lr_gamma ** 1) < 10e-6
    for param,param_loaded in zip(my_trainer.model.parameters(),model_loaded.parameters()):
        assert torch.equal(param,param_loaded)



def test_confusion_matrix():
    """test function for confusion matrix
    
    :return: 
    :rtype: 
    """
    y_true = np.random.randint(low=0,high=10,size=50,dtype=int)

    y_pred = np.random.randint(low=0,high=9,size=50,dtype=int)
    cf_matrix = confusion_matrix(y_true,y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix,axis=0,keepdims=True),index=[i for i in range(10)],
                         columns=[i for i in range(10)])

    # Plot the dataframe using seaborn
    fig = plt.figure(figsize=(12,7))
    sns.heatmap(df_cm,annot=True)
    plt.title(f"Neural Network Confusion Matrix step{iter}",fontsize=20)





if __name__=='__main__':
    _test_trainer()
