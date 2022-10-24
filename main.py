from mypackage.train import Trainer
from mypackage.DataManipulation import  load_train_csv,load_test_csv
import torch
from mypackage.Architectures import mnist_model_pool,mnist_model_pool_leaky,mnist_model_pool_kern5,\
    mnist_model_pool_inddim3,mnist_model_pool_inddim5,mnist_model_pool3fc,mnist_model_pool_bn_leaky
import pandas as pd
from runOptions import opt_kwargs,epochs,init_width

TRAIN_FILE = load_train_csv()
TEST_FILE  = load_test_csv()

train_ = True




model      = mnist_model_pool_bn_leaky(init_width=init_width)
log_loc = './logs'
my_trainer = Trainer(model,kwargs=opt_kwargs,log_dir=log_loc,create_save_loc=True,transform=True,
                     train_file=TRAIN_FILE,val_file=TEST_FILE)
print("Training model "+model._get_name())
if train_:
    my_trainer.train(epochs=epochs)
del model
del my_trainer
torch.cuda.empty_cache()


# model      = mnist_model_pool_kern5(init_width=init_width)
# log_loc = './logs'
# my_trainer = Trainer(model,kwargs=opt_kwargs,log_dir=log_loc,create_save_loc=True,transform=True,
#                      train_file=TRAIN_FILE,val_file=TEST_FILE)
# print("Training model "+model._get_name())
# if train_:
#     my_trainer.train(epochs=epochs)
# del model
# del my_trainer
# torch.cuda.empty_cache()

#
# model      = mnist_model_pool_inddim3(init_width=init_width)
# log_loc = './logs'
# my_trainer = Trainer(model,kwargs=opt_kwargs,log_dir=log_loc,create_save_loc=True,transform=True,
#                      train_file=TRAIN_FILE,val_file=TEST_FILE)
# print("Training model "+model._get_name())
# if train_:
#     my_trainer.train(epochs=epochs)
# del model
# del my_trainer
# torch.cuda.empty_cache()
#
#
#
# model      = mnist_model_pool_inddim5(init_width=init_width)
# log_loc = './logs'
# my_trainer = Trainer(model,kwargs=opt_kwargs,log_dir=log_loc,create_save_loc=True,transform=True,
#                      train_file=TRAIN_FILE,val_file=TEST_FILE)
# print("Training model "+model._get_name())
# if train_:
#     my_trainer.train(epochs=epochs)
# del model
# del my_trainer
# torch.cuda.empty_cache()
#
#
# model      = mnist_model_pool3fc(init_width=init_width)
# log_loc = './logs'
# my_trainer = Trainer(model,kwargs=opt_kwargs,log_dir=log_loc,create_save_loc=True,transform=True,
#                      train_file=TRAIN_FILE,val_file=TEST_FILE)
# print("Training model "+model._get_name())
# if train_:
#     my_trainer.train(epochs=epochs)
# del model
# del my_trainer
# torch.cuda.empty_cache()

