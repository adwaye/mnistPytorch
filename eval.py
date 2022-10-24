from mypackage.custom_transforms import create_train_test_transform


from mypackage.train import Trainer
from mypackage.DataManipulation import  load_train_csv,load_test_csv,MnistDataset,DataLoader,load_sample_submission
import torch
from mypackage.Architectures import mnist_model_pool,mnist_model_pool_leaky,mnist_model_pool_kern5,\
    mnist_model_pool_inddim3,mnist_model_pool_inddim5,mnist_model_pool3fc,mnist_model_pool_bn_leaky
import pandas as pd
from runOptions import opt_kwargs,epochs,init_width

TRAIN_FILE = load_train_csv()
TEST_FILE  = load_test_csv()
SAMPLE_SUBMISSION = load_sample_submission()
chckpt_path = "logs/mnist_model_pool_bn_leaky/1/checkpoint_epoch_100.pt"

train_ = True




model      = mnist_model_pool_bn_leaky(init_width=init_width)
log_loc = './logs'

checkpoint = torch.load(chckpt_path)
model.load_state_dict(checkpoint['net'])
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
train_trans,test_trans = create_train_test_transform(log_dir='./')
submission_dataset = MnistDataset(csvfile=TEST_FILE,im_size=(28,28),transforms=test_trans)

submission_loader  = DataLoader(submission_dataset,batch_size=100)
predictions = torch.LongTensor().to(device)
for i,data in enumerate(submission_loader):
    if i ==0:
        print(data[0].size())
    preds = model(data[0].to(device,dtype=torch.float))

    predictions = torch.cat((predictions, preds.argmax(dim=1)), dim=0)
print(predictions.shape)
submission_file = SAMPLE_SUBMISSION.copy()
submission_file['label'] = predictions.cpu().numpy()
submission_file.to_csv("submission.csv", index=False)