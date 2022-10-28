from mypackage.custom_transforms import *
from mypackage.DataManipulation import *



train_trans,test_trans = create_train_test_transform(log_dir='./sample_logs',write_params=False)
train_file = load_train_csv()
dataset = MnistDataset(train_file,im_size=(28,28))



fig,ax = plt.subplots(ncols=8,nrows=2,figsize=(16,5))
for i in range(8):
    image,label = dataset.__getitem__(i)
    trans_image = train_trans(image)
    if i==0:
        ax[0,i].set_ylabel("Original Image")
        ax[1,i].set_ylabel("Transformed Image")
    ax[0,i].imshow(image[0],cmap='Greys_r')
    ax[0,i].set_title(f'label={label}')
    ax[0,i].set_xticks([])
    ax[0,i].set_yticks([])
    ax[1,i].imshow(trans_image[0],cmap='Greys_r')
    ax[1,i].set_title(f'label={label}')
    ax[1,i].set_xticks([])
    ax[1,i].set_yticks([])

fig.savefig('/home/adwaye/Documents/presentations/presentationGSK/figures/augmented_images.png',bbox='tight')



