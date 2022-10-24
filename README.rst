=============================
Introduction
=============================
This package provides the methods to reproduce the kaggle results found here:

https://www.kaggle.com/code/adwayerambojun/submission?scriptVersionId=108973950

#.  :ref:`Installation`.
#.  :ref:`Usage`.
        * :ref:`Creating a runOptions.py file`
        * :ref:`Training a neural network`
        * :ref:`Running inference with a trained model`
        * :ref:`Model visualisation with tensorboard`





Installation
=============================
You can either install it using ``setup.py`` or add the modules as part of the path.

============================
Using setup.py
============================
Clone the repository from https://github.com/adwaye/mnistPytorch

.. code-block::


    git clone https://github.com/adwaye/mnistPytorch [path_to_cloned_repo]
    cd mnistPytorch
    pip install -r requirements.txt
    python setup.py install

============================
Using ``sys.path.append``
============================
If you do not want to install the package you can instead add the following at the start of your python script to

.. code-block::

    import sys
    sys.path.append(path_to_cloned_repo)

where ``path_to_cloned_repo`` points to the folder containing the cloned repository


Usage
=============================

============================
Creating a runOptions.py file
============================
Create a file named ``runOptions.py`` (alternatively, copy the runOptions.py file found in the cloned repository. For more information on how these variables interact with the, see the package documentation.


.. code-block::

    nano runOptions.py

Paste the following contents in runOptions.py

.. code-block::

    #cnn params

    init_width = 32
    #dataloader params
    batch_size = 100
    l_rate = 0.001
    momentum = 0.0
    weight_decay = 0.0
    dampening = 0.0
    nesterov = 0  #0 or 1
    lr_gamma = 0.9
    epochs = 100

    #transforms params
    trans_probability = 0.5
    max_rotation_right = 9
    max_rotation_left = 9
    max_factor = 1.15
    min_factor = 1.05
    grid_height = 5
    grid_width = 5
    magnitude = 3
    brightness = 2
    contrast = 2
    translate = 0.1
    blur_max = 3
    blur_min = 0.5
    blur_kernel = 5
    opt_kwargs = dict(batch_size=batch_size,
                      l_rate=l_rate,
                      lr_gamma=lr_gamma,
                      momentum=momentum,
                      weight_decay=weight_decay,
                      dampening=dampening,
                      nesterov=nesterov,
                      probability=trans_probability,
                      max_rotation_right=max_rotation_right,
                      max_rotation_left=max_rotation_left,
                      max_factor=max_factor,
                      min_factor=min_factor,
                      grid_height=grid_height,
                      grid_width=grid_width,
                      magnitude=magnitude,
                      init_width=init_width,
                      brightness=brightness,
                      contrast=contrast,
                      translate=translate,
                      blur_kernel=blur_kernel,
                      blur_min=blur_min,
                      blur_max=blur_max,

                      )
    optimizer = 'adam'


==========================
Training a neural network
==========================

You can train any of the architectures found in ``mypackage.Architectures`` in the following way. The code pasted here reproduces the results obtained from


.. code-block::

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


======================================
Running inference with a trained model
======================================

The above code saves checkpoints for the trained model in ``./logs/mnist_model_bn_leaky/1``. Note that running this again saves a set of new checkpoints in ``./logs/mnist_model_bn_leaky/2``. To load a model and test it on the KannadaMnist test set run the following code:


.. code-block::

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
    submission_file = SAMPLE_SUBMISSION.copy()
    submission_file['label'] = predictions.cpu().numpy()
    submission_file.to_csv("submission.csv", index=False)


======================================
Model visualisation with tensorboard
======================================
If ``write_logs=True`` when initialising the trainer class, then tensorboard logs are written at ``./logs/mnist_model_pool_bn_leaky/1``. In fact one can compare different models if the ``log_dir`` argument is the same for different models. For example, one can 2 different models by running the following:

.. code-block::

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


    model      = mnist_model_pool_kern5(init_width=init_width)
    log_loc = './logs'
    my_trainer = Trainer(model,kwargs=opt_kwargs,log_dir=log_loc,create_save_loc=True,transform=True,
                     train_file=TRAIN_FILE,val_file=TEST_FILE)
    print("Training model "+model._get_name())
    if train_:
    my_trainer.train(epochs=epochs)
    del model
    del my_trainer
    torch.cuda.empty_cache()


Running the above creates multiple directories under ``./logs``. One can then compare models by running


.. code-block::
    tensorboard --logdir=./logs