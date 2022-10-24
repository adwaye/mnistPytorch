=============================
Introduction
=============================
This package provides the methods to reproduce the kaggle results found here:

https://www.kaggle.com/code/adwayerambojun/submission?scriptVersionId=108973950

#.  :ref:`Installation`.
#.  :ref:`Usage`.
        * :ref:`Creating a runOptions.py file`
        * :ref:`Tranining a neural network`
        * :ref:`Inference from a trained model`





Installation
=============================
You can either install it using ``setup.py`` or add the modules as part of the path.

============================
Using setup.py
============================
Clone the repository from https://github.com/adwaye/mnistPytorch

.. code-block::

    pip install -r requirements.txt
    git clone https://github.com/adwaye/mnistPytorch [path_to_cloned_repo]
    cd mnisPytorch
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
===========================

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


The above code saves checkpoints for the trained model in ``./logs/mnist_model_bn_leaky/1``. Note that running this again saves a set of new checkpoints in ``./logs/mnist_model_bn_leaky/2``. To load a model and test it on the KannadaMnist test set run the following code:



