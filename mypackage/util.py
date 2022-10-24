import os


def make_train_test_log_dir(model,log_loc='./logs',create_dirs=True):
    """Creates a train, test output directory where model parameters are saved

    :param model:  model whose logs need to be saved, directory will have name model.__get_name()
    :type model: pytorch.nn.module
    :param log_loc: path pointing where new directory should be created
    :type log_loc: string
    :param create_dirs: if true, log dirs are created
    :type create_dirs: bool
    :return: tuple containing train_loc,test_loc,oudir, train_loc = outdir/train ttest_loc=outdir/test outdir: path
             pointing to log_loc/model.__get_name()
    :rtype: tuple()



    >>> from mypackage.Architectures import simple_model
    >>> model = simple_model()
    >>> x = make_train_test_log_dir(model)
    >>> print(x)
    ['./logs/simple_model/1/train','./logs/simple_model/1/test','./logs/simple_model/1']

    """
    out_dir   = make_log_dir(model,log_loc,create_dirs=create_dirs)
    train_loc = os.path.join(out_dir,'train')
    val_loc   = os.path.join(out_dir,'test')
    if create_dirs:
        if not os.path.isdir(val_loc): os.makedirs(val_loc)
        if not os.path.isdir(train_loc): os.makedirs(train_loc)
    return train_loc,val_loc,out_dir



def make_log_dir(model,log_loc = './logs',create_dirs=True):
    """makes an expriment directory for a pytorch model

    :param model:  model whose logs need to be saved, directory will have name model.__get_name()
    :type model: pytorch.nn.module
    :param log_loc: path pointing where new directory should be created
    :type log_loc: string
    :param create_dirs: if true, log dirs are created
    :type create_dirs: bool
    :return: location of newly created folder
    :rtype: string


    >>> from mypackage.Architectures import simple_model
    >>> model = simple_model()
    >>> x = make_log_dir(model)
    >>> print(x)
    './logs/simple_model/1'

    """


    log_directory = os.path.join(log_loc,model._get_name())
    if create_dirs:
        if not os.path.isdir(log_directory): os.makedirs(log_directory)
    return update_dir(log_directory)


def update_dir(nameInit,create_dirs=True):
    """Functions that creates numbered folders within directory nameInit. If a folder with name x where x is a number
    exists, the function creates through i until x+i is no more a folder name. Then creates the folder with name
    nameInit/[x+i]

    :param nameInit: path within which the numbered folder needs to be located
    :type nameInit: str
    :param create_dirs:  whether to create the directories if they are not present, default to True
    :type create_dirs: bool
    :return: newly created directory
    :rtype: None



    >>> import os
    >>> nameInit = ./
    >>> os.listdir(nameInit)
    1 2
    >>> x = update_dir(nameInit)
    >>> os.listdir(nameInit)
    1/ 2/ 3/
    >>> x
    3

    """

    output_directory = nameInit
    if not os.path.exists(output_directory):
        output_directory = os.path.join(output_directory,'1')
    else:
        highest_num = 0
        for f in os.listdir(output_directory):
            if os.path.exists(os.path.join(output_directory, f)):
                file_name = os.path.splitext(f)[0]
                try:
                    file_num = int(file_name)
                    if file_num > highest_num:
                        highest_num = file_num
                except ValueError:
                    print('The file name "%s" is not an integer. Skipping' % file_name)

        output_directory = os.path.join(output_directory, str(highest_num + 1))
    if create_dirs: os.makedirs(output_directory)
    return output_directory





if __name__=='__main__':
    try:
        from mypackage.Architectures import simple_model
    except ModuleNotFoundError:
        from Architectures import simple_model
    model = simple_model()
    log_dir = make_log_dir(model)