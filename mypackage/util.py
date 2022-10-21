import os


def make_train_test_log_dir(model,log_loc='./logs',create_dirs=True):
    out_dir   = make_log_dir(model,log_loc,create_dirs=create_dirs)
    train_loc = os.path.join(out_dir,'train')
    val_loc   = os.path.join(out_dir,'test')
    if create_dirs:
        if not os.path.isdir(val_loc): os.makedirs(val_loc)
        if not os.path.isdir(train_loc): os.makedirs(train_loc)
    return train_loc,val_loc,out_dir



def make_log_dir(model,log_loc = './logs',create_dirs=True):
    """
        Uses the model name and creates a numbered folder withing model name
        example
    from Architectures import simple_model
    model = simple_model()
    x = make_lod_dir(model)
    print(x)
    './logs/simple_model/1'
    :param model:
    :param log_loc:
    :param create_dirs: bool whether to create directorues
    :return:
    """

    log_directory = os.path.join(log_loc,model._get_name())
    if create_dirs:
        if not os.path.isdir(log_directory): os.makedirs(log_directory)
    return update_dir(log_directory)


def update_dir(nameInit,create_dirs=True):
    """
    Functions that creates numbered folders within directory nameInit. If a folder with name x where x is a number
    exists, the function creates through i until x+i is no more a folder name. Then creates the folder with name
    nameInit/[x+i]
    :param nameInit: path within which the numbered folder needs to be located
    :return: path of the newly created folder
    example
    nameInit = ./
    ls ./ returns folder1 1 2 3
    x = update_dir('./)
    print(x)
    ./4
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
        from kannadamnistpackage.Architectures import simple_model
    except ModuleNotFoundError:
        from Architectures import simple_model
    model = simple_model()
    log_dir = make_log_dir(model)