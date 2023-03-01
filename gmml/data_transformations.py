"""
this is a dummy file because I previously saved a checkpoint with this module included in it.
The actual data transformation is in dataloaders folder

"""



class DataTransform_finetune(object):
    def __init__(self, augment, input_size, in_chans=1, data_mean=None, data_std=None):
        
        return

    def __call__(self, image):
        
        return


class DataTransform_pretrain(object):
    def __init__(self, augment, input_size, in_chans=1, data_mean=None, data_std=None):

        return

    def __call__(self, image):
        
        return



class DataAugmentation_MCSSL(object):

    def __init__(self, args, transform=None):
        
        return

    def GMML_drop_rand_patches(self, X, max_block_sz=0.3):
        return
  
    def __call__(self, img):
        return
            
