import numpy as np

def get_normalized(value,var_min,var_max):
    '''
    Normalizes any 'value' which range lies between [var_min,var_max] to the interval [-1,1]
    '''
    norm_value = 2*(value - var_min)/(var_max - var_min) - 1
    return norm_value

def get_denormalized(value,var_min,var_max):
    '''
    De-normalizes any 'value' within the range [-1,1] to the range [var_min, var_max]
    '''
    denorm_value = (value + 1)*(var_max - var_min)/2 + var_min
    return denorm_value

def state_initialiser(shape,mode='g'):
    '''
    Return initialization vector for recurrent models. Can be a vector of zeros, or gaussian noise
    '''
    if mode == 'z': #Zero
        initial = np.zeros(shape=shape)
    elif mode == 'g': #Gaussian
        initial = np.random.normal(loc=0.,scale=1./float(shape[1]),size=shape)
    else: # May do some adaptive initialiser
        raise NotImplementedError

    return initial
