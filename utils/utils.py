
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
