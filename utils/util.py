import torch
from torch.autograd import Variable


def to_numpy(var, use_cuda=False):
    '''
    Converts pytorch tensor into numpy array
    '''
    return var.cpu().data.numpy() if use_cuda else var.data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=torch.FloatTensor):
    '''
    Converts numpy array into pytorch Variable
    '''
    return Variable(torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad).type(dtype)

def soft_update(target, source, tau):
    '''
    Updates the parameters of the target model towards the values given by the source model
    '''
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    '''
    Replaces the parameters of target with the ones in the source model
    '''
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
