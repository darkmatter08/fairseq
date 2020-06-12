'''
Define the new module that using meProp
Both meProp and unified meProp are supported
'''
import math
import torch
import torch.nn as nn

from torch.nn.parameter import Parameter

from .jain_functions import linear, linearUnified, linear_crs, linearUnified_shawn


class Linear_meProp(nn.Module):
    '''
    A linear module (layer without activation) with meprop
    The initialization of w and b is the same with the default linear module.
    '''

    def __init__(self, in_, out_, k, unified=False):
        super(Linear_meProp, self).__init__()
        self.in_ = in_
        self.out_ = out_
        self.k = k
        self.unified = unified

        self.w = Parameter(torch.Tensor(self.in_, self.out_))
        self.b = Parameter(torch.Tensor(self.out_))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_)
        self.w.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # Note: Need to modify this to work with 3-tensor inputs -- see LinearCRS.forward()
        input_shape = x.shape
        if len(input_shape) > 2:
            # TODO fix this shaping for linear_meProp. conventions around w shape are different.
            common_dim = input_shape[-1]
            assert common_dim == self.w.shape[0]  # since we multiply with w
            outer_dim = torch.prod(torch.tensor(input_shape[:-1]))
            x = x.reshape(outer_dim, common_dim)
        if self.unified:
            result = linearUnified(self.k)(x, self.w, self.b)
        else:
            result = linear(self.k)(x, self.w, self.b)
        if len(input_shape) > 2:
            result = result.reshape(input_shape[:-1] + (self.w.shape[-1],))
        return result

    def __repr__(self):
        if self.unified:
            layer_description = 'unified'
        else:
            layer_description = ''
        return '{} ({} -> {} <- {}k={})'.format(self.__class__.__name__,
                                              self.in_, self.out_, layer_description, self.k)


class LinearCRS(nn.Module):
    '''
    A linear module (no activation function) with CRS.
    The initialization of w and b is the same with the default linear module.
    '''

    def __init__(self, in_, out_, k, strategy='random'):
        super(LinearCRS, self).__init__()
        self.in_ = in_
        self.out_ = out_
        self.k = k
        self.strategy = strategy

        # TODO Modernize this code, change Paramters to simple Tensors.
        self.w = Parameter(torch.Tensor(self.out_, self.in_))
        self.b = Parameter(torch.Tensor(self.out_))
        assert self.w.requires_grad
        assert self.b.requires_grad

        self.reset_parameters()

    def reset_parameters(self):
        # TODO: why this initialization? Why not a non-uniform init?
        stdv = 1. / math.sqrt(self.out_)
        self.w.data.uniform_(-stdv, stdv)
        self.b.data.zero_()  # note mismatch in init vs Linear()

    def forward(self, x):
        input_shape = x.shape
        if len(input_shape) > 2:
            common_dim = input_shape[-1]
            assert common_dim == self.w.shape[-1]  # since we multiply with w.T
            outer_dim = torch.prod(torch.tensor(input_shape[:-1]))
            x = x.reshape(outer_dim, common_dim)
        result = linear_crs(k=self.k, strategy=self.strategy)(x, self.w, self.b)
        if len(input_shape) > 2:
            result = result.reshape(input_shape[:-1] + (self.w.shape[0],))
        return result

    def __repr__(self):
        # TODO add strategy.
        return '{} ({} -> {} <- {}{})'.format(self.__class__.__name__, self.in_, self.out_, 'CRS, k=', self.k)


class LinearShawn(nn.Module):
    '''
    A linear module (layer without activation) with meprop
    The initialization of w and b is the same with the default linear module.
    '''

    def __init__(self, in_, out_, k):
        super(LinearShawn, self).__init__()
        self.in_ = in_
        self.out_ = out_
        self.k = k

        # TODO Modernize this code, change Paramters to simple Tensors.
        self.w = Parameter(torch.Tensor(self.out_, self.in_))
        self.b = Parameter(torch.Tensor(self.out_))
        assert self.w.requires_grad
        assert self.b.requires_grad

        self.reset_parameters()

    def reset_parameters(self):
        # TODO: why this initialization? Why not a non-uniform init?
        stdv = 1. / math.sqrt(self.out_)
        self.w.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # Note: Need to modify this to work with 3-tensor inputs -- see LinearCRS.forward()
        return linearUnified_shawn(self.k)(x, self.w, self.b)

    def __repr__(self):
        return '{} ({} -> {} <- {}{})'.format(self.__class__.__name__,
                                              self.in_, self.out_, 'shawnunified', self.k)
