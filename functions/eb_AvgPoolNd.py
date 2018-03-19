# excitation_bp: visualizing how deep networks make decisions
# Sam Greydanus. July 2017. MIT License.

import torch
from torch.autograd import Function
from torch._thnn import type2backend
from torch.nn._functions.thnn.auto import function_by_name
import torch.backends.cudnn as cudnn

from torch.nn import _functions
from torch.nn.modules import utils
from torch.nn import ConstantPad2d
from torch.nn.modules.utils import _single, _pair, _triple

_thnn_convs = {}


class EBAvgPool2d(Function):

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)
        self.padding = _pair(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        # print("forward EBAvgPool2d")
        backend = type2backend[type(input)]
        output = input.new()
        # can avoid this with cudnn
        self.save_for_backward(input)
        backend.SpatialAveragePooling_updateOutput(
            backend.library_state,
            input, output,
            self.kernel_size[1], self.kernel_size[0],
            self.stride[1], self.stride[0],
            self.padding[1], self.padding[0],
            self.ceil_mode, self.count_include_pad)
        return output

    def backward(self, grad_output):
        raise NotImplementedError("EB for Average Pooling has not been implemented yet. \
            See excitation_bp/functions/README.md for instructions on how to implement \
            your own EB autograd module. Contact greydanus.17@gmail.com with questions.")
        backend = type2backend[type(grad_output)]
        input, = self.saved_tensors
        grad_input = grad_output.new()
        backend.SpatialAveragePooling_updateGradInput(
            backend.library_state,
            input, grad_output, grad_input,
            self.kernel_size[1], self.kernel_size[0],
            self.stride[1], self.stride[0],
            self.padding[1], self.padding[0],
            self.ceil_mode, self.count_include_pad)
        return grad_input


def eb_avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    return EBAvgPool2d(kernel_size, stride, padding, ceil_mode, count_include_pad)(input)
