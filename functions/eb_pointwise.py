# excitation_bp: visualizing how deep networks make decisions
# Sam Greydanus. July 2017. MIT License.

import torch
from itertools import repeat
from torch.autograd.function import Function, InplaceFunction


class EBTanh(InplaceFunction):

    def forward(self, i):
        print("FORWARD pass on tanh...")
        if self.inplace:
            self.mark_dirty(i)
            result = i.tanh_()
        else:
            result = i.tanh()
        self.save_for_backward(result)
        return result

    def backward(self, grad_output):
        result, = self.saved_tensors
        grad_input = grad_output * (1 - result * result)

        print('grad_output:', torch.sum(grad_output<0))
        print('result:', torch.sum(result<0))
        print('grad_input:', torch.sum(grad_input<0))

        return grad_input


def eb_tanh(input):
    if type(input) != torch.autograd.variable.Variable:
        return torch.Tensor.tanh(input)
    return EBTanh()(input)
