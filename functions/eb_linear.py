# excitation_bp: visualizing how deep networks make decisions
# Sam Greydanus. July 2017. MIT License.

import torch
from torch.autograd import Function
from torch.autograd import Variable

class EBLinear(Function):

    def forward(self, input, weight, bias=None):
        # print("forward EBLinear")
        self.save_for_backward(input, weight, bias)

        output = input.new(input.size(0), weight.size(0))
        output.addmm_(0, 1, input, weight.t())
        if bias is not None:
            # cuBLAS doesn't support 0 strides in sger, so we can't use expand
            self.add_buffer = input.new(input.size(0)).fill_(1)
            output.addr_(self.add_buffer, bias)
        return output

    def backward(self, grad_output):
        # print("backward EBLinear")
        input, weight, bias = self.saved_tensors

    ### *EB MODE* start excitation backprop part  ###

        s = 1 if torch.sum(grad_output) > 0 else -1
        weight_ = (weight.clone()*s).clamp(min=0)
        input_ = input.clone() ; input_ -= input_.min()
        # if torch.sum(input_) != torch.sum(input):
        #     print('\tinput contains some negative values...shifting input so min=0')

        norm_factor = input_.new(input_.size(0), weight_.size(0))
        norm_factor.addmm_(0, 1, input_, weight_.t())
        # don't add bias as you would in a normal forward pass...

        grad_output /= torch.abs(norm_factor) + 1e-10

    ### *EB MODE* end of excitation backprop part ###

        grad_input = grad_weight = grad_bias = None

        # *EB MODE* remove the if statement that's usually here:
        grad_input = torch.mm(grad_output, weight_)

        if self.needs_input_grad[1]:
            grad_weight = torch.mm(grad_output.t(), input_)
        if bias is not None and self.needs_input_grad[2]:
            grad_bias = torch.mv(grad_output.t(), self.add_buffer)

    ### *EB MODE* the excitation backprop part    ###
        grad_input *= input_
    ### *EB MODE* end of excitation backprop part ###

        if bias is not None:
            return grad_input, grad_weight, grad_bias
        else:
            return grad_input, grad_weight