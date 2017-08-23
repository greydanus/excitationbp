# excitation_bp: visualizing how deep networks make decisions
# Sam Greydanus. July 2017. MIT License.

import torch
from torch.autograd import Function
from torch.autograd import Variable

# Inherit from Function
class EBLinear(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables

    ### *EB MODE* start excitation backprop part  ###
        s = 1 if torch.sum(grad_output.data) > 0 else -1
        weight_ = (weight.clone()*s).clamp(min=0)
        input_ = input.clone() ; input_ -= input_.min()

        grad_output /= torch.abs(input.mm(weight_.t())) + 1e-10

    ### *EB MODE* end of excitation backprop part ###
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight_)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input_)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

    ### *EB MODE* the excitation backprop part    ###
        grad_input = grad_input*input_ #; print('\t', torch.sum(grad_input.data))
    ### *EB MODE* end of excitation backprop part ###

        return grad_input, grad_weight, grad_bias