# excitation_bp: visualizing how deep networks make decisions
# Sam Greydanus. July 2017. MIT License.

import torch
import numpy as np

def contrastive_eb(model, X, layer_top, layer_second, target=None, top_dh=None):
    
    # obtain internal variables
    global top_h_, target_dh_, second_h_, second_dh_
    top_h_ = target_dh_ = second_h_ = second_dh_ = None

    def hook_top_h(m, i, o): global top_h_ ; top_h_ = o.clone()
    def hook_target_dh(m, grad_i, grad_o): global target_dh_ ; target_dh_ = grad_o[0].clone()
    def hook_second_h(m, i, o): global second_h_ ; second_h_ = o.clone()
    def hook_second_dh(m, grad_i, grad_o): global second_dh_ ; second_dh_ = grad_o[0].clone()
        
    h1 = layer_top.register_forward_hook(hook_top_h)
    h2 = None if target is None else target.register_backward_hook(hook_target_dh)
    h3 = layer_second.register_forward_hook(hook_second_h)
    h4 = layer_second.register_backward_hook(hook_second_dh)
    
    X.requires_grad = True
    _ = model(X) # do a forward pass so the forward hooks can be called
    if top_dh is None: top_dh, ixs = explore_random_unit(top_h_)
        
    # backward pass 
    '''in the next release of pytorch there will be a torch.autograd.grad function
    which will allow us to backprop through just a subset of the graph. usage would
    look like "torch.autograd.grad(outputs, inputs, grad_outputs)". this will make 
    the runtime of contrastive backprop the same as a regular pass of backprop. the
    current implementation performs three backprops for each one pass of excitation
    backprop. i'll release an updated version after the next pytorch release.
    '''
    
    # backprop positive signal
    optimizer = torch.optim.SGD([X], lr=1)
    torch.autograd.backward(top_h_, top_dh, retain_variables=True) # backward hooks are called here
    pos = second_dh_.data.clone()
    
    # backprop negative signal
    optimizer.zero_grad()
    torch.autograd.backward(top_h_, -top_dh, retain_variables=True) # backward hooks are called here
    neg = second_dh_.data.clone()
    
    # backprop contrastive signal
    optimizer.zero_grad()
    torch.autograd.backward(second_h_, pos + neg, retain_variables=True) # backward hooks are called here
    
    h1.remove() ; h3.remove() ; h4.remove()
    if target is not None: h2.remove()
    
    return X.grad.clone().data.clamp(min=0) if target is None else target_dh_.clone().data.clamp(min=0)

def eb(model, X, layer_top, target=None, top_dh=None):
    
    # obtain internal variables
    global top_h_, target_dh_
    top_h_ = target_dh_ = None

    def hook_top_h(m, i, o): global top_h_ ; top_h_ = o.clone()
    def hook_target_dh(m, grad_i, grad_o): global target_dh_ ; target_dh_ = grad_o[0].clone()
        
    h1 = layer_top.register_forward_hook(hook_top_h)
    h2 = None if target is None else target.register_backward_hook(hook_target_dh)

    X.requires_grad = True
    _ = model(X) # do a forward pass so the forward hooks can be called
    if top_dh is None: top_dh, ixs = explore_random_unit(top_h_, verbose=True)
    
    # backprop positive signal
    optimizer = torch.optim.SGD([X], lr=1)
    torch.autograd.backward(top_h_, top_dh) # backward hooks are called here

    h1.remove()
    if target is not None: h2.remove()
    
    return X.grad.clone().data.clamp(min=0) if target is None else target_dh_.clone().data.clamp(min=0)

def explore_random_unit(input, verbose=False):
    if type(input) is not list:
        input = list(top_h.size()) # assume input is a Variable or a Tensor
    ixs = tuple(np.random.randint(d) for d in input)
    layer_dh = torch.zeros(input)
    layer_dh[ixs] = 1 # fake label for this neuron
    if verbose: print('exploring unit {} from layer of size {}'.format(ixs, input))
    return layer_dh, ixs