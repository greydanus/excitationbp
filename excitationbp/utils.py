# excitation_bp: visualizing how deep networks make decisions
# Sam Greydanus. July 2017. MIT License.

import torch
import numpy as np
from torch.autograd import Function, Variable

def trainable_modules(orig, flat=None, param_only=True):
    flat = [] if flat is None else flat
    submodules = list(orig.children())
    if len(submodules) > 0:
        for m in submodules:
            flat += trainable_modules(m, flat=[])
    else:
        if param_only and len(list(orig.parameters())) > 0:
            return flat + [orig]
    return flat

def excitation_backprop(model, inputs, prob_outputs, contrastive=False, target_layer=0):
    inputs = Variable(inputs.data) # assure that 'inputs' is a leaf variable
    inputs.requires_grad = True # assure that gradients will end up on this leaf variable
    torch.use_pos_weights = True
    
    # get internal variables of the model
    layer_top = trainable_modules(model)[-1]
    layer_target = trainable_modules(model)[target_layer]
    
    global top_h_, contr_h_, target_h_
    top_h_, contr_h_, target_h_ = None, None, None

    def hook_top_h(m, i, o): global top_h_ ; top_h_ = o
    def hook_contr_h(m, i, o): global contr_h_ ; contr_h_ = i[0]
    def hook_target_h(m, i, o): global target_h_ ; target_h_ = i[0]

    h1 = layer_top.register_forward_hook(hook_top_h)
    h2 = layer_top.register_forward_hook(hook_contr_h)
    h3 = layer_target.register_forward_hook(hook_target_h)
    
    _ = model(inputs)
    h1.remove() ; h2.remove() ; h3.remove()
    if target_layer == 0:
        target_h_ = inputs # sometimes the user will modify the input before first module

    # do regular eb
    if not contrastive:
        outputs = model(inputs)
        return torch.autograd.grad(top_h_, target_h_, grad_outputs=prob_outputs)[0]
    
    pos_evidence = torch.autograd.grad(top_h_, contr_h_, grad_outputs=prob_outputs.clone())[0]
    torch.use_pos_weights = False
    neg_evidence = torch.autograd.grad(top_h_, contr_h_, grad_outputs=prob_outputs.clone())[0]
    
    torch.use_pos_weights = True
    contrastive_signal = pos_evidence - neg_evidence
    return torch.autograd.grad(contr_h_, target_h_, grad_outputs=contrastive_signal)[0]
