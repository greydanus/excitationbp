# excitation_bp: visualizing how deep networks make decisions
# Sam Greydanus. July 2017. MIT License.

from __future__ import absolute_import

from . import functions

from .functions.eb_linear import *
from .functions.eb_convNd import *
from .functions.eb_AvgPoolNd import *

from .utils import *
import copy

__version__ = '0.1'

real_fs = []
real_fs.append(copy.deepcopy(torch.nn.functional.linear))

real_fs.append(copy.deepcopy(torch.nn.functional.conv1d))
real_fs.append(copy.deepcopy(torch.nn.functional.conv2d))
real_fs.append(copy.deepcopy(torch.nn.functional.conv3d))

real_fs.append(copy.deepcopy(torch.nn.functional.avg_pool2d))

def use_eb(use_eb, verbose=True):
    global real_torch_funcs
    if use_eb:

        # this is a super-hacky way to indicate whether to use pos or neg weights
        torch.use_pos_weights = True

        if verbose: print("using excitation backprop autograd mode:")

        if verbose: print("\t->replacing torch.nn.functional.linear with eb_linear...")
        torch.nn.functional.linear = EBLinear.apply

        if verbose: print("\t->replacing torch.nn.functional.conv{1,2,3}d with eb_conv{1,2,3}d...")
        torch.nn.functional.conv1d = eb_conv1d
        torch.nn.functional.conv2d = eb_conv2d
        torch.nn.functional.conv3d = eb_conv3d

        if verbose: print("\t->replacing torch.nn.functional.avg_pool2d with eb_avg_pool2d...")
        torch.nn.functional.avg_pool2d = eb_avg_pool2d

    else:
        if verbose: print("using regular backprop autograd mode:")

        if verbose: print("\t->restoring torch.nn.backends.thnn.backend.Linear...")
        torch.nn.functional.linear = real_fs[0]

        if verbose: print("\t->restoring torch.nn.functional.conv{1,2,3}d...")
        torch.nn.functional.conv1d = real_fs[1]
        torch.nn.functional.conv2d = real_fs[2]
        torch.nn.functional.conv3d = real_fs[3]

        if verbose: print("\t->restoring torch.nn.functional.avg_pool2d...")
        torch.nn.functional.avg_pool2d = real_fs[4]