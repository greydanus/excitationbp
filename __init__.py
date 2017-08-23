# excitation_bp: visualizing how deep networks make decisions
# Sam Greydanus. July 2017. MIT License.

from .functions.eb_linear import *
from .functions.eb_convNd import *
from .functions.eb_AvgPoolNd import *
from .functions.eb_pointwise import *

from .utils import *
import copy

real_fs = []
real_fs.append(copy.deepcopy(torch.nn.functional.linear))

real_fs.append(copy.deepcopy(torch.nn.functional.conv1d))
real_fs.append(copy.deepcopy(torch.nn.functional.conv2d))
real_fs.append(copy.deepcopy(torch.nn.functional.conv3d))

real_fs.append(copy.deepcopy(torch.nn.functional.avg_pool2d))

real_fs.append(copy.deepcopy(torch.tanh))
real_fs.append(copy.deepcopy(torch.nn.functional.tanh))

def use_eb(use_eb):
    global real_torch_funcs
    if use_eb:
        print("using excitation backprop autograd mode:")

        # print("\t->replacing torch.nn.backends.thnn.backend.Linear with EBLinear...")
        # torch.nn.backends.thnn.backend.Linear = EBLinear
        print("\t->replacing torch.nn.functional.linear with eb_linear...")
        torch.nn.functional.linear = EBLinear.apply

        print("\t->replacing torch.nn.functional.conv{1,2,3}d with eb_conv{1,2,3}d...")
        torch.nn.functional.conv1d = eb_conv1d
        torch.nn.functional.conv2d = eb_conv2d
        torch.nn.functional.conv3d = eb_conv3d

        print("\t->replacing torch.nn.functional.avg_pool2d with eb_avg_pool2d...")
        torch.nn.functional.avg_pool2d = eb_avg_pool2d

        print("\t->replacing torch.tanh & torch.nn.functional.tanh with eb_tanh...")
        torch.tanh = eb_tanh
        torch.nn.functional.tanh = eb_tanh

    else:
        print("using regular backprop autograd mode:")

        print("\t->restoring torch.nn.backends.thnn.backend.Linear...")
        # torch.nn.backends.thnn.backend.Linear = real_fs[0]
        raise NotImplementedError("need to fix indexing")
        torch.nn.functional.linear = real_fs[0]

        print("\t->restoring torch.nn.functional.conv{1,2,3}d...")
        torch.nn.functional.conv1d = real_fs[1]
        torch.nn.functional.conv2d = real_fs[2]
        torch.nn.functional.conv3d = real_fs[3]

        print("\t->restoring torch.nn.functional.avg_pool2d...")
        torch.nn.functional.avg_pool2d = real_fs[4]

        print("\t->restoring torch.tanh & torch.nn.functional.tanh...")
        torch.tanh = real_fs[5]
        torch.nn.functional.tanh = real_fs[6]