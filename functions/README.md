Extending PyTorch autograd functions
=======

Implemented
--------
These extensions have been implemented and tested. You can find them in the .py files of this directory.

* [linear](http://pytorch.org/docs/master/nn.html?highlight=linear#linear-layers)
* [1D, 2D, and 3D convolutions](http://pytorch.org/docs/master/nn.html?highlight=linear#conv2d)
* [tanh](http://pytorch.org/docs/master/nn.html?highlight=tanh#torch.nn.functional.tanh)

No changes necessary
--------
See page 8 of [original paper](https://arxiv.org/abs/1608.00507) for discussion.

* positive elementwise activations (e.g. **relu**, **softmax**, **exp**, etc.)
* [max pooling](http://pytorch.org/docs/master/nn.html?highlight=linear#maxpool2d)
* others? read paper.

Pending
--------
See page 8 of [original paper](https://arxiv.org/abs/1608.00507) for discussion. See below for instructions on how to make your own implementation.

* [average pooling](http://pytorch.org/docs/master/nn.html?highlight=linear#avgpool2d)
* [batch norm](http://pytorch.org/docs/master/nn.html?highlight=linear#batchnorm2d)
* [local response normalization (LRN)](https://github.com/pytorch/pytorch/issues/653)

How to write your own extensions
--------
Implementing a custom autograd module in PyTorch is not difficult. Here are some recommended steps:

1. Read this short [tutorial](http://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html#sphx-glr-beginner-examples-autograd-two-layer-net-custom-function-py) on extending PyTorch autograd functions.
2. Read and understand at least pages 8 and 9 of the [EB paper](https://arxiv.org/abs/1608.00507)
3. Copy the autograd function you want to extend to this file
	* for example, if we were extending `linear`, we would copy `torch.nn._functions.linear.py`
4. Rename the file and function/class with an `EB` prefix
	* for example, file `linear.py` becomes `eb_linear.py` and class `Linear` becomes `EBLinear`
5. Replace the real autograd function with your own
	* open the \_\_init\_\_.py file in the root directory of this repo
	* if you're extending a class, something like:
		* `torch.nn.backends.thnn.backend.Linear = EBLinear`
	* if you're extending a function, something like:
		* `torch.nn.functional.conv1d = eb_conv1d`
	* follow the structure of \_\_init\_\_.py so that the user can turn EB on and off
6. Share it!
	* email me at _greydanus dot 17 at gmail dot com_ or submit a pull request

