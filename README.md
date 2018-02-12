UMBC-Verisk
=============

Installation
------------

Installation is simple. You need to install:
- [Torch7](http://torch.ch/docs/getting-started.html#_)
- [cunn](https://github.com/torch/cunn) for training on GPU
- [cudnn](https://github.com/soumith/cudnn.torch) for faster training on GPU
- [tds](https://github.com/torch/tds) for some data structures
- [display](https://github.com/szym/display) for graphs 

You can install all of these with the commands:
```bash
# install torch first
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh

# install libraries
luarocks install cunn
luarocks install cudnn
luarocks install tds
luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec
```

### Learning Resources
- [Torch Cheat Sheet](https://github.com/torch/torch7/wiki/Cheatsheet)
- [60 minute blitz](https://github.com/soumith/cvpr2015/blob/master/Deep%20Learning%20with%20Torch.ipynb)

Model
-----
3D conv-nets [C3D](http://vlg.cs.dartmouth.edu/c3d/)

Data Setup 
----------
THUMOS'15 dataset can be downloaded from [THUMOS](http://www.thumos.info/download.html)


Training
--------

To start training, just do:

```bash
$ CUDA_VISIBLE_DEVICES=0 th main.lua
```
where you replace the number after `CUDA_VISIBLE_DEVICES` with the GPU you want to run on. 
You can find which GPU to use with `$ nvidia-smi` on our GPU cluster. Note: this number is 0-indexed, unlike the rest of Torch!

During training, it will dump snapshots to the `checkpoints/` directory every epoch. Each time you start a new experiment, you should change the `name` (in `opt`), to avoid overwriting previous experiments.

Evaluation
----------
To evaluate your model, you can use the `eval.lua` script. It mostly follows the same format as `main.lua`. It reads your validation/testing dataset from a file similar to before, and sequentially runs through it, calculating both the top-1 and top-5 accuracy. 

Graphics, Logs
--------------
If you want to see graphics and the loss over time, in a different shell on the same machine, run this command:
```bash
$ th -ldisplay.start 8000 0.0.0.0
```
then navigate to ```http://HOST:8000``` in your browser. Every 10th iteration it will push graphs.
