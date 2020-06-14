# Deep Closest Point in TensorFlow

## Introduction

This project implements [Deep Closest Point](https://arxiv.org/abs/1905.03304) model in TensorFlow. It also includes C++ code that compare its performance with other registration methods (ICP, 4-PCS, Go-ICP).

## Dependencies

To run DCP model, you may have to install these Python packages:

* tensorflow>=2.0.0
* tensorflow-graphics (none of its dependencies is required)
* numpy
* h5py

To run comparison program, you may have to install these libraries:

* PCL 1.9 (and its dependencies)
* HDF5
* TBB

## Usage

Basic usage is encapsulated into classes or functions. You can directly call them in the program. Hyperparameters are directly defined in source code, and command line arguments is not supported.

### Dataset

Download [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) and unzip files into directory `modelnet40`. Run `util.pack_to_one()` to pack all dataset files into single `train.h5` and `test.h5` files. 

### Training and evaluation

Trained weights `dcp_v2.h5` can be unzipped from [weights/dcp_v2.zip](weights/dcp_v2.zip). Place it in `weights` directory so that evaluation and testing procedure can find it. If you want to train by yourself, run `train.train()` to train, or your owning training procedure. Run `train.evaluate()` to evaluate the trained model with test dataset.

### Comparison

The comparison program tests registration methods on the first 100 models of the test dataset. It is divided into Python and C++ code. Run `compare.test_dcp()` to test DCP. Compile and run the C++ program to test ICP, 4-PCS and Go-ICP. ICP and 4-PCS implementation is from PCL. Go-ICP is from my previous project [OptICP](https://github.com/wzh99/OptICP).

## Documents

The project [proposal](doc/proposal.md) and [report](doc/dcp_report.md) are provided (both in Chinese). Refer to them for better understanding of this project. 
