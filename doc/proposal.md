# TFG 大作业项目说明

###### 王梓涵　517021911179

## 简介

本项目使用深度神经网络来求解点云配准问题，使用 [TensorFlow Graphics](https://tensorflow.google.cn/graphics) 来为模型的训练和测试生成数据。

## 背景

在计算机图形学的三维点云重建中，受测量方式和被测物体形状的限制，一次只能测量有限的点云数据，需要在多个视角下进行多次扫描。由于每次扫描下的点云数据都具有独立的坐标系，所以需要获取这些视角下的坐标变换，将每次扫描的结果和原始点集进行对应。

点云配准问题已经被研究了数十年，研究人员对该问题提出了相当多的算法。传统算法能够一定程度上解决这些问题，但也各自存在明显的缺陷，如经典 ICP 算法极易陷入局部最小值，基于随机采样一致性（RANSAC）的算法稳定性较差，Go-ICP 能求出全局最优值但耗时极长等。近年来，随着深度学习的发展，一种新的思路是使用神经网络来求解该问题，良好设计的网络能同时做到配准误差小、速度快、稳定性高。

TensorFlow Graphics 基于 TensorFlow 的基本运算构建了适合计算机图形学任务的神经网络层。该框架提供了一系列教程，其中的“[六自由度对齐](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/notebooks/6dof_alignment.ipynb)”是配准的简化版本（顶点关系已知）。本项目对 TFG 的使用和该教程类似，即使用 TFG 生成训练和测试数据。

## 问题阐述

下面将给出点云配准在数学上的定义。假定 $\mathcal{X,Y}$ 为两片点云，本项目中假定它们大小均为 $N$，即 $\mathcal{X}=\{\mathbf{x}_i\in\R^3,i\in1..N\}$, $\mathcal{Y}=\{\mathbf{y}_i\in\R^3,i\in1..N\}$。 需要找到一个刚体变换 $(\mathbf{R}_{\mathcal{XY}},\mathbf{t}_\mathcal{XY})$ 使得下列目标函数最小化：
$$
E(\mathbf{R}_\mathcal{XY},\mathbf{t}_\mathcal{XY})=\frac{1}{N}\sum_{i=1}^N||\mathbf{R}_\mathcal{XY}\cdot\mathbf{x}_i+\mathbf{t}_\mathcal{XY}-\mathbf{y}_{m(\mathbf{x}_i)}||^2 \tag{1}
$$
上式中的 $m(\mathbf{x}_i)$ 表示 $\mathcal{X}$ 中的每个点 $\mathbf{x}_i$ 在 $\mathcal{Y}$ 中的对应点，其表达式为：
$$
m_\mathcal{Y}(\mathbf{x}_i)=\mathop{\arg\min}_j||\mathbf{R}_\mathcal{XY}\cdot\mathbf{x}_i+\mathbf{t}_\mathcal{XY}-\mathbf{y}_j||^2 \tag{2}
$$
式 (1) (2) 形成了“鸡生蛋蛋生鸡”的问题。为了求得使目标函数最小的刚体变换，需要知道点云中点的对应关系；而为了求出对应关系，又需要求出使目标函数最小的刚体变换。这也正是该问题的困难之处。

## 工作基础

* 在计算机图形学课程大作业中使用传统方法完成过点云配准，对点云配准的基本理论和方法较为熟悉。
* 已经找到了可以支持项目开展的论文 [Deep Closest Point](https://arxiv.org/abs/1905.03304) 及其 [PyTorch 实现](https://github.com/WangYueFt/dcp)。

## 任务目标

1. 使用 TFG 编写数据的生成程序，该程序应能随机生成给定范围内的合法的刚体变换，并且将该变换作用于任意点云。
2. 使用 TensorFlow 复现该论文的模型，和数据生成程序一起构成独立的系统（不依赖任何原作者的代码），验证该模型的性能。
3. 时间充裕的情况下，可以考虑将其他配准算法和该方法进行比较。