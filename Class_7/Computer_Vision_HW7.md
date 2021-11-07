# 超分算法

> 计算机视觉第七次作业  |  2101212840   游盈萱

[TOC]

## 任务描述

Github或者主页下载运行一个超分算法，获得结果试着训练一两个Epoch，给出超分结果

## 超分算法 EDSR 学习

**Enhanced Deep Residual Networks for Single Image Super-Resolution** from CVPRW 2017

EDSR是NTIRE2017超分辨率挑战赛上获得冠军的方案。EDSR的网络结构如下：

![2](./2.png)

EDSR借鉴了ResNet网络基于残差的学习机制，输入经过一层卷积后兵分两路，一条经过n层的ResBlock再卷积一次，另一条直接通到交汇处，进行加权求和，再进行上采样和卷积输出结果。

## 实验

### 训练1个epoch

**训练过程**：

<img src="./1.png" alt="1" style="zoom: 83%;" />

**测试过程**：

![3](./3.png)

**超分结果**：

左 - 原图，右 - 超分结果

<div align='center' >
<img src="./0901x2.png" alt="4-1" style="zoom: 40%;" /> <img src="0901x2_x2_SR_1.png" alt="4-2" style="zoom: 20%;" />
<img src="./5.png" alt="4-1" style="zoom: 30%;" /> <img src="./6.png" alt="4-1" style="zoom: 30%;" />
</div>


放大2倍之后可以看到超分结果中耳鬓的毛发稍微清晰一些。

### 训练300个epochs

**训练过程**：

![11](./11.png)

**测试过程**：

![12](./12.png)

**超分结果**：

左 - 原图，右 - 超分结果

<div align='center' >
<img src="./0901x2.png" alt="8-1" style="zoom: 40%;" /> <img src="0901x2_x2_SR_300.png" alt="8-2" style="zoom: 20%;" />
<img src="./13.png" alt="8-1" style="zoom: 30%;" /> <img src="./14.png" alt="8-1" style="zoom: 30%;" />
</div>

放大2倍之后可以看到超分结果中耳鬓的毛发比1个epoch的更清晰。

### 官方模型

**测试过程**：

![8](./8.png)

**超分结果**：

左 - 原图，右 - 超分结果

<div align='center' >
<img src="./0901x4.png" alt="8-1" style="zoom: 80%;" /> <img src="0901x4_x4_SR.png" alt="8-2" style="zoom: 20%;" />
<img src="./10.png" alt="8-1" style="zoom: 30%;" /> <img src="./9.png" alt="8-1" style="zoom: 30%;" />
</div>

放大4倍之后可以看到超分结果中耳鬓的毛发也很清晰。

