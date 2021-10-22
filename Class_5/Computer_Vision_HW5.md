# 导数表达式

> 计算机视觉第五次作业  |  2101212840   游盈萱

[TOC]

## 问题描述

$$
\begin{aligned}
&h=XW_1+b_1\\ \\
&h_{sigmoid}=sigmoid(h) \\ \\
&Y{pred}=h_{sigmoid}W_2+b_2\\ \\
&f=||Y-Y_{pred}||^2_F\\ \\
\end{aligned}
$$

给出变量$W_1, b_1, W_2, b_2$导数表达式

## 推导过程

令$Z=XW$，$h=max(Z,0)$，

于是，
$$
\begin{aligned}
f &=||Y-Y_{pred}||^2_F \\ \\
  &=tr[(Y-Y_{pred})^T(Y-Y_{pred})] 
\end{aligned}
$$
方程两边取微分得到，
$$
\begin{aligned}
df&=d\{tr[(Y-Y_{pred})^T(Y-Y_{pred})]\} \\ \\
  &=tr\{d[(Y-Y_{pred})^T(Y-Y_{pred})]\} \\ \\
\end{aligned}
$$

### 推导$$\frac {\partial f} {\partial Y}$$，

$$
\begin{aligned}
df&=tr\{d[(Y-Y_{pred})^T(Y-Y_{pred})]\} \\ \\
  &=tr[(dY)^T(Y-Y_{pred})+(Y-Y_{pred})^TdY]\} \\ \\
  &=tr[2(Y-Y_{pred})^TdY]
\end{aligned}
$$



得到$$\frac {\partial f} {\partial Y}$$，
$$
=>\ \frac {\partial f}{\partial Y} = 2(Y-Y_{pred})
$$
### 推导$$\frac {\partial f} {\partial Y_{pred}}$$，

$$
\begin{aligned}
df&=tr\{d[(Y-Y_{pred})^T(Y-Y_{pred})]\} \\ \\
  &=tr[(-dY_{pred})^T(Y-Y_{pred})+(Y-Y_{pred})^T(-dY_{pred})]\} \\ \\
  &=tr[2(Y_{pred}-Y)^TdY_{pred}]
\end{aligned}
$$

得到$$\frac {\partial f} {\partial Y_{pred}}$$，
$$
=>\ \frac {\partial f}{\partial Y_{pred}}=2(Y_{pred}-Y)
$$
### 推导$$\frac {\partial f} {\partial W_2}$$
$$
\begin{aligned}
df&=tr[2(Y_{pred}-Y)^TdY_{pred}] \\ \\
  &=tr[2(Y_{pred}-Y)^Td(h_{sigmoid}W_2+b_2)] \\ \\
  &=tr[2(Y_{pred}-Y)^Th_{sigmoid}d(W_2)]
\end{aligned}
$$

得到$$\frac {\partial f} {\partial W_2}$$，
$$
=>\ \frac {\partial f}{\partial W_2} = 2h_{sigmoid}^T[(Y_{pred}-Y)]
$$
### 推导$\frac {\partial f} {\partial b_2}$
$$
\begin{aligned}
df&=tr[2(Y_{pred}-Y)^TdY_{pred}] \\ \\
  &=tr[2(Y_{pred}-Y)^Td(h_{sigmoid}W_2+b_2)] \\ \\
  &=tr[2(Y_{pred}-Y)^Td(b_2)]
\end{aligned}
$$

得到$$\frac {\partial f} {\partial b_2}$$，
$$
=>\ \frac {\partial f}{\partial b_2} = 2(Y_{pred}-Y)
$$
### 推导$$\frac {\partial f} {\partial h_{sigmoid}}$$

$$
\begin{aligned}
df&=tr[2(Y_{pred}-Y)^TdY_{pred}] \\ \\
  &=tr[2(Y_{pred}-Y)^Td(h_{sigmoid}W_2+b_2)] \\ \\
  &=tr[2(Y_{pred}-Y)^Td(h_{sigmoid})W_2]\\ \\
  &=tr[2W_2(Y_{pred}-Y)^Td(h_{sigmoid})]
\end{aligned}
$$

得到$$\frac {\partial f} {\partial h_{sigmoid}}$$，
$$
=>\ \frac {\partial f}{\partial h_{sigmoid}} = 2(Y_{pred}-Y)W_2^T
$$

### 推导$$\frac {\partial f} {\partial h}$$

$$
\begin{aligned}
\frac {\partial f} {\partial h} &= \frac {\partial f} {\partial h_{sigmoid}} \cdot \frac {\partial h_{sigmoid}} {\partial h} \\ \\
 &=2(Y_{pred}-Y)W_2^T \odot sigmoid'(h)
\end{aligned}
$$

### 推导$$\frac {\partial f} {\partial W_1}$$

由$\frac {\partial f} {\partial h}=2(Y_{pred}-Y)W_2^T \odot sigmoid'(h)$
$$
\begin{aligned}
df&=tr[2[(Y_{pred}-Y)W_2^T \odot sigmoid'(h)]^Tdh] \\ \\
  &=tr[2[(Y_{pred}-Y)W_2^T \odot sigmoid'(h)]^Td(XW_1+b_1)] \\ \\
  &=tr[2[(Y_{pred}-Y)W_2^T \odot sigmoid'(h)]^TXd(W_1)]
\end{aligned}
$$

得到$$\frac {\partial f} {\partial W_1}$$，
$$
=>\ \frac {\partial f}{\partial W_1} = 2X^T(Y_{pred}-Y)W_2^T \odot sigmoid'(h)
$$

### 推导$$\frac {\partial f} {\partial b_1}$$

由$\frac {\partial f} {\partial h}=2(Y_{pred}-Y)W_2^T \odot sigmoid'(h)$
$$
\begin{aligned}
df&=tr[2[(Y_{pred}-Y)W_2^T \odot sigmoid'(h)]^Tdh] \\ \\
  &=tr[2[(Y_{pred}-Y)W_2^T \odot sigmoid'(h)]^Td(XW_1+b_1)] \\ \\
  &=tr[2[(Y_{pred}-Y)W_2^T \odot sigmoid'(h)]^Td(b_1)]
\end{aligned}
$$

得到$$\frac {\partial f} {\partial b_1}$$，
$$
=>\ \frac {\partial f}{\partial b_1} = 2(Y_{pred}-Y)W_2^T \odot sigmoid'(h)
$$

### 综上所述

$$
\begin{aligned}
 &\frac {\partial f}{\partial W_1} = 2X^T(Y_{pred}-Y)W_2^T \odot sigmoid'(h)\\ \\
 &\frac {\partial f}{\partial b_1} = 2(Y_{pred}-Y)W_2^T \odot sigmoid'(h)\\ \\
 &\frac {\partial f}{\partial W_2} = 2h_{sigmoid}^T[(Y_{pred}-Y)]\\ \\
 &\frac {\partial f}{\partial b_2} = 2(Y_{pred}-Y)\\ \\
\end{aligned}
$$

