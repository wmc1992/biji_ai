# 线性回归

## 1、问题描述

给定数据集 $D=\{(\overrightarrow{x^{(1)}}, y^{(1)}), (\overrightarrow{x^{(2)}}, y^{(2)}), ..., (\overrightarrow{x^{(n)}}, y^{(n)})\}$

其中：

$\overrightarrow{x^{(i)}}=(x_1^{(i)}, x_2^{(i)}, x_3^{(i)}, ..., x_d^{(i)})^T \in R^d$ 表示第 $i$ 条数据的特征；

$d$ 表示每条数据特征的维度；

$n$ 表示总的数据量；

$y^{(i)} \in R$ 表示第 $i$ 条数据的标签；

**符号说明**：右上角的角标代表第几条数据，右下角的角标代表第几个特征，比如 $x_j^{(i)}$ 表示第 $i$ 条数据的第 $j$ 个特征的值；

## 2、建立模型

$$
\begin{equation}
\begin{split}   
h_ \theta (\overrightarrow{x^{(i)}}) &= \theta_0 + \theta_1 x_1^{(i)} + \theta_2 x_2^{(i)} + ... + \theta_d x_d^{(i)} \\
&= \sum_{j=0}^d \theta_j x_j^{(i)}
\end{split}
\end{equation}
$$

上式中 $\theta_0$ 是bias，为了便于计算，假设 $x_0^{(i)}=1$，则可以直接将 $\theta_0$ 合并到 $\sum_{j=0}^d \theta_j x_j^{(i)}$ 中。

一般的，模型的输出会记为 $\hat{y}^{(i)}$，则有：

$$\hat{y}^{(i)}=h_ \theta (\overrightarrow{x^{(i)}})$$

## 3、代价函数

使用最小二乘（least mean squares）作为代价函数，如下：

$$J(\theta) = \frac{1}{2} \sum_{i=1}^n (h_ \theta (\overrightarrow{x^{(i)}}) - y_{(i)})^2$$

## 4、求解梯度

$$
\begin{equation}
\begin{split}   
\frac{\partial}{\partial \theta_j} J(\theta) &= \frac{\partial}{\partial \theta_j} \frac{1}{2} \sum_{i=1}^n (h_ \theta (\overrightarrow{x^{(i)}}) - y_{(i)})^2 \\
&= \frac{1}{2} \cdot 2 \cdot \sum_{i=1}^n (h_ \theta (\overrightarrow{x^{(i)}}) - y_{(i)}) \cdot \frac{\partial}{\partial \theta_j} (h_ \theta (\overrightarrow{x^{(i)}}) - y_{(i)}) \\
&= \sum_{i=1}^n (h_ \theta (\overrightarrow{x^{(i)}}) - y_{(i)}) x_j
\end{split}
\end{equation}
$$

## 5、梯度下降

对于权重 $\theta_j$ 每次迭代更新的公式为：

$$
\begin{equation}
\begin{split}   
\theta_j &:= \theta_j - \gamma \frac{\partial}{\partial \theta_j} J(\theta) \\
&:= \theta_j - \gamma \sum_{i=1}^n (h_ \theta (\overrightarrow{x^{(i)}}) - y_{(i)}) x_j
\end{split}
\end{equation}
$$

其中 $\gamma$ 为学习率；

## 6、随机梯度下降

在梯度下降中，每次对 $\theta_j$ 的更新都是求解出 $n$ 条数据在 $\theta_j$ 上的所有梯度之后再进行更新。如果每条数据求解出梯度之后都对 $\theta_j$ 更新一次，则为随机梯度下降，公式如下：

$$\theta_j := \theta_j - \gamma (h_ \theta (\overrightarrow{x^{(i)}}) - y_{(i)}) x_j$$
