# Softmax 函数求导

写在前面，其他的大部分笔记都没有区分向量与标量，不过该篇内容中有比较多的向量与标量的区分。所以使用粗体字母表示向量，正常粗细的字母表示标量。

## 1、问题描述

softmax的公式为：

$$\begin{equation}\mathbf{\hat{y}} = \text{softmax}(\mathbf{z})\end{equation}$$

上述公式中：

* $\mathbf{z}$ 表示softmax的输入，是一个向量，维度为$d$，即 $\mathbf{z}=[z_1, z_2, \cdots, z_d]$；
* $\mathbf{\hat{y}}$ 表示softmax的输出，是一个向量，维度为$d$，即 $\mathbf{\hat{y}}=[\hat{y}_1, \hat{y}_2, \cdots, \hat{y}_d]$；

将softmax的具体公式代入到公式（1）中，则有：

$$\begin{equation}
\begin{split}
\mathbf{\hat{y}} &= \text{softmax}(\mathbf{z}) \\
&= \text{softmax}([z_1, z_2, \cdots, z_d]) \\
&= \Big[ \frac{e^{z_1}}{\sum_{i=1}^d e^{z_1}}, \frac{e^{z_2}}{\sum_{i=1}^d e^{z_1}}, \cdots, \frac{e^{z_d}}{\sum_{i=1}^d e^{z_1}} \Big] \\
&= [\hat{y}_1, \hat{y}_2, \cdots, \hat{y}_d]
\end{split}
\end{equation}$$

对softmax求导就是要求解下式：

$$\begin{equation}\frac{\partial \mathbf{\hat{y}}}{\partial \mathbf{z}}\end{equation}$$

## 2、对softmax求导

由于是向量对向量求导，所以其最终结果为Jacobi矩阵，如下：

$$\begin{equation}
\frac{\partial \mathbf{\hat{y}}}{\partial \mathbf{z}}=\begin{bmatrix}
   \frac{\partial \hat{y}_1}{\partial z_1} & \frac{\partial \hat{y}_1}{\partial z_2} & \cdots & \frac{\partial \hat{y}_1}{\partial z_n} \\
   \frac{\partial \hat{y}_2}{\partial z_1} & \frac{\partial \hat{y}_2}{\partial z_2} & \cdots & \frac{\partial \hat{y}_2}{\partial z_n} \\
   \vdots & \vdots & \cdots & \vdots \\
   \frac{\partial \hat{y}_m}{\partial z_1} & \frac{\partial \hat{y}_m}{\partial z_2} & \cdots & \frac{\partial \hat{y}_m}{\partial z_n} \\
\end{bmatrix}
\end{equation}$$

该矩阵中每一行的求导方式是相同的，我们仅求导第 $j$ 行。

将上述Jacobi矩阵的第$j$行摘取出来，并进行变形整理得：

$$\begin{equation}
\begin{split}
\frac{\partial \hat{y}_j}{\partial \mathbf{z}}
&=[ \frac{\partial \hat{y}_j}{\partial z_1}, \frac{\partial \hat{y}_j}{\partial z_2}, \cdots ,\frac{\partial \hat{y}_j}{\partial z_j}, \cdots,\frac{\partial \hat{y}_j}{\partial z_n}] \\
&=[ \frac{\partial}{\partial z_1}\big(\frac{e^{z_j}}{\sum_{i=1}^d e^{z_i}} \big), \frac{\partial}{\partial z_2}\big(\frac{e^{z_j}}{\sum_{i=1}^d e^{z_i}} \big), \cdots ,\frac{\partial}{\partial z_j}\big(\frac{e^{z_j}}{\sum_{i=1}^d e^{z_i}} \big), \cdots,\frac{\partial}{\partial z_n}\big(\frac{e^{z_j}}{\sum_{i=1}^d e^{z_i}} \big)]
\end{split}
\end{equation}$$

$\frac{\partial \hat{y}_j}{\partial \mathbf{z}}$是一个向量，该向量中有 $d$ 个元素，下面逐个元素进行求解（下述公式中的(6)、(7)、(9)三个式子推导过程完全相同，只看一个即可；公式(8)的推导过程与另外三个式子是不同的）：

$$\begin{equation}
\begin{split}
\frac{\partial \hat{y}_j}{\partial z_1}
&=\frac{\partial}{\partial z_1}\big(\frac{e^{z_j}}{\sum_{i=1}^d e^{z_i}} \big)=\frac{\frac{\partial}{\partial z_1}(e^{z_j}) \cdot \sum_{i=1}^d e^{z_i} - e^{z_j} \cdot \frac{\partial}{\partial z_1}(\sum_{i=1}^d e^{z_i}) }{(\sum_{i=1}^d e^{z_i})^2} \\
&= \frac{0 - e^{z_j} e^{z_1}}{(\sum_{i=1}^d e^{z_i})^2}= - \frac{e^{z_j}}{\sum_{i=1}^d e^{z_i}} \frac{e^{z_1}}{\sum_{i=1}^d e^{z_i}}= - \hat{y}_j \hat{y}_1
\end{split}
\end{equation}$$

$$\begin{equation}
\begin{split}
\frac{\partial \hat{y}_j}{\partial z_2}
&=\frac{\partial}{\partial z_2}\big(\frac{e^{z_j}}{\sum_{i=1}^d e^{z_i}} \big)=\frac{\frac{\partial}{\partial z_2}(e^{z_j}) \cdot \sum_{i=1}^d e^{z_i} - e^{z_j} \cdot \frac{\partial}{\partial z_2}(\sum_{i=1}^d e^{z_i}) }{(\sum_{i=1}^d e^{z_i})^2} \\
&= \frac{0 - e^{z_j} e^{z_2}}{(\sum_{i=1}^d e^{z_i})^2}= - \frac{e^{z_j}}{\sum_{i=1}^d e^{z_i}} \frac{e^{z_2}}{\sum_{i=1}^d e^{z_i}}= - \hat{y}_j \hat{y}_2
\end{split}
\end{equation}$$

$$\vdots$$

$$\begin{equation}
\begin{split}
\frac{\partial \hat{y}_j}{\partial z_j}
&=\frac{\partial}{\partial z_j}\big(\frac{e^{z_j}}{\sum_{i=1}^d e^{z_i}} \big)=\frac{\frac{\partial}{\partial z_j}(e^{z_j}) \cdot \sum_{i=1}^d e^{z_i} - e^{z_j} \cdot \frac{\partial}{\partial z_j}(\sum_{i=1}^d e^{z_i}) }{(\sum_{i=1}^d e^{z_i})^2} \\
&= \frac{e^{z_j} \cdot \sum_{i=1}^d e^{z_i} - e^{z_j} e^{z_j}}{(\sum_{i=1}^d e^{z_i})^2}=\frac{e^{z_j}}{\sum_{i=1}^d e^{z_i}} - (\frac{e^{z_j}}{\sum_{i=1}^d e^{z_i}})^2 = \hat{y}_j - (\hat{y}_j)^2
\end{split}
\end{equation}$$

$$\vdots$$

$$\begin{equation}
\begin{split}
\frac{\partial \hat{y}_j}{\partial z_d}
&=\frac{\partial}{\partial z_d}\big(\frac{e^{z_j}}{\sum_{i=1}^d e^{z_i}} \big)=\frac{\frac{\partial}{\partial z_d}(e^{z_j}) \cdot \sum_{i=1}^d e^{z_i} - e^{z_j} \cdot \frac{\partial}{\partial z_d}(\sum_{i=1}^d e^{z_i}) }{(\sum_{i=1}^d e^{z_i})^2} \\
&= \frac{0 - e^{z_j} e^{z_d}}{(\sum_{i=1}^d e^{z_i})^2}= - \frac{e^{z_j}}{\sum_{i=1}^d e^{z_i}} \frac{e^{z_d}}{\sum_{i=1}^d e^{z_i}}= - \hat{y}_j \hat{y}_d
\end{split}
\end{equation}$$

这样就求解出了上述Jacobi矩阵的第$j$行，如下式所示：

$$\begin{equation}
\begin{split}
\frac{\partial \hat{y}_j}{\partial \mathbf{z}}
&=[ \frac{\partial \hat{y}_j}{\partial z_1}, \frac{\partial \hat{y}_j}{\partial z_2}, \cdots ,\frac{\partial \hat{y}_j}{\partial z_j}, \cdots,\frac{\partial \hat{y}_j}{\partial z_n}] \\
&=[- \hat{y}_j \hat{y}_1, - \hat{y}_j \hat{y}_2, \cdots, \hat{y}_j - (\hat{y}_j)^2, \cdots, - \hat{y}_j \hat{y}_d]
\end{split}
\end{equation}$$

从该结果中可以看出，仅有第 $j$ 个元素是比较特殊的，其他的 $d-1$ 个元素的求导过程是相同的；这个结论不只适用于上述Jacobi矩阵的第 $j$ 行，对整个Jacobi矩阵来说：主对角线上的元素求导过程是相同的，非主对角线上的元素求解过程是相同的；

接下来可直接写出最终的Jacobi矩阵了：

$$\begin{equation}
\frac{\partial \mathbf{\hat{y}}}{\partial \mathbf{z}}=\begin{bmatrix}
   \hat{y}_1-(\hat{y}_1)^2 & -\hat{y}_1 \hat{y}_2 & \cdots & -\hat{y}_1 \hat{y}_d \\
   -\hat{y}_2 \hat{y}_1 & \hat{y}_2-(\hat{y}_2)^2 & \cdots & -\hat{y}_2 \hat{y}_d \\
   \vdots & \vdots & \cdots & \vdots \\
   -\hat{y}_d \hat{y}_1 & -\hat{y}_d \hat{y}_2 & \cdots & \hat{y}_d-(\hat{y}_d)^2 \\
\end{bmatrix}
\end{equation}$$

至此，对softmax函数的求导全部完成。

## 3、总结

本文主要是对神经网络中的常用函数softmax进行求导。

## Reference

* [https://zhuanlan.zhihu.com/p/105758059](https://zhuanlan.zhihu.com/p/105758059)

* [https://www.cnblogs.com/pinard/p/10750718.html](https://www.cnblogs.com/pinard/p/10750718.html)
