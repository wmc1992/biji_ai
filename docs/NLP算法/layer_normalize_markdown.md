# Layer Normalize

## 1、LN的具体操作步骤

其操作步骤可分为三部分：

* 求每条数据各特征之间的均值和标准差；
* 每条数据的每个特征减去各自数据的均值，除上各自数据的标准差；
* 对经过上一步骤的输出再经过一个线性变换；

以上是文字形式的说明，以下是公式形式。

**输入**：

一个 mini-batch 的数据在某层网络的输出为 $\{\alpha_1, \alpha_2, ..., \alpha_m\}$，其中 $m$ 为batch size；

记 $\alpha_i^{(j)}$ 为该mini-batch中第 $i$ 条数据的第 $j$ 个特征；

$T$ 为每条数据的特征数，每条数据的特征数不一定相同；

$g$ 和 $b$ 为可学习参数；

> 按照上述定义，某层网络的输出的shape为 $[m, T]$，$m$ 为batch-size，$T$ 为每条数据的特征数量；

**输出**：

$y_i^{(j)}$ 为该 mini-batch 中第 $i$ 条数据第 $j$ 个特征经过LN之后的输出；

**公式**：

第 $i$ 条数据各特征的均值：

$$\mu_i = \frac{1}{T} \sum_{j=1}^{T} \alpha_i^{(j)} $$

第 $i$ 条数据各特征的方差：

$$\sigma_i^2 = \frac{1}{T} \sum_{j=1}^{T} (\alpha_i^{(j)} - \mu_i)^2$$

减去均值，除上标准化，$\epsilon$ 用于避免除数为0：

$$\hat{\alpha_i^{(j)}} = \frac{\alpha_i^{(j)} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} $$

第 $i$ 条数据第 $j$ 个特征经过LN后的结果：

$$y_i^{(j)} = g \hat{\alpha_i^{(j)}} + b $$

## 2、LN 为了解决什么问题

1. 深度模型训练时所需要的计算资源非常大，想要减少训练所需时间的一个方法是：normalize the activities of neurons

2. 增加训练过程的稳定性；

## 3、LN 出现之前是如何解决上述问题的

LN 出现之前通过 BN 解决上述问题；

BN的优点：

* 可以解决 "convariate shift" 问题，缩短了模型训练所需的时间；
* 能够使饱和激活函数的输入落在非饱和区，增加了训练的稳定性；

BN的缺点：

* 当 batch size 特别小时，表现不好；
* 当每条数据的长度不一致时，比如文本数据，效果不好；
* 在 RNN 网络中，表现不好；

## 4、LN 的优势

Normalization 的作用：降低了对参数初始化的需求，允许使用更大的学习率，有一定的正则化作用可抗过拟合，使训练更加稳定。

假设某一层输出的中间结果为 $[m, T]$，$m$ 为batch-size，$T$ 为每条数据的特征数量，那么：

* BN 是对 $m$ 这个维度做归一化；
* LN 是对 $T$ 这个维度做归一化；

优势（以下都有待考证）：

* 在 RNN 网络中，表现较好；
* 在 batch size 较小的网络中，表现较好；
* LN 抹杀了不同样本间的大小关系，保留了同一个样本内部的特征之间的大小关系，这对于时间序列任务或NLP任务来说非常重要；

## 5、LN效果测试代码

```python
import torch
import torch.nn as nn

# NLP例子，一般在NLP任务中，其维度为[batch_size, seq_len, hidden_dim]，LayerNorm操作仅对最后一个维度做操作
batch, sentence_length, hidden_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, hidden_dim)

print("LayerNorm前, 均值: ")
mean_result = embedding.mean((-1))  # 计算维度 hidden_dim 的均值
print([f"%.2f" % float(y) for x in mean_result.detach().numpy().tolist() for y in x][:20], "...")
print("LayerNorm前, 方差: ")
var_result = embedding.var((-1))  # 计算维度 hidden_dim 的方差
print([f"%.2f" % float(y) for x in var_result.detach().numpy().tolist() for y in x][:20], "...")

# 该LayerNorm层的input的维度为[*, hidden_dim]，其仅对初始化时给定的hidden_dim这个维度做归一化
layer_norm = nn.LayerNorm(hidden_dim)
embedding = layer_norm(embedding)

print("LayerNorm后, 均值: ")
mean_result = embedding.mean((-1))  # 计算维度 hidden_dim 的均值
print([f"%.2f" % float(y) for x in mean_result.detach().numpy().tolist() for y in x][:20], "...")
print("LayerNorm后, 方差: ")
var_result = embedding.var((-1))  # 计算维度 hidden_dim 的方差
print([f"%.2f" % float(y) for x in var_result.detach().numpy().tolist() for y in x][:20], "...")
```

输出结果：

```
LayerNorm前, 均值: 
['0.23', '0.23', '0.18', '-0.06', '-0.45', '-0.24', '0.34', '0.23', '-0.47', '-0.44', '0.12', '-0.26', '-0.37', '0.33', '-0.50', '0.11', '0.14', '0.37', '-0.12', '0.31'] ...
LayerNorm前, 方差: 
['1.34', '0.51', '1.16', '0.90', '0.17', '0.50', '0.56', '0.61', '0.70', '1.06', '0.85', '1.26', '1.34', '1.45', '1.52', '0.75', '0.63', '1.37', '1.34', '1.51'] ...
LayerNorm后, 均值: 
['0.00', '-0.00', '-0.00', '0.00', '0.00', '-0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00', '-0.00', '0.00', '0.00', '-0.00', '-0.00', '-0.00', '0.00'] ...
LayerNorm后, 方差: 
['1.11', '1.11', '1.11', '1.11', '1.11', '1.11', '1.11', '1.11', '1.11', '1.11', '1.11', '1.11', '1.11', '1.11', '1.11', '1.11', '1.11', '1.11', '1.11', '1.11'] ...
```

可以看出，经过归一化之后，其均值为0，方差为1.11（这里为什么是1.11，而不是1，还没搞清楚）；

