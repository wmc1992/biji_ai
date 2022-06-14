# KL散度

## 一、定义

KL散度是一种衡量两个概率分布 P 和 Q 的差异的方法；

对于连续随机变量的概率分布来说，KL散度公式为：

$$D_{KL}(p||q) = \int_x p(x) \log \frac{p(x)}{q(x)} dx$$

对于离散随机变量的概率分布来说，KL散度公式为：

$$D_{KL}(p||q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$$

机器学习和深度学习中使用的都是离散随机变量的概率分布，下面将仅讨论离散情况下的KL散度。

## 二、在机器学习中KL散度的作用

机器学习的目标就是：希望模型学到的分布 $p_{model}$ 与该任务的真实分布 $P_{real}$ 一致。

问题在于该任务的真实分布 $P_{real}$ 是无法获取到的，能够获取到的是训练集的分布 $P_{train}$，我们一般认为训练数据是从总体中独立同分布采样出来的，基于该条件下，就可以认为训练集的分布 $P_{train}$ 与真实分布 $P_{real}$ 是一致的。这样机器学习的目标就是：希望模型学到的分布 $P_{model}$ 与训练集的分布 $P_{train}$ 一致。

然后剩余的问题就是如何评估两个分布是否一致？答案是使用KL散度进行评估。因为KL散度的定义就是衡量两个概率分布 $p$ 和 $q$ 的差异。

两个分布越相近，KL散度越小；两个分布的差异越大，KL散度也越大；当两个分布相同时，KL散度为0。

## 三、熵、KL散度、交叉熵

先对这三个概念给出一个通俗但不严谨的描述：

* 熵：可以表示一个事件 A 的自信息量，即 A 包含多少信息；
* KL散度：可以表示从事件 A 的角度看，事件 B 有多大的不同；
* 交叉熵：可以表示从事件 A 的角度看，如何描述事件 B；

下面使用数据公示给出这三个概念的严谨的表示：

**熵：**

$$H(p) = - \sum_i p_i \log p_i$$

**KL散度：**

$$D_{KL}(p||q) = \sum_i p_i \log \frac{p_i}{q_i} = \sum_i p_i \log p_i - \sum_i p_i \log q_i$$

**交叉熵：**

$$H(p||q) = - \sum_i p_i \log q_i$$

> 注意熵和交叉熵公式中都带有一个负号，而KL散度的公式中并没有负号；

分析一下上面的KL散度的公式，左侧项 $\sum_i p_i \log p_i$ 很像是熵的公式，即 $-H(p)$；右侧项 $-\sum_i p_i \log q_i$ 就是交叉熵的公式，即 $H(p||q)$；所以会推导出如下公式：

$$D_{KL}(p||q) = H(p||q) - H(p)$$

即从公式上来说：KL散度等于交叉熵减熵。

## 四、机器学习中为什么多用交叉熵而不是KL散度

在第二部分的描述中已经很清晰的提到：机器学习就是将模型分布 $P_{model}$ 学到与训练集分布 $P_{train}$ 一致的过程。而衡量两个分布是否一致最直接的评估方式就是KL散度，那么为什么机器学习中常用交叉熵而不是KL散度？

在第三部分的最后推导出了一个公式，再次记录如下：

$$
\begin{equation}
\begin{split}   
D_{KL}(p||q) &= \sum_i p_i \log \frac{p_i}{q_i} \\
&= \big[ -\sum_i p_i \log q_i \big] - \big[ -\sum_i p_i \log p_i \big] \\
&= H(p||q) - H(p)
\end{split}
\end{equation}
$$

将上述公式放到机器学习这个具体应用场景中，公式中的概率分布 $q$ 就是需要学习才能得到的模型分布 $P_{model}$，公式中的概率分布 $p$ 就是训练集分布 $P_{train}$。

我们知道在机器学习中，训练集是固定的，所以训练集的熵 $H(p)$ 也是固定的，不随着模型的优化过程而变化。即在机器学习这个应用场景下 $H(p)$ 是常数。此时使用 $D_{KL}(p||q)$ 对模型优化与使用 $H(p||q)$ 对模型优化是等价的。由于使用交叉熵 $H(p||q)$ 时还能少计算一项，节省计算资源，所以机器学习中一般较多情况使用交叉熵。

## 五、KL散度的性质

最后记录一下KL散度的两个数学性质：

* **正定性**：$D_{KL}(p||q) \geqslant 0$

* **不对称性**：$D_{KL}(p||q) != D_{KL}(q||p)$

由于KL散度不具有**对称性**，所以KL散度不是一种距离（度量）。

> 一般来说距离（度量）要满足3个条件：正定性、对称性、三角不等式；

## Reference

* [https://blog.csdn.net/qq_40406773/article/details/80630280](https://blog.csdn.net/qq_40406773/article/details/80630280)

* [https://zhuanlan.zhihu.com/p/39682125](https://zhuanlan.zhihu.com/p/39682125)

* [https://hsinjhao.github.io/2019/05/22/KL-DivergenceIntroduction/](https://hsinjhao.github.io/2019/05/22/KL-DivergenceIntroduction/)

* [https://www.zhihu.com/question/336677048](https://www.zhihu.com/question/336677048)

* [https://www.zhihu.com/question/65288314/answer/244557337](https://www.zhihu.com/question/65288314/answer/244557337)
