# sklearn计算不平衡数据的权重

函数 `sklearn.utils.class_weight.compute_class_weight` 可以计算不平衡数据的权重，给数据量较少的类别一个较大的权重，给数据量较多的类别一个较小的权重。


测试代码如下：

```python
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def calculate_class_weights(data_list):
    labels = []
    for data in data_list:
        for label in data["labels"]:
            labels.append(label)

    class_weight_result = compute_class_weight('balanced', np.unique(labels), np.array(labels))

    label2weight = {}
    for label, weight in zip(np.unique(labels), class_weight_result):
        label2weight[label] = weight
    return label2weight


def test_calculate_class_weights():
    data_list = [
        {"content": "这里是文本", "labels": ["电影", "仙侠"]},
        {"content": "这里是文本", "labels": ["电影", "仙侠"]},
        {"content": "这里是文本", "labels": ["电影", "仙侠"]},
        {"content": "这里是文本", "labels": ["电影", "仙侠"]},
        {"content": "这里是文本", "labels": ["体育", ]},
    ]
    label2weight = calculate_class_weights(data_list)
    for label, weight in label2weight.items():
        print(label, weight)


if __name__ == "__main__":
    test_calculate_class_weights()
```

输出结果如下：

```
仙侠 0.75
体育 3.0
电影 0.75
```
