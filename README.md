# VGG16实现分类任务

VGG是2014年由牛津大学著名研究组VGG(Visual Geometry Group)提出，斩获当年ImageNet竞赛中定位任务第一名和分类任务第二名。

## 原理图

![IKeETO.png](https://z3.ax1x.com/2021/11/05/IKeETO.png)

## 原理简述

例如有张224x224的RGB图片，我们让他通过两个卷积核为3的卷积层，再通过最大池化层**（核尺寸为2，步距为2）**。至于卷积层的输入输出维度，参考下图，我们一般常用VGG16模型，**最终用多个FC实现分类，也可将FC换成卷积核为1的卷积层**。



![IKeZkD.png](https://z3.ax1x.com/2021/11/05/IKeZkD.png)


## 数据集

同样选用本人常用的海贼王数据集来测试，可以根据个人需求修改。

**下面提供一个简单数据集。**

Kaggle的鸟儿分类数据集，共315个分类

https://www.kaggle.com/gpiosenka/100-bird-species

**可以用kaggle命令选择下载**

```shell
kaggle datasets download -d gpiosenka/100-bird-species
```

## 数据预处理

讲图片预resize为224x224的图片，并进行随机翻转，**可参考图像增广**

**https://www.quarkhackers.space/2021/10/15/%E5%9B%BE%E5%83%8F%E5%A2%9E%E5%B9%BF/**

最终转为Tensor，并进行归一化处理。

**参考代码**

```python
transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
```

## 训练结果

![I3kvnI.png](https://z3.ax1x.com/2021/11/07/I3kvnI.png)

**结果分析**

在训练过程，针对此数据集，VGG16的学习率不宜过高，过高会出现loss不收敛

**在loss为0.01时结果如下**

![I3AK4U.png](https://z3.ax1x.com/2021/11/07/I3AK4U.png)

**VGG网络acc可达到百分之90以上，只是对于小数据集而言，模型过于繁重，可选用层数少的模型，并且在后面3层FC，可根据个人需求修改。**
