# edge-SR:Super Resolution For The Masses

arxiv link: https://arxiv.org/pdf/2108.10335.pdf

## Highlights

<center>
   <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://pdf.cdn.readpaper.com/parsed/fetch_target/e1b1c5707aa288169be83aa6bc81126d_0_Figure_1.png" width = "65%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Figure 1. 模型结果对比
    </div>
</center>

1. 面向端侧设备推出了网络结构极小的超分模型并取得不错效果

---

## Main Work

### 问题描述

<center>
   <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://pdf.cdn.readpaper.com/parsed/fetch_target/e1b1c5707aa288169be83aa6bc81126d_2_Figure_2.png" width = "65%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Figure 2. 转置卷积实现upscaling（上）， 利用s2个filter处理后再pixelshuffle的高效实现
    </div>
</center>

文章指出传统的downscaling方法pooling和downsample处理的图像包括高频和低频信息，而实际采样限制会导致aliasing问题。为了避免aliasing，一种传统做法是先用低通滤波器过滤图像中的高频信息后再进行downsample。

据上所述，定义超分应当是downscaling的逆操作，即对应“filter-then-downsampling”逆操作为“upsampling-then-filter”。

upsampling-then-filter过程如Figure2上半部分所示，先将原图间隔插值(通常插值为0)的方式放大$s$倍，再将upsampling后的特征图送入卷积层进行处理。上述过程通常被称为transposed convolution，即转置卷积，反卷积。

转置卷积的实现方式非常低效，因为引入了很多无意义的0进行计算。一种更高效的实现方式为通过$s^2$ 个filter处理原图，再通过pixelshuffle将$s^2$个特征图重排成最后结果。本文提出网络模型遵循这一基本结构。

---

### 模型结构

#### 1.Maxout

![](https://pdf.cdn.readpaper.com/parsed/fetch_target/e1b1c5707aa288169be83aa6bc81126d_2_Figure_3.png)

本文提出的第一个网络结构即**Maxout**。模型结构如上图。

输入为$1\times H \times W$,经过一层卷积层得到$(s*s*C) \times H \times W$的特征图fmap1，fmap1通过系数为$s$的PixelShuffle层得到$C \times sH \times sW$的特征图fmap2，fmap2在第一维度上取max得到最终的$1 \times sH \times sW$的输出结果。其中$s$ 和$c$是模型可调节的超参。

```python
# eSR-Max implemented by pytorch
import torch
from torch import nn
class edgeSR_MAX(nn.Module):
    def __init__(self, C, k, s):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(s)
        self.filter = nn.Conv2d(
            in_channels=1,
            out_channels=s*s*C,
            kernel_size=k,
            stride=1,
            padding=(k-1)//2,
            bias=False,
        )
    def forward(self, input):
        return self.pixel_shuffle(
            self.filter(input)
        ).max(dim=1, keepdim=True)[0]
```

#### 2.Self–Attention

<div id="1" >跳转位置</div>

![](https://static.cdn.readpaper.com/aiKnowledge/screenshot/2022-05-14/eb1717ba65234d5f8b1846497d4c8d2a-18355445-c149-48c8-a203-560cc9829e4f.png)

eSR-TM是文中提出的引入注意力机制的模型，其主要结构如上。在PixelShuffle层前将卷积输出通道数加倍。在PixelShuffle后将一半通道的特征图通过softmax用于产生注意力系数，乘上另一半通道的特征图上，再累加得到最后的结果。

![](https://pdf.cdn.readpaper.com/parsed/fetch_target/e1b1c5707aa288169be83aa6bc81126d_3_Figure_5.png)

上图是文章给出eSR-TM的原理解释。其想表达的意思是先通过一个模板matching机制产生置信度，作用于特征图上赋予其权重。不过match机制中的卷积核也是训练出来的而不是手工设计的特征，所以其有效性分析也只是对黑盒模型的一种合理推测。

```python
# eSR-TM implemented by pytorch
import torch
from torch import nn
class edgeSR_TM(nn.Module):
    def __init__(self, C, k, s):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(s)
        self.softmax = nn.Softmax(dim=1)
        self.filter = nn.Conv2d(in_channels=1, out_channels=2*s*s*C, kernel_size=k, tride=1, padding=(k-1)//2, bias=False, )
    def forward(self, input):
        filtered = self.pixel_shuffle(self.filter(input))
        B, C, H, W = filtered.shape
        filtered = filtered.view(B, 2, C, H, W)
        upscaling = filtered[:, 0]
        matching = filtered[:, 1]
        return torch.sum(upscaling * self.softmax(matching),30 dim=1, keepdim=True)
```

#### 3.Transformer

![](https://static.cdn.readpaper.com/aiKnowledge/screenshot/2022-05-14/1760deca226b404e924980f8024c6dac-8e2890dd-9eb2-4ab6-9f65-76f201c434eb.png)

上图是本文提出的第三种模型，加入了Transformer设计。其相对Self Attention增加了Transformer中的Query和Key机制去产生自注意力系数，控制其余通道特征图生成最后结果。

```python
# edgeST-Transformer implemented by pytorch
import torch
from torch import nn
class edgeSR_TR(nn.Module):
    def __init__(self, C, k, s):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(s)
        self.softmax = nn.Softmax(dim=1)
        self.filter = nn.Conv2d(in_channels=1, out_channels=3*s*s*C, kernel_size=k, stride=1, padding=(k-1)//2, bias=False)
    def forward(self, input):
        filtered = self.pixel_shuffle(self.filter(input))
        B, C, H, W = filtered.shape
        filtered = filtered.view(B, 3, C, H, W)
        value = filtered[:, 0]
        query = filtered[:, 1]
        key = filtered[:, 2]
        return torch.sum(value * self.softmax(query*key), dim=1, keepdim=True )
```

#### 4.eSR-CNN

![](https://static.cdn.readpaper.com/aiKnowledge/screenshot/2022-05-14/830c3be3ba7f4068bf7bdaedc2c8e54c-8fa5a16b-2003-4e9f-8faf-2a83d593bc1a.png)

上图是本文提出的第四种模型结构eSR-CNN。这是一种加了Self Attention的[ESPCN](https://readpaper.com/paper/2476548250)网络，本文旨在测试注意力机制对ESPCN是否有加成作用。

```python
# eSR-CNN implemented by pytorch
import torch
from torch import nn
class edgeSR_CNN(nn.Module):
    def __init__(self, C, D, S, s):
         super().__init__()
         self.softmax = nn.Softmax(dim=1)
         if D == 0:
             self.filter = nn.Sequential(
                 nn.Conv2d(D, S, 3, 1, 1),
                 nn.Tanh(),
                 nn.Conv2d(14 in_channels=S, out_channels=2*s*s*C,16 kernel_size=3, stride=1,padding=1,bias=False,),
                 nn.PixelShuffle(s),
            )
        else:
            self.filter = nn.Sequential(
                nn.Conv2d(1, D, 5, 1, 2),
                nn.Tanh(),
                nn.Conv2d(D, S, 3, 1, 1),
                nn.Tanh(),
                nn.Conv2d(in_channels=S, out_channels=2*s*s*C, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(s),
            )
    def forward(self, input):
        filtered = self.filter(input)
        B, C, H, W = filtered.shape
        filtered = filtered.view(B, 2, C, H, W)
        upscaling = filtered[:, 0]
        matching = filtered[:, 1]
        return torch.sum(upscaling * self.softmax(matching),dim=1, keepdim=True)
```

### 实验



---

## 分析

虽然从结果来看这么大开销比bicubic方法好的有限，但是文章中的小型网络优化策略值得参考，去掉PixelShuffle部分就是输入输出同尺寸的模型，可用于降噪等场景。

此外文章实验部分很棒。
