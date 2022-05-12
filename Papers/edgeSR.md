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

本文提出的第一个网络结构即**Maxout**。模型结构如上图，pytorch实现如下图。

输入为$1\times H \times W$,经过一层卷积层得到$(s*s*C) \times H \times W$的特征图fmap1，fmap1通过系数为$s$的PixelShuffle层得到$C \times sH \times sW$的特征图fmap2，fmap2在第一维度上取max得到最终的$1 \times sH \times sW$的输出结果。其中$s$ 和$c$是模型可调节的超参。

```python
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
