# Denoising Survey

paper link [Deep Learning on Image Denoising: An overview (readpaper.com)](https://readpaper.com/paper/2998718015)
---

## Problem

图像视频的采集、压缩、传输、显示等过程中都会发生不同层度的失真。我们ISP去噪关注的是图像采集过程中的一种普遍失真：噪声。ISP图像噪声主要有以下几种。

1. 光子散列噪声（Photon shot noise）
   
   当光子按照一定的概率分布撒到探测器上会因为量子涨落形成光子散列噪声，这种噪声无法通过改善相机设计来减少。光子散列噪声符合泊松分布。

2. 热噪声（Thermal noise）
   
   sensor中自由电子的布朗运动引起的噪声，服从高斯分布，又称高斯白噪声。

3. 读出噪声（Read noise）
   
   是读出电路前后电压随机波动的结果，服从高斯分布。

4. 暗电流噪声（Dark current noise）
   
   sensor由于热激发产生的电子形成电流被称为暗电流，温度越高暗电流也随之变高。暗电流噪声服从泊松分布。

---

## Dataset

[ Waterloo Exploration Database](https://readpaper.com/paper/2556068545)

[polyU-Real-World-Noisy-Images datasets](https://readpaper.com/paper/2795722336)

数据集制作方法：

1. 图片加人工噪声（如加性高斯白噪声）

2. 高ISO长曝光/低ISO短曝光，制作数据集

---

## Traditional methods

### 单帧降噪

**速度快**：均值/中值/高斯滤波，小波滤波，Bilateral Filter， Guided Filter， Domain Transform Filter， Adaptive Manifold Filter等

<center>
   <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://github.com/Jiangggg1995/Notes/blob/main/images/filters.png?raw=true" width = "65%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Figure 1. 均值/中值/高斯滤波
    </div>
</center>

双边滤波简介：以高斯滤波来看，其intuition在于认为像素在空间域上变化缓慢，所以利用其像素点周围像素的距离赋予其权重。但是这中思路在图像边缘部分是不妥的，边缘部分像素值差异很大，继续按照空间距离来赋权会带来边缘模糊的问题。

双边滤波的处理思路是将同时考虑像素点的空域（domain）信息和值域（range）信息，即如果相邻像素值差距很大，即使距离很近也会被赋予很低的权重，从而解决边缘模糊问题。

举例：[Multiresolution Bilateral Filtering for Image Denoising (readpaper.com)](https://readpaper.com/paper/2100925004)

![](https://pdf.cdn.readpaper.com/parsed/fetch_target/08c14da43c2fb9eb46c2ee68837d8537_3_Figure_5.png)

双边滤波+小波

**效果好**：统计模型， Low Rank， 稀疏编码， 字典学习， 自相似， 自相似+变换域

举例：BM3D算法

### 多帧降噪

多帧降噪等同于增加曝光时间，提高信号量，提高信噪比。主要步骤包括多帧对齐+融合。

举例： [Google HDR+](https://readpaper.com/paper/2552290192)

## Deep Learning methods

### DnCNN
