# Denoising Survey

paper link [Deep Learning on Image Denoising: An overview (readpaper.com)](https://readpaper.com/paper/2998718015)

## Dataset

[ Waterloo Exploration Database](https://readpaper.com/paper/2556068545)

[ polyU-Real-World-Noisy-Images datasets](https://readpaper.com/paper/2795722336)

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

## Traditional methods

### 单帧降噪

速度快：均值/中值/高斯滤波，小波滤波，Bilateral Filter， Guided Filter， Domain Transform Filter， Adaptive Manifold Filter

举例：[Multiresolution Bilateral Filtering for Image Denoising (readpaper.com)](https://readpaper.com/paper/2100925004)

效果好：统计模型， Low Rank， 稀疏编码， 字典学习， 自相似， 自相似+变换域

举例：BM3D算法

### 多帧降噪

多帧降噪等同于增加曝光时间，提高信号量，提高信噪比。主要步骤包括多帧对齐+融合。

举例： [Google HDR+](https://readpaper.com/paper/2552290192)



## Deep Learning methods

### DnCNN


