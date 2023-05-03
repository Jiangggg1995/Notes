# Denoising Survey

[TOC]

<div STYLE="page-break-after: always;"></div>

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

---

<div STYLE="page-break-after: always;"></div>

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

双边滤波简介：以高斯滤波来看，其intuition在于认为像素在空间域上变化缓慢，所以利用其像素点周围像素的距离赋予其权重。但是这种思路在图像边缘部分是不妥的，图像边缘部分像素值差异可能很大，继续按照空间距离来赋权会带来边缘模糊的问题。

双边滤波的处理思路是将同时考虑像素点的空域（domain）信息和值域（range）信息，即如果相邻像素值差距很大，即使距离很近也会被赋予很低的权重，从而解决边缘模糊问题。

[双边滤波的python和c++实现](https://github.com/anlcnydn/bilateral)

[双边滤波的一种快速实现](https://github.com/ufoym/recursive-bf)

小波变换简介：傅里叶是将信号分解成一系列不同频率的正余弦函数的叠加，同样小波变换是将信号分解为一系列的小波函数的叠加(或者说不同尺度、时间的小波函数拟合)，而这些小波函数都是一个母小波经过平移和尺度伸缩得来的。小波变换不同于傅里叶变换的地方是傅里叶函数是分解成正弦波，而小波变换的小波母函数可以不同，不同的小波母函数的变换结果也不尽相同。符合小波特征的函数都可以作为小波母函数。

举例：[Multiresolution Bilateral Filtering for Image Denoising (readpaper.com)](https://readpaper.com/paper/2100925004)

![](https://pdf.cdn.readpaper.com/parsed/fetch_target/08c14da43c2fb9eb46c2ee68837d8537_3_Figure_5.png)

**效果好**：统计模型， Low Rank， 稀疏编码， 字典学习， 自相似， 自相似+变换域

举例：[BM3D算法](https://readpaper.com/pdf-annotate/note?noteId=690452415860723712&pdfId=4531127163620581377)

![](https://pdf.cdn.readpaper.com/parsed/fetch_target/b096b1a0b2a34606189d6d4e352313ce_4_Figure_3.png)

BM3D算法是对NLM（No local mean,非局部均值）算法的继承和发展。NLM算法的主要思想是，以目标区域的图片块（block）为基础，在图像中搜索跟目标区域相似的块，再根据相似度对这些块进行加权平均，求得降噪结果。其Intution主要基于图像中区域的自相似性。

BM3D算法Step1：**首先**也会寻找多个相似块，将这些相似块堆叠成3D数组。**继而**对这个3D数组的第三维即图块叠起来后，每个图块同一个位置的像素点构成的数组，进行DCT变换。**接着**采用硬阈值的方式将变换后的数组小于阈值的成分置为0。**同时**统计非零成分的数量作为后续权重的参考。**然后**将数据进行3D逆变换。**最后**这些图块逆变换后放回原位，利用非零成分数量统计叠加权重，将叠放后的图除以每个点的权重就得到<u>基础估计</u>的图像。

Step2：**开始**第二步中的聚合过程与第一步类似，不同的是，这次将会得到两个三维数组：噪声图形成的三维矩阵和<u>基础估计</u>结果的三维矩阵。**接着**两个3D矩阵都进行二维和一维变换。用维纳滤波将噪声图形成的3D矩阵进行系数放缩，该系数通过基础估计的3D矩阵的值以及噪声强度得出。**最后**与Step1中一样，只是此时加权的权重取决于维纳滤波的系数和噪声强度，得到最终结果。

论文中给出的效果如图：

![](https://pdf.cdn.readpaper.com/parsed/fetch_target/b096b1a0b2a34606189d6d4e352313ce_9_Figure_6.png)

[pure Python implement of BM3D (github.com)](https://github.com/Ryanshuai/BM3D_py)

### 多帧降噪

多帧降噪等同于增加曝光时间，提高信号量，提高信噪比。主要步骤包括多帧对齐+融合。

举例： [Google HDR+](https://readpaper.com/paper/2552290192)

---

<div STYLE="page-break-after: always;"></div>

## Deep Learning methods

### DnCNN

#### Paper

[Link]([ReadPaper](https://readpaper.com/pdf-annotate/note?noteId=686868716781154304&pdfId=4500372972119941121))

#### Architecture

![](https://static.cdn.readpaper.com/aiKnowledge/screenshot/2022-05-28/3ae694d8c9c24126b1ba65cc55948601-6cf548c3-c6b3-4bfd-88ed-e8ec2db2a096.png)

文章将噪声定义为$y=x+v$，其中$x$是无噪声图像，$v$是噪声，$y$为含噪图像。一般去噪的过程相当于输入$y$，求$x$。

DnCNN的网络结构相当于一个改版的vgg，其模型内部并没有skip connection。不过模型的输出却并不是$x$，而是$v$，相当于一个global的skip connection。文章采用MSE作为损失函数，数据集只有无噪声图像，人工增加加性高斯白噪声构建需要的输入和输出。

#### Code

[DnCNN-官方(github.com)](https://github.com/cszn/DnCNN)

[DnCNN-pytorch-第三方](https://github.com/SaoYan/DnCNN-PyTorch)

---

### FFDNet

#### Paper

[Link](https://readpaper.com/pdf-annotate/note?noteId=686864061743390720&pdfId=4498438318995431425)

#### Architecture

![](https://pdf.cdn.readpaper.com/parsed/fetch_target/f9d5f02da61de664e719086791a8e07f_2_Figure_1.png)

FFDNet是DnCNN基础上的一次升级，算法的backbone并没有改变，但是输入输出不同。FFDNet的输入为原始带噪图像降采样获得的四张子图以及一张由用户输入的参数$\sigma$生成的噪声水平图像，输出为四张降噪后的子图，通过上采样获得最终的降噪图像。使用的损失函数仍然是MSE。

由原图生成4张子图过程就是pixel shuffle的逆操作，根据超参数$\sigma$生成噪声水平图）。从设计上来看，该网络的处理能力更加灵活，似乎能够针对不同噪声水平来做不同层度的降噪处理，这点对于深度学习方法来说是值得**参考**的。

#### Code

[FFDNet](https://github.com/cszn/FFDNet)

```python
# code to generate noise
noise_level_img = 15 #15 30 50
img = cv2.imread
img = np.float32(img/255.)
noise = np.random.normal(0, noise_level_img/255., img.shape)
img_ns = img+noise
```

---

### CBDNet

#### Paper

[Link](https://readpaper.com/pdf-annotate/note?noteId=690817275652734976&pdfId=4544599851060060161)

#### Architecture

![](https://pdf.cdn.readpaper.com/parsed/fetch_target/b6eea24aa9e0c41533ded3a1b3a22222_2_Figure_2.png)

CBDNet是FFDNet网络基础上的又一次升级。对于FFDNet网络中的噪声水平为$\sigma$的噪声图，CBDNet中采用一个五层的小网络估计出来，另一部分用去去噪的主要网络也由之前的VGG换成了一个16层的UNet。这种设计看起来有点自注意力机制的味道，跟SENet有点像，对自身的噪声水平进行估计的方法比较符合人类解决问题的思路，从论文给出的结果看也是一种work的思路，可以**参考**。

此外论文设计了一种非对称的损失函数。作者总结已有的去噪方法FFDNet和BM3D的实验结果，发现对于输入图片的AWGN噪声的SD（标准差）等于或高于ground truth的SD时，能得到不错的去噪表现，而当输入的SD小于ground truth的SD时去噪效果会很差。因此作者设计了一种非对称损失函数：

$L_{asymm}=\sum\limits_{i}|\alpha-\prod_{(\hat{\sigma(y_{i})}-\sigma(y_{i})<0}|\cdot(\hat{\sigma(y_{i})}-\sigma(y_{i})^2$

其中 $I_{e} = 1 \quad for\quad e < 0\quad and \quad0\quad otherwise$

此外作者还引入了一个**total variation**(TV) **regularizer**来约束 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%28y_i%29) 的平滑度：

$L_{TV} = \left \|\bigtriangledown_{h} \hat{\sigma}(y) \right \|_{2} ^{2}+\left \|\bigtriangledown_{v} \hat{\sigma}(y) \right \|_{2} ^{2}$

最后还有一个MSE损失：

$L_{rec}=\left \| \hat{x}-x\right \|_{2}^{2}$

最终的损失函数由上面三项相加得到：

$L=L_{rec}+\lambda _{asymm}L_{asymm}+\lambda _{TV}L_{TV}$

#### Code

[CBDNet](https://github.com/GuoShi28/CBDNet)

```python
# CBDNet loss unofficial implementation
class fixed_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out_image, gt_image, est_noise, gt_noise, if_asym):
        l2_loss = F.mse_loss(out_image, gt_image)

        asym_loss = torch.mean(if_asym * torch.abs(0.3 - torch.lt(gt_noise, est_noise).float()) * torch.pow(est_noise - gt_noise, 2))

        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, : ,1:])
        h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x-1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w

        loss = l2_loss +  0.5 * asym_loss + 0.05 * tvloss

        return loss

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]
```

---

### PMRID

#### Paper

[Link](https://readpaper.com/pdf-annotate/note?noteId=690821675036463104&pdfId=4498439968699080705)

#### Architecture

![](https://pdf.cdn.readpaper.com/parsed/fetch_target/bdf70df6b54203a0144f44e3d86f36e9_6_Figure_5.png)

PMRID是旷世推出的一个面向移动端设备的去噪模型。其网络模型是一个大体上是一个改版的UNet。模型主要特点有：1.使用5×5的separable conv增大感受野降低计算量；2.使用stride 为2的卷积来进行下采样；使用2x2的deconv来进行上采样；3.使用3x3的separable conv来进行skip connection时的通道匹配；

虽然其面向移动端的网络结构优化工作很棒，但这篇文章网络结构之外的变换部分非常值得**参考**。在硬件实际取景过程中，ISO是动态变化的，而不同的ISO带来的噪音强度是不一样的。所以文章采取了一种k-sigma变换的方式将噪声强度归一化，从而避免不同噪声强度对去噪算法带来的负担。

![](https://pdf.cdn.readpaper.com/parsed/fetch_target/bdf70df6b54203a0144f44e3d86f36e9_3_Figure_2.png)

假设理想无噪模型:

$$
x^{*}=g\alpha u^{*}
$$

其中 $u^{*}$为撞击到像素点上的光子的期望数， $\alpha$为光电转换因子，$g$为模拟增益，$x^{*}$是无噪声像素值。

正如我们开篇所述的噪声类型很多，所以考虑到噪声，实际的应为：

$$
x=g*(\alpha u +n_{d})+n_{r}
$$

$u$表示实际收集到的光子，是满足$u^{*}$的泊松分布：$u \sim P(u^*)$。$n_d \sim N(0, \sigma^2_d)$，$n_r \sim N(0, \sigma^2_r)$为模拟增益前的高斯形态噪音。

因此：

$$
x\sim (g\alpha)P(\frac{x^*}{g\alpha})+N(0,g^2\sigma^2_d+\sigma^2_r)
$$

可知$x$符合一个高斯泊松分布，令$k=g\alpha$，$\sigma^2=g^2\sigma^2_d+\sigma^2_r$，则上式可化简为：

$$
x\sim (g\alpha)P(\frac{x^*}{k})+N(0,\sigma^2)
$$

于是我们发现$k$和$\sigma^2$只跟$g$有关，$g$为模拟增益，也就是ISO。根据$x$服从高斯泊松分布可得：

$$
\left\{\begin{array}{l}
  E(x)=x^* \\ Var(x)=kx^*+\sigma^2
\end{array}\right.
$$

如此，我们可以通过拍摄某个ISO下一系列的raw图，求每一个像素位置的均值$E(x)$，之后将所有均值为一样的像素值集合起来求其方差$Var(x)$，然后将该点画在以E(x)为横轴，$Var(x)$为纵轴的图中，使用线性方程拟合便得到该ISO下的$k$和 $\sigma^2$。

上述过程揭示了k和sigma的含义。为了得到一个ISO independent的数据，作者定义了一种线性变换：

$$
f(x)=\frac{x}{k}+\frac{\sigma^2}{k^2}
$$

为什么要对输入作上述线性变换呢？我们根据之前$x$的高斯泊松分布可得：

$$
f(x)\sim P(\frac{x^*}{k})+N(\frac{\sigma^2}{k^2},\frac{\sigma^2}{k^2})
$$

为了方便分析该分布，可把泊松分布 $P(\lambda)$视为$N(\lambda,\lambda)$ , 因此有:

$$
\begin{equation}
\begin{aligned}
P(\frac{x^*}{k})+N(\frac{\sigma^2}{k^2},\frac{\sigma^2}{k^2})
&\approx N(\frac{x^*}{k},\frac{x^*}{k})+N(\frac{\sigma^2}{k^2},\frac{\sigma^2}{k^2})
\\&=N(\frac{x^*}{k}+\frac{\sigma^2}{k^2},\frac{x^*}{k}+\frac{\sigma^2}{k^2})
\\&=N[f(x^*),f(x^*)] \nonumber
\end{aligned}
\end{equation}
$$

也就是$f(x)\sim N[f(x^*),f(x^*)]$

由此发现这种转换转换排除了ISO的影响，且输出和输出符合是由简单的高斯分布特征。

#### Code

[PMRID](https://github.com/MegEngine/PMRID.git)

```python
# k sigma transformation
# a little different beacuse of https://github.com/MegEngine/PMRID/issues/3
class KSigma:
    def __init__(self, K_coeff: Tuple[float, float], B_coeff: Tuple[float, float, float], anchor: float, V: float = 959.0):
        self.K = np.poly1d(K_coeff)
        self.Sigma = np.poly1d(B_coeff)
        self.anchor = anchor
        self.V = V

    def __call__(self, img_01, iso: float, inverse=False):
        k, sigma = self.K(iso), self.Sigma(iso)
        k_a, sigma_a = self.K(self.anchor), self.Sigma(self.anchor)

        cvt_k = k_a / k
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a

        img = img_01 * self.V

        if not inverse:
            img = img * cvt_k + cvt_b
        else:
            img = (img - cvt_b) / cvt_k

        return img / self.V
```

---

<div STYLE="page-break-after: always;"></div>

## Some idea

### 1.模型小型化

模型小型化对运算速度和硬件成本有着最直接的影响，如何在确保效果的情况下把模型做小是工作的重要方向。从其他论文中借鉴了一些不错的小型化模型结构供参考：

1.[edge-SR](https://arxiv.org/pdf/2108.10335.pdfhttps://arxiv.org/pdf/2108.10335.pdf)中的eSR-TM模型

![](https://static.cdn.readpaper.com/aiKnowledge/screenshot/2022-05-14/eb1717ba65234d5f8b1846497d4c8d2a-18355445-c149-48c8-a203-560cc9829e4f.png)

2.[dh_isp](https://readpaper.com/pdf-annotate/note?noteId=691056777091895296&pdfId=4558136780486877185)

![](https://pdf.cdn.readpaper.com/parsed/fetch_target/a300bacf9a070a958c5a4d95af106f9a_4_Figure_3.png)

3.[AIISP](https://readpaper.com/pdf-annotate/note?noteId=691056777091895296&pdfId=4558136780486877185)

![](https://pdf.cdn.readpaper.com/parsed/fetch_target/a300bacf9a070a958c5a4d95af106f9a_5_Figure_4.png)

![](https://pdf.cdn.readpaper.com/parsed/fetch_target/a300bacf9a070a958c5a4d95af106f9a_5_Figure_5.png)

4.[ENERZAi](https://readpaper.com/pdf-annotate/note?noteId=691056777091895296&pdfId=4558136780486877185)

![](https://pdf.cdn.readpaper.com/parsed/fetch_target/a300bacf9a070a958c5a4d95af106f9a_6_Figure_6.png)

5.[isp_forever](https://readpaper.com/pdf-annotate/note?noteId=691056777091895296&pdfId=4558136780486877185)

![](https://pdf.cdn.readpaper.com/parsed/fetch_target/a300bacf9a070a958c5a4d95af106f9a_6_Figure_8.png)

6.[CVML](https://readpaper.com/pdf-annotate/note?noteId=691056777091895296&pdfId=4558136780486877185)

![](https://pdf.cdn.readpaper.com/parsed/fetch_target/a300bacf9a070a958c5a4d95af106f9a_7_Figure_10.png)

### 2.变换域的探究

传统方法中将图像从空间域转到其他域进行去噪是有效且常见的。旷世的那篇论文在深度学习方法中也采用了k-sigma变换这种类似的域变换方法。能否将深度学习和域变换两种有效方法耦合起来是接下来工作中可以探究的一个点。

### 3.数据集和损失函数

跟目标检测这种任务不一样的是，去噪的数据集的label并不是很统一。有效的数据集对模型效果应该会有很重要的影响。常见的去噪数据集制作方法有：1.图片加人工噪声（如加性高斯白噪声）；2.高ISO长曝光/低ISO短曝光，制作数据集。如何制作出最贴近ISP数据中噪声数据集也是可探究的点。

上述论文中采用的损失函数多为L1 loss和MSE loss。而PSNR和SSIM作为评价指标这些年也备受争议，因为实验中发现L1 loss有助于提高PSNR表现，但是PSNR高的结果并不一定符合人类观感中对图像“高质量”的评价。如何用某种指标量化人类视觉中的“高质量”感受很重要，据说这也是荣耀手机的AIISP团队主要工作方向。

此外有文章提到tanh比relu在小型网络中表现更好，可以作为实验点探究。

---

## Refenenced

1.[Deep Learning on Image Denoising: An overview (readpaper.com)](https://readpaper.com/paper/2998718015)

2.[图像视频降噪的现在与未来——从经典方法到深度学习 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/106191981)
