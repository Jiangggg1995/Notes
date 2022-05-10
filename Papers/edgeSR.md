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
1. 面向端侧设备推出了网络结构极小的超分模型

---

## Main Work

<center>
   <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" src="https://pdf.cdn.readpaper.com/parsed/fetch_target/e1b1c5707aa288169be83aa6bc81126d_2_Figure_2.png" width = "65%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      Figure 2. Classic s × s image upscaling is performed by a transposed convolutional layer. An efficient implementation splits the filter into s2 smaller filters that work at LR. The final output is obtained by multiplexing the s2 channels using a pixel–shuffle layer.
    </div>
</center>

定义超分问题

---
