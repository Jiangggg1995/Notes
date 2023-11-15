# 快速理解Attention和大模型

随笔记录一下最近学习大模型相关的信息，包括一些自己的理解。本文更关注通俗易懂的方式描述常见算子的计算过程，而不是深究一些学术思想。如有差错，请指正。

笔者： 江运衡

Email： yunheng_jiang@outlook.com

## 从Attention is all you need说起

一切都得从**Attention is all you need**这篇文章说起。这篇文章提出了一个self-attention机制的结构叫transformer，并以transformer为基础构建了一个encoder-decoder模型用于翻译任务。transformer结构成为了现在各种大模型的基础。

### 什么是attention？

文章中这么描述，*An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.* 

我的理解就是用一个query值，去一个键值对的map里通过相关性查询结果。这里的相关性本质是一种相似度，query和key相似度越高，说明当前key对应的结果对于此次query的结果影响越大，我们的attention就更应该倾向这个key对应的value。

举一个不太恰当的例子，我们有这么一个map，{水果：好吃，  中药：难吃}。现在我们用香蕉作为query，香蕉与水果这个key相似度为0.9，与中药相似度为0.1，所以香蕉好不好吃这个query的结果为0.9×好吃+0.1×难吃，得到结论：还是挺好吃的。

现在我们把用数学的语言再描述一次这个过程。这里Q(uery) K(ey) V(alue)都是矩阵。假设Q是一个长度为n的一维向量（或者1×n的矩阵），K是5×n的矩阵，V是5×m的矩阵（Key-Value键值对，所以K和V必须都是5）。回忆上面步骤，我们应该首先找Q和K的相似度，上面香蕉的例子我们直接假设相似度为0.9的，那向量的相似度应该怎么求呢？文中提出用点积（dot-product），$Q \cdot K^{T}$得到一个1× 5的向量，然后通过softmax将这5个值处理成上面提到相似度。然后通过1×5的相似度矩阵去乘V，得到最终1×m的结果。这个最终结果的维度和每个value的维度是一样的。
![](https://github.com/Jiangggg1995/Notes/blob/main/images/attention.png?raw=true)

理解了上面这个过程，我们对文中这个QKV计算过程应该就没有太多疑问了。原文的公式中除了上述描述过程外，在$Q \cdot K^{T}$后还统一处理一个$\sqrt{d_k}$，这个看网上解释是为了防止输入softmax的值过大而梯度趋近于0，不利于训练。不管如何，这个值不影响数据分布。（这里还有个可选的Mask层待补充）。

$Attention(Q,K,V) = softmax(\frac {QK^{T}}{d_k})$

为什么文中叫self-attention呢？因为这里的QKV其实都是由同一个输入x和不同的权重$W^{Q},W^{K},W^{V}$ 运算得到的（查阅llama2的代码发现这里就是一个MLP层），这里权重就是我们深度学习模型里面的可训练参数，随训练过程更新。

那么multi-head attention又是啥呢？其实就是由一个x和多组权重，计算得到多组QKV，可以认为是多个attention计算完后的结果concat起来，一种给大模型堆参数量的方式？

### Transformer结构
![](https://github.com/Jiangggg1995/Notes/blob/main/images/transformer.png?raw=true)
文中一个transformer除了multi-head attention结构外还有ADD，Norm和FeedForward。这里add其实就是一个skip connection（跳连接，残差连接），这个在resnet之后已经被广泛接受了，好处不再赘叙。

transformer里面的normlization可以提一句。它用的是layer norm而不是CNN传统的batch norm。我们知道做norm首先要求均值，再用均值对每个元素做归一。batch norm和layer norm的区别就在于，对什么元素集合做均值。batch norm是在batch的维度对每个元素做均值，而layer norm是对每个batch类的单个输入做norm。在NLP领域layer norm更适合，是因为输入sequence的长短可能不一样而需要padding，用batch norm的话padding的值会对归一化产生较大的不利影响。两者区别在于
![](https://github.com/Jiangggg1995/Notes/blob/main/images/BN_vs_LN.png?raw=true)

FeedForward这个前馈层从代码看就是一个mlp层，好像没啥特别的。

### Encoder和Decoder

encoder和decoder是一种模型结构。我觉得这个结构可能是一种比较符合人类直觉的设计理念。以翻译任务举例，将“我今天很开心”翻译成“I'm happy today”。人类翻译过程是这样的，阅读（encoder）“我今天很开心”这句话，理解其中这个意思，是人类大脑了解这句话表达的内容，然后翻译（decoder）成英文，按照理解的意思再次用英文表达成“I'm happy today”。（这里涉及到输入输出序列的长度不同，时间状语的前置和后置等问题，我自己尚有疑惑，暂时略过不表。）

文中的encoder和decoder类似于这种过程，encoder将输入编码（类似于阅读），将编码后的一种抽象概念输出给decoder，由decoder用另一种语言再次表达。

文中的encoder和decoder就是上面描述的transformer结构重复堆叠。

（Positional encoding待补充）

## 大模型

Attention is all you need文章后，transformer结构成了NLP领域的基本结构，被快速用于各种NLP任务。但是对于不同的任务，模型结构也不尽相同。行业发现：

Encoder only models：对于一些理解任务更高效，比如句子分类，名称主体识别等

Decoder only models：对于一些文本生成性的任务更高效

Encoder-decoder models：又称为sequence to sequenc，对于翻译、总结等理解输入并输出的任务更高效


大家发现堆叠transformer结构和更大的数据量，可以显著提升模型效果，因此在这条路上一直前进，也因此产生了今天“大模型”的说法。





## 关于hugging face

hugging face本身是一家做聊天机器人的公司。也不知道怎么走偏了，他们做的工具链火起来了，现在几乎成了NLP领域的github。大家开源的模型，都用他们家提供的库，调用起来只需要几行代码，很方便。

简单来看一下baichuan2-7B的代码
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Base", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-7B-Base", device_map="auto", trust_remote_code=True)
inputs = tokenizer('登鹳雀楼->王之涣\n夜雨寄北->', return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
```
这里`from transformers import...`不是我们以为的一个nn.Module派生的一个论文中的transformer结构。而是hugging face开发的一个python高阶函数库，里面提供了方便模型调用的各种函数接口。
从中`import AutoModelForCausalLM, AutoTokenizer`，这是两个用于自动化模型的函数。AutoModelForCausalLM是一个自动导入CauselLM任务模型的函数，只要给出这个模型在hugging face hub上的名称，函数会自动去hugging face hub上下载对应名称的模型结构、配置和权重的代码并构造出这个模型。同理，AutoTokenizer也是完成相同的工作。

简单介绍一下tokenizer。NLP任务输入的是句子，但是模型只能接受一个矩阵。所以我们需要将句子转换成矩阵表示。这个工作通常被认为是将句子token化，所以这个过程被叫做tokenizer。tokenizer会根据一个vocabulary将一个句子转换成一个一维向量。不同句子长短不同，但是输入矩阵的维度必须固定，因此tokenizer还会做一些padding和truncate等工作来保证多个句子生成的向量长度相同。细节此处不表。
像llama2的github主页直接给出了模型结构，在repo/llama/model.py中，代码中部分module用的是facebook开源的fairscale中的模块，其余基本结构都来自余pytorch本身，阅读起来并不复杂。但是meta好像封了国内ip，翻墙也没法申请下载llama2的权重，这里还得想办法...
像baichuan2的github主页只给出来hugging face接口的调用代码。模型代码和权重在hugging face hub上baichuan2的主页中的Files and versions中。运行其github中调用代码，会自动去hugging face的repo下载模型代码和权重，如果服务器无外网的话就只能自己手动去hugging face hub的Files and versions中下载对应的文件，然后在调用代码处指定本地模型地址。

## 关于算子融合
看下来发现大模型中的绝大多数算子都很常见，主要就是MLP的矩阵乘法和softmax。矩阵乘法的优化已经持续很多年了，Softmax看起来是个令人头疼的玩意，因为涉及指数运算。
因为大模型实在太大了，对内存和计算量的消耗都很大，跑起来又慢又耗电。所以目前有很多工程性工作对大模型做加速优化。像flash attention，微软家的deepspeed框架，nvidia的tensorRT-LLM等等。
网上介绍DeepSpeed提到将基本的transformer结构中的很多层算子全都fusion成一个了。具体代码明天再看。
（待补充）



