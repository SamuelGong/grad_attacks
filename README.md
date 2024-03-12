<p align="center">
    <a href="https://github.com/SamuelGong/GradAttacks"><img src="https://img.shields.io/badge/-github-teal?logo=github" alt="github"></a>
    <img src="https://badges.toozhao.com/badges/01HRPJRYYBJW7V9K56NSXB2BBT/green.svg" />
</p>

<h1 align="center">Gradient Attacks against GPT-2: A Self-Learning Note</h1>

## Table of Contents
1. [概述](#1-概述)
2. [TAG 复现结果和难点讨论](#2-tag-复现结果和难点讨论)
   - [结果概览（GPT-2 自回归文本生成）](#21-结果概览gpt-2-自回归文本生成)
   - [要点讨论](#22-要点讨论)
3. [杂项](#3-杂项)

## 1 概述

本仓库主要为作者自习用，另公开以促进技术交流。
建仓缘由是看到如下两篇梯度泄露攻击论文，于是想在自己搭建的虚拟联邦学习（FL）靶子复现以学习其技术：

1. [TAG: Gradient Attack on Transformer-based Language Models (EMNLP '21)](https://arxiv.org/abs/2103.06819)
   - 该论文提出了 TAG，据称是第一个针对 NLP 领域的梯度泄露攻击方法。
   实质上，TAG 的技术路线直接继承了其祖师爷 [Deep Leakage from Gradients (NeurIPS '19)](https://arxiv.org/abs/1906.08935) (简称 DLG)
   的**梯度匹配**（Gradient Matching）思想——优化虚拟数据，通过令梯度逼近真实梯度来让虚拟数据逼近真实数据。
   单从技术上看，TAG（唯一）重要的创新点在于其**目标函数的定义**：(i) 除常规的梯度差的 L2 范数外，额外引入了 L1 范数，
   且 (ii) 该项的系数引入了变化的思想（根据参数所在的层数）。
   作者相信这样做是 TAG 相比以往工作更能应对各种（i）模型初始化场景（不同方式的随机初始化/从预训练参数初始化），(ii) 各种数据集，以及 (iii) 各种模型的原因。
2. [APRIL: Finding the Achilles' Heel on Privacy for Vision Transformers (CVPR '22)](https://arxiv.org/abs/2112.14087)
   - 待定。

**【参考资料】** 虽然以上文章皆无直接公开的代码，但认真搜索一番想必不难找到一个名为 [JonasGeiping/breaching](https://github.com/JonasGeiping/breaching) 
的 GitHub 仓库（后称为「参考仓库」），它其实已经给出两篇文章的 demo：

1. [TAG - Optimization-based Attack - FL-Transformer for Causal LM.ipynb](https://github.com/JonasGeiping/breaching/blob/main/examples/TAG%20-%20Optimization-based%20Attack%20-%20FL-Transformer%20for%20Causal%20LM.ipynb)
2. [APRIL - Analytic Attack - Vision Transformer on ImageNet.ipynb](https://github.com/JonasGeiping/breaching/blob/main/examples/APRIL%20%20-%20Analytic%20Attack%20-%20Vision%20Transformer%20on%20ImageNet.ipynb)

参考仓库的作者名作 [Jonas Geiping](https://jonasgeiping.github.io/)，现在是 ELLIS Institute 和 MPI-IS 的独立 PI。
这位是梯度泄露攻击的专家，上述仓库是其出于热爱，为支持多种现成梯度泄露攻击而实现的通用框架。
我最早接触他的文章是 [Inverting Gradients - How Easy Is It to Break Privacy in Federated Learning? (NeurIPS '20)](https://arxiv.org/pdf/2003.14053.pdf),
该文章基于 DLG，使用 (i) 已知标签的条件，(ii) 新的梯度距离度量，以及 (iii) 新的迭代优化器，大大提升了 CV 领域的梯度泄露攻击效果。

**【复现路线】** 由于之前有过 CV 领域进行梯度泄露攻击的工程经验（约2023年2月到3月），
我这里的复现并不照搬参考仓库的代码。
而是尝试先 (i) 按照自己的工程经验和论文理解搭建完整的**最小攻击框架**。
之后再 (ii) 根据提高攻击效果的具体需要，分析与参考仓库代码的异同以找到改进现有框架的思路。
因而本仓库不仅包含能够（基本）复现论文攻击的轻量级代码，也自然融入了复现过程的难点分析。
相信本仓库相比参考仓库，更有助于广大初学者避开一些隐形的「坑」。

**【环境配置】**

要运行本仓库的代码，只需要按照下述指令配置好环境即可：

```
bash install.sh
```

然后每一次运行指令都应在`grad_attacks`的 conda 环境中操作，进入指令为

```
conda activate grad_attacks
```

## 2 TAG 复现结果和难点讨论

### 2.1 结果概览（GPT-2 自回归文本生成）

这里使用预训练的 GPT-2，任务为自回归文本生成，输入为一个 16 个 token 的序列：

```
 The Tower Building of the Little Rock Arsenal, also known as U.S.
```

也即预训练的 GPT-2 会学习自回归地生成以上序列，而攻击者据其产生的梯度尝试还原该序列。
以不同的攻击迭代轮数分别进行一次实验，参考结果为：

1. 迭代 1000 轮（约 180 秒）：token 准确率 0.1875，RougeL 0.3704，具体恢复文本如下

```
 The Tower Tower Building the positively the Restaurant Little and Rockanaly Jab obfusc supplementalfet
```


2. 迭代 20000 轮（约 3359 秒）：token 准确率 0.25，RougeL 0.4444，具体恢复文本如下

```
 The Tower Tower Building the of of Arsenal Little meteor alsoBear Rock being Ufet
```

上述攻击过程对应复现命令分别为

```bash
python main 1_gpt2_gen/wikitext_2_tag_s42_it1000_lr005.yml
python main 1_gpt2_gen/wikitext_2_tag_s42_it20000_lr005.yml
```

程序将在前台运行直至结束，期间往文件写日志。上述实验参考的日志文件可见于

* [1_gpt2_gen/wikitext_2_tag_s42_it1000_lr005_20240312-132658.log](1_gpt2_gen/wikitext_2_tag_s42_it1000_lr005_20240312-132658.log)
* [1_gpt2_gen/wikitext_2_tag_s42_it20000_lr005_20240312-133822.log](1_gpt2_gen/wikitext_2_tag_s42_it20000_lr005_20240312-133822.log)

里面包含攻击过程的快照。欢迎尝试和拍砖。

### 2.2 要点讨论

在迁移 CV 领域的梯度泄露攻击经验至本次攻击复现的过程中，我遇到了两个不平凡的问题，以下是相关的思考。

#### 2.2.1 确定虚拟特征

图像分类模型的输入特征是图片。
使用梯度匹配的思想进行梯度泄露攻击时，我们自然地将虚拟特征（dummy feature）初始化为一张随机的图片。
其中每个图层中每个像素值都是连续的随机变量，独立地服从某种简单的分布（如正态分布或均匀分布）。

考虑自回归文本生成模型，虚拟特征类似地应当初始化为一个随机句子。挑战在于，什么是随机句子（假定长度可知）？
鉴于 GPT-2 模型的直接输入为`input_ids`，即句子中每个 token 的在词表的索引值。
假设词表大小为`V`，一个很直接的构造虚拟特征的做法，就是假设随机句子中每个 token 的索引值服从`[1, V]`均匀分布。

这的确是个随机的句子。
然而，和像素值不同，词表的索引值既离散、又缺乏语义空间上的局部性，
以其作为优化特征可想而知是低效的（如果它还能收敛的话）。
其实，这里虚拟特征不应再对应于原本端到端工作流中的输入——token的索引值——而是应该对标文本嵌入，
即每个 token 对应的词向量。
词向量既连续，又在语义空间上具有局部性，对梯度匹配的优化过程更加友好。
从这个角度出发，随机句子应该是一系列独立生成的随机词向量。

另外，确认这个思路后，工程实现上对应地、严格来说应该作出的适应有（其中第二点是必要的）：

* 【模型结构的改变】词向量不是语言模型输入的默认格式，要让目标模型额外接收这种格式，通常需要改变模型结构。
大概地，这需要有函数负责拿到模型中进行词嵌入的参数，用以生成词向量后，再送入正确的层。

  > 幸运地，本次实验针对的 GPT-2 具有很强的扩展性，其原本的设计就直接支持文本嵌入作为输入（通过参数`inputs_embeds`），
不过要获得并利用这个观察，也需要对其[源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py )有足够了解。

* 【模拟训练流程的改变】虚拟特征梯度的产生建立在“其是词向量、并跳过了原本的第一层词嵌入”的基础上的。
相应地，在模拟训练过程中，真实梯度的产生也应采取这种方式，这样两边产生的梯度进行匹配才有意义。
具体地，真实数据喂入目标模型也应该先转化为词向量（且这步不应生成梯度），然后采用上一点的解决方法输入到目标模型。

* 【TAG 相关的适应】TAG 在计算目标函数（原文公式5）有一个依赖于参数所在模型的层数而变化的权重`alpha`。
具体地，`alpha` 设置为越靠近输入的层越大，其设计背后的思考为“越靠近输入的层受输入的变化影响越大，也就对恢复输入更有价值”。
现在，由于虚拟特征是词向量，而非 token 索引值，其经过目标模型时将跳过词嵌入层。
因之词嵌入层不再是最靠近输入的层（相反，在 GPT-2 中，由于词嵌入层和 最后一层预测头 是 tied 的，该层实际上变成最远离输入的）。
注意到这点后，需要在写目标函数时格外关注（本仓库的实现），或重构每层梯度出现的顺序（参考仓库的实现）。

  > 这一点也许没有那么重要，因为我发现即使不那样做，
  > 攻击效果在 1000 次迭代的实验中也没有显著变差（具体参见日志[1_gpt2_gen/wikitext_2_tag_s42_it1000_lr005_wrong_alpha_20240312-143638.log](1_gpt2_gen/wikitext_2_tag_s42_it1000_lr005_wrong_alpha_20240312-143638.log)）。
  > 如果这个确为普遍规律，那么倒推回来，每层使用不同的 `alpha` 也没有那么重要，这篇文章的技术贡献也就更少了。

#### 2.2.2 确定虚拟标签

如何初始化虚拟标签也是个问题。
分类任务中，数据的标签很自然地，是离散的类别索引值。
进行梯度匹配攻击时，出于类似上述的原因，虚拟标签一般初始化为随机的类别 logit 值。
其为连续值而方便优化，且在经过`Softmax`函数和`argmax`函数后可与类别索引值等价。

然而，自回归文本生成并不天然是个分类任务。
要学习生成一个句子`S`，概念上并不需要什么标签。
只有在认识到 (i) 语言建模的目标函数是“用`<x`的 token 预测`x`”，
以及 (ii) GPT-2 是并行处理每个 `<x` 的时候，
其与分类任务才有一定的联系——其中特征可视作`S[:-1]`，而标签可视作`S[1:]`。
如果从这个联系出发，一个很自然的想法就是我们并不需要额外优化一个完全独立于虚拟特征的虚拟标签，
而只须复用上述虚拟特征的相应部分。

不过这种思路是有点问题的。
上述虚拟特征是一个句子每个 token 的词向量，而不是每个 token 的 logit 值。
最重要的是，前者也无法转换为后者
其一，概率信息已经丢失——每个词向量对应唯一的 token，
其最多能够等价于一个在 token 对应的分类索引值无限大，而在其余类别位置无穷小的 logit。
因而直接复用虚拟特征的相应部分作为虚拟标签，在类别的语义空间上做不到连续。
其二，词向量实际上转换为唯一对应的 token 时避不开`argmax`或`argmin`函数。
因为这个过程一般是将待转换的词向量与词表里的所有 token 的词向量进行比对，选择某种度量空间下最相近的 token。
这个「最」就是`argmax`或`argmin`函数。
这类函数是不可导的，因而就算是强行转换了，后面也无法使用梯度下降这类常见算法优化。

从而，尽管自回归文本生成有其特殊性，虚拟标签仍需要单独于虚拟特征初始化，正如普通的分类任务一般。
基于“初始化为随机的类别 logit 值”的通用思想，应当初始化为随机的 token logit 值。


确认以上设置后，复现 TAG 攻击过程比较简单，
因为其本身只是 DLG 的延伸，其中比较重要的目标函数以及优化器等超参数的确定，
原文也有到位的描述。


## 3 杂项

时间线：

|  3月5日   | 3月6日至10日 |    3月11日     | 3月12日至今  |
|:-------:|:--------:|:------------:|:--------:|
| 读文章、搭框架 |  调试 TAG  | 调通 TAG 并撰写文档 |    待定    |

主要依赖资源：

一张 NVIDIA GeForce RTX 3090。


















