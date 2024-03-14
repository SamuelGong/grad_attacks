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
3. [APRIL 复现结果和难点讨论](#3-april-复现结果和难点讨论)
   - [第一种攻击 vs 第二种攻击？](#31-第一种攻击-vs-第二种攻击)
   - [结果概览（GPT-2 自回归文本生成）](#32-结果概览gpt-2-自回归文本生成)
   - [要点讨论](#33-要点讨论)
4. [杂项](#4-杂项)

## 1 概述

本仓库主要为作者自习用，另公开以促进技术交流。
建仓缘由是看到如下两篇和文本领域相关的梯度泄露攻击论文，于是想在自己搭建的虚拟联邦学习（FL）靶子复现以学习其技术：

1. [TAG: Gradient Attack on Transformer-based Language Models (EMNLP '21)](https://arxiv.org/abs/2103.06819)
   - 该论文提出了 TAG，据称是第一个针对 NLP 领域对梯度泄露攻击的系统性评估工作。
   攻击上，TAG 的技术路线直接继承了其祖师爷 [Deep Leakage from Gradients (NeurIPS '19)](https://arxiv.org/abs/1906.08935) (简称 DLG)
   的**梯度匹配**（Gradient Matching）思想——优化虚拟数据，通过令梯度逼近真实梯度来让虚拟数据逼近真实数据。
   技术上看，TAG（唯一）重要的创新点在于其**目标函数的定义**：(i) 除常规的梯度差的 L2 范数外，额外引入了 L1 范数，
   且 (ii) 该项的系数引入了变化的思想（根据参数所在的层数）。
   作者相信这样做是 TAG 相比以往工作更能应对各种（i）模型初始化场景（不同方式的随机初始化/从预训练参数初始化），(ii) 各种数据集，以及 (iii) 各种模型的原因。
2. [APRIL: Finding the Achilles' Heel on Privacy for Vision Transformers (CVPR '22)](https://arxiv.org/abs/2112.14087)
   - 该论文提出 APRIL，实质上是针对基于 transformer 的模型的两种梯度攻击方法。
   第一种是**梯度分析**攻击，在三个强假设下，提出一种能通过梯度反推输入的解析解的方法。
   其主要技术贡献在于提出可利用“可训练位置嵌入层”的梯度，间接推知 transformer 输入的梯度，从而推动整个解析过程。
   第二种是**梯度匹配**攻击，其拿掉了上述三个假设之二，另与上述技术贡献关系不大。
   作者在攻击图像分类任务上和 TAG 进行了对比，结果显示相比 TAG 恢复效果有提升。

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

另外，确认这个思路后，工程实现上对应地、严格来说应该作出的适应有：
* 【模型结构的改变】词向量不是语言模型输入的默认格式，要让目标模型额外接收这种格式，通常需要改变模型结构。
大概地，这需要有函数负责拿到模型中进行词嵌入的参数，用以生成词向量后，再送入正确的层。

  > 幸运地，本次实验针对的 GPT-2 具有很强的扩展性，其原本的设计就直接支持文本嵌入作为输入（通过参数`inputs_embeds`），
不过要获得并利用这个观察，也需要对其[源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py )有足够了解。

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

#### 2.2.3 梯度截断加速迭代

有了上述设定，论文中整个攻击框架几近能百分百复现了。
但其实有一个工程实现上的 trick 并没有特别写在纸面上，而它对成功复现论文的攻击效率几乎起决定性作用。
起初，我没有意识到这个问题，因而攻击的损失函数降低的速率一直很慢。
苦苦调参良久，也无从解决。
后来仔细比对参考仓库的代码，终于发现一个（起码在 TAG 这篇论文）根本没有写明的 trick：进行**梯度截断**。
具体地，参考仓库对虚拟特征和虚拟标签的梯度均进行了检查：
如果它们梯度的二范数大于某个常量（示例配置中设置为`1.0`），就对其中元素进行等比例缩小，使新梯度的二范数刚好等于该常量。
另这个过程特意引入了`with torch.no_grad()`，防止对原本的优化过程带来副作用。
不做这个的话，攻击速度非常慢。
在上述同等设置下，1000 轮迭代的攻击最终只能恢复出如下句子，其中 token 准确率为 0.0625，rougeL 为 0.1481：

```
 The Operation Tower preserves c historicUrban ninja faded photos promote primal bi boiling Hansendon
```

相应的训练过程参见 [grad_attacks/1_gpt2_gen/wikitext_2_tag_s42_it1000_lr005_no_grad_clip_20240312-210641.log](grad_attacks/1_gpt2_gen/wikitext_2_tag_s42_it1000_lr005_no_grad_clip_20240312-210641.log).



## 3 APRIL 复现结果和难点讨论

### 3.1 第一种攻击 vs 第二种攻击？

如上所述，APRIL 实际上包含了两种攻击。
先看第一种基于分析方法的攻击。
具体地，它能够精准恢复输入的特征，但需要作以下三个假设：

1. 被攻击的数据的标签是可知的。
2. 目标模型是基于 transformer 的，且最靠近输入的 transformer 层里的第一个子层应当是 attention 子层。
3. 输入特征到达该 transformer 层前，会先分别经过词嵌入和位置嵌入层，而位置嵌入层是可以被训练的。

而我的复现目标模型选定为 GPT-2 模型，以上三个假设至少有两个一定不成立。
这**限制了我复现该梯度分析攻击的可能性**。

首先，假设一不成立，数据的标签在自回归文本生成任务中是不可知的。
在普通单标签的分类任务中，由于最后一层一般为`Softmax`，连接它的最后一个全连接层的梯度的数字特征的确能够准确反应输入数据的标签。
具体地，在该层中，对应正确类别的权重梯度为一个符号，其余类别对应权重的梯度为另一个符号。
在这种情况下，这个假设很容易便成立了。
然而如果我们限定目标模型的训练任务为自回归文本生成，那就是另一个故事了。
如上所述，该任务下，GPT-2 的数据实际为多标签（回忆一下，也即如果输入特征为`S[:-1]`，则对应标签为`S[1:]`）。
从而，如果假设 1 真的能够成立，其实都不需要后续的攻击，因为数据的标签和数据的特征几乎是一个东西——知道了标签也就直接知道了特征。
实际上，只有在`S`的长度等于2的情况下（即目标模型学习用第一个 token 预测第二个 token），标签才能直接从梯度的观察中精准推知（因为此时已经退化到单标签分类任务）。

同时，假设二不成立，GPT-2 的结构并不符合这种假设。
实际上，GPT-2 几乎是首创地把 LayerNorm 子层搬到了每个 transformer 层里的最前面，而后面 GPT-3 等模型都依照了这个设计。
这里可以参见[原论文的描述](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)。

我只得再看第二种基于梯度匹配的攻击方法。
第二种方法和 TAG 类似，也是 DLG 攻击的直接衍生物。
孤立地看，APRIL 相比 DLG 在技术上就是更新了一下攻击的目标函数——
如果说 TAG 的目标函数在 DLG 的梯度差二范数的基础上是加上了带有依层数变化权重的梯度差的一范数的话，
那么第二种方法则是加上了带有固定权重的位置嵌入层梯度差的二范数。
如此强调位置嵌入层，缘于作者在设计第一种攻击时的观察——
但这种联系在论文中没有讲得很透彻，显得多少有些牵强。

相比第一种方法，第二种方法大大放松了对攻击场景的假设。
上述三个假设只需要成立最后一个，因而具有更强的通用性，能够直接支持 GPT-2 （自回归文本生成）。
因此余下部分专注于第二种方法（此后统一简称为 APRIL）。
值得注意的是，参考仓库对于 APRIL 只实现了第一种方法，因而接下来的复现实验没有现存代码参考。

### 3.2 结果概览（GPT-2 自回归文本生成）

这里针对的场景和第二节的类似，且示例输入同样为下面的一个 16 tokens 的序列：

```
 The Tower Building of the Little Rock Arsenal, also known as U.S.
```

以不同的攻击迭代轮数分别进行一次实验，参考结果为：

1. 迭代 1000 轮（约 163 秒）：token 准确率 0.25，RougeL 0.4444，具体恢复文本如下

```
 The Tower Building Building the unsc Little Tower Rock aspiring specializes worried spokesman HY semicfet
```


2. 迭代 20000 轮（约 3065 秒）：token 准确率 0.375，RougeL 0.6923，具体恢复文本如下

```
 The Tower Building of the continent Little Maj Rock also Arsenal also ofJudassebridge
```

上述攻击过程对应复现命令分别为

```bash
python main.py 1_gpt2_gen/wikitext_2_april_s42_it1000_lr005_s2.yml
python main.py 1_gpt2_gen/wikitext_2_april_s42_it20000_lr005_s2.yml
```

程序将在前台运行直至结束，期间往文件写日志。上述实验参考的日志文件可见于

* [1_gpt2_gen/wikitext_2_april_s42_it1000_lr005_s2_20240313-001440.log](1_gpt2_gen/wikitext_2_april_s42_it1000_lr005_s2_20240313-001440.log)
* [1_gpt2_gen/wikitext_2_april_s42_it20000_lr005_s2_20240313-224857.log](1_gpt2_gen/wikitext_2_april_s42_it20000_lr005_s2_20240313-224857.log)

### 3.3 要点讨论

#### 3.3.1 主超参数的设置

复现 APRIL 的梯度匹配攻击，需要注意的地方大体上和 TAG 一致（参见[2.2 要点讨论](#22-要点讨论)）。
需要额外注意的点是，攻击目标函数中位置嵌入梯度差的二范式的系数`alpha`，原文中并没有显式的设置指引。
为了探寻合适的设置，这里做了一个粗粒度的超参数搜索。
<<<<<<< HEAD
下表对比了使用不同`alpha`而其余设定相同的情况下，攻击结束时的损失函数值：
=======
下表对比了使用不同`alpha`而其余设定相同的情况下，攻击结束时的恢复数据的一些指标：
>>>>>>> 7bb31be7ef402934c24bdcb90b94fa3e89c7fe72

|  `alpha`  |       50       |   20    |   10   |   5    |     2      |     1      |   0.5   |  0.2   |  0.1   |   0    |
|:---:|:--------------:|:-------:|:------:|:------:|:----------:|:----------:|:-------:|:------:|:------:|:------:|
| accuracy |   **0.3125**   |  0.25   |  0.25  | 0.25  |   0.25    |     0.1875 |  0.125  | 0.125  |   0.125   |  0.25  |
| rougeL | 0.3077 |  0.3846 |  0.4138  | 0.3846 | 0.4286 | **0.4444** | 0.2963 | 0.2963 |  0.32  | 0.3704 |

以上结果仅针对一个特定句子的攻击，在这个情况下，恢复质量并不是太敏感于超参数`alpha`的设置。
（虽然取值 2-50 似乎能够获得比较好的综合效果，因之 [3.2 结果概览](#32-结果概览gpt-2-自回归文本生成) 展现的结果是基于`alpha`取值`2`）
特别地，`alpha`等于零时 APRIL 方法其实退化到了普通的 DLG。
但可见在此情况下，恢复效果也没有很差（比起 APRIL 里`alpha`取现存的最优，甚至比起上面的 TAG 算法）。
这或许意味着，APRIL 和 TAG 对攻击自回归文本生成的 GPT-2 或许都是牛刀杀鸡。
大道至简，DLG （即攻击目标函数是单纯的梯度差的二范数）已经足矣。
不过从另一个角度出发，这或许也意味着**上述表现已经是基于梯度匹配的攻击在当前场景的效率上限**，
如要再行提高，APRIL 和 TAG 这些小修小补都无能为力，只得依据更深入的观察，寻求更高阶的做法。

p.s. 上述结果相关日志可分别见于

```
1_gpt2_gen/wikitext_2_april_s42_it1000_lr005_s50_20240313-002315.log
1_gpt2_gen/wikitext_2_april_s42_it1000_lr005_s20_20240313-002027.log
1_gpt2_gen/wikitext_2_april_s42_it1000_lr005_s10_20240313-001727.log
1_gpt2_gen/wikitext_2_april_s42_it1000_lr005_s5_20240312-222853.log
1_gpt2_gen/wikitext_2_april_s42_it1000_lr005_s2_20240313-001440.log
1_gpt2_gen/wikitext_2_april_s42_it1000_lr005_s1_20240312-223149.log
1_gpt2_gen/wikitext_2_april_s42_it1000_lr005_s05_20240312-223442.log
1_gpt2_gen/wikitext_2_april_s42_it1000_lr005_s02_20240313-001149.log
1_gpt2_gen/wikitext_2_april_s42_it1000_lr005_20240312-214228.log
1_gpt2_gen/wikitext_2_april_s42_it1000_lr005_s0_20240312-235738.log
```

## 4 杂项

时间线：

|  3月5日   | 3月6日至10日 |    3月11日     |   3月12日至14日    |
|:-------:|:--------:|:------------:|:--------------:|
| 读文章、搭框架 |  调试 TAG  | 调通 TAG 并撰写文档 | 调通 APRIL 并撰写文档 |

主要依赖资源：

一张 NVIDIA GeForce RTX 3090。
