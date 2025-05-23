# DPO在代码优化任务上性能糟糕的原因分析

Direct Preference Optimization（DPO）是一种基于人类偏好数据直接优化语言模型的方法。尽管DPO在许多自然语言处理任务中表现出色，但在代码优化任务上，其性能可能显著下降。以下是DPO在代码优化任务上表现不佳的主要原因，以及相关的理论分析。

## 1. DPO倾向于降低人类不喜欢的回答的输出概率

### 1.1 原理回顾

在DPO的训练过程中，模型通过最大化好回答相对于差回答的偏好分数来优化：

$$
\mathcal{L}_\text{DPO} = -\log(\sigma(r(x,y_w,y_l)))
$$

其中，偏好分数 $r(x,y_w,y_l)$ 定义为：

$$
r(x,y_w,y_l) = \log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}
$$

优化目标是通过最小化损失 $\mathcal{L}_\text{DPO}$ 来增大模型对好回答 $y_w$ 的偏好，同时降低对差回答 $y_l$ 的偏好。这导致模型倾向于提升 $y_w$ 的概率，同时抑制 $y_l$ 的概率。

### 1.2 在代码优化任务中的表现

在代码优化任务中，输出的代码通常具有多种变体，而这些变体之间可能只有细微的差别（即编辑距离小）。然而，DPO在优化过程中不仅降低了差回答的概率，还可能错误地降低了一些好回答的概率，尤其是在这些好回答与差回答在编辑距离上非常接近时。

#### 例子：

假设有两个代码片段：

- **好回答 $y_w$**: 优化后的代码，其性能略优于参考代码。
- **差回答 $y_l$**: 性能稍差的代码。

由于好回答和差回答之间的编辑距离很小，DPO在优化时可能会误判某些微小变动也降低了好回答的概率，因为这些变动在训练数据中被标记为差回答。这种情况下，模型不仅未能有效提升好回答的概率，反而可能削弱其生成能力。

## 2. 特别在编辑距离小的数据上DPO效果尤为糟糕

### 2.1 编辑距离与概率分布

**编辑距离**（Edit Distance）衡量的是两个序列（在此为代码片段）之间的最小编辑操作次数，包括插入、删除和替换操作。对于代码优化任务，优良代码与次优代码之间的编辑距离通常较小，因为优化往往涉及细微的改动，如变量重命名、算法微调等。

### 2.2 DPO的局限性

DPO通过比较整个序列的概率来优化模型。然而，当编辑距离很小时，多个代码片段可能在模型的潜在空间中非常接近，导致：

1. **梯度信号稀释**：微小的差异可能不足以提供有效的梯度信号，尤其是当差回答的概率被频繁调整时。
2. **错误普及**：模型可能将微小修改的代码片段归类为差回答，进而整体降低了这些区域的概率，包括可能的好回答。

### 2.3 理论分析
考虑两个代码片段 $y_1$ 和 $y_2$，它们之间仅相差一个 token（即编辑距离为 1），并且 $y_1$ 被标记为好回答，而 $y_2$ 被标记为差回答。
DPO 优化目标试图满足以下不等式：
$$
\log \pi_\theta(y_1|x) - \log \pi_\text{ref}(y_1|x) > \log \pi_\theta(y_2|x) - \log \pi_\text{ref}(y_2|x)
$$
由于 $y_1$ 和 $y_2$ 仅有一个 token 的不同，模型在调整这些概率时，可能需要同时调整多个相关 token 的概率。在这种情况下，由于训练信号的稀释，模型可能无法准确区分微小的差异。这可能导致 $\pi_\theta(y_1|x)$ 的调整不够理想，甚至低于预期。



## 编辑距离（Edit Distance）解释

**编辑距离**是衡量两个序列（例如字符串或代码片段）之间差异的一种指标，表示将一个序列转换为另一个序列所需的最少编辑操作次数。常见的编辑操作包括：

1. **插入**（Insertion）：在序列中添加一个元素。
2. **删除**（Deletion）：从序列中移除一个元素。
3. **替换**（Substitution）：将序列中的一个元素替换为另一个元素。

例如，考虑两个字符串 "kitten" 和 "sitting"：

- 将 "k" 替换为 "s"（替换）
- 将 "e" 替换为 "i"（替换）
- 在末尾添加 "g"（插入）

因此，这两个字符串的编辑距离为 3。




<iframe src="/AIDIY/RLHF_Pages/DPO_Problem_viz.html" width="100%" height="600px" style="border: 1px solid #ccc;" title="DPO Problem Interactive Content">
    您的浏览器不支持 iframe，无法加载交互式内容。
    请 <a href="/AIDIY/RLHF_Pages/DPO_Problem_viz.html" target="_blank">点击这里在新窗口中查看</a>。
</iframe>


<script src="https://giscus.app/client.js"
        data-repo="InuyashaYang/AIDIY"
        data-repo-id="R_kgDOM1VVTQ"
        data-category="Announcements"
        data-category-id="DIC_kwDOM1VVTc4Ckls_"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="bottom"
        data-theme="preferred_color_scheme"
        data-lang="zh-CN"
        crossorigin="anonymous"
        async>
</script>
