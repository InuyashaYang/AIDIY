# RAG七十二式：2024年度RAG清单

引用自 **作者：** 范志东，蚂蚁图计算开源负责人、图计算布道师

**GitHub地址：** [Awesome-RAG](https://github.com/awesome-rag/awesome-rag)

---

## 2024.01

### 1. GraphReader【图解专家】

**简介：** GraphReader是一种基于图的智能体系统，旨在通过将长文本构建成图并使用智能体自主探索该图来处理长文本。在接收到问题后，智能体首先进行逐步分析并制定合理的计划。然后，它调用一组预定义的函数来读取节点内容和邻居，促进对图进行从粗到细的探索。在整个探索过程中，智能体不断记录新的见解并反思当前情况以优化过程，直到它收集到足够的信息来生成答案。

- **时间：** 01.20
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/eg3zIZ_3yhiJK83aTvQE2g)

### 2. MM-RAG【多面手】

**简介：** 多面手：就像一个能同时精通视觉、听觉和语言的全能选手，不仅能理解不同形式的信息，还能在它们之间自如切换和关联。通过对各种信息的综合理解，它能在推荐、助手、媒体等多个领域提供更智能、更自然的服务。

- **时间：** 01.22
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/wGar-qBfvjdi5juO1c0YxQ)

### 3. CRAG【自我校正】

**简介：** CRAG通过设计轻量级的检索评估器和引入大规模网络搜索，来改进检索文档的质量，并通过分解再重组算法进一步提炼检索到的信息，从而提升生成文本的准确性和可靠性。CRAG是对现有RAG技术的有益补充和改进，它通过自我校正检索结果，增强了生成文本的鲁棒性。

- **时间：** 01.29
- **论文：** [Corrective Retrieval Augmented Generation](链接)
- **项目：** [CRAG GitHub](https://github.com/HuskyInSalt/CRAG)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/HTN66ca6OTF_2YcW0Mmsbw)

### 4. RAPTOR【分层归纳】

**简介：** RAPTOR（Recursive Abstractive Processing for Tree-Organized Retrieval）引入了一种新方法，即递归嵌入、聚类和总结文本块，从下往上构建具有不同总结级别的树。在推理时，RAPTOR 模型从这棵树中检索，整合不同抽象级别的长文档中的信息。

- **时间：** 01.31
- **论文：** [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](链接)
- **项目：** [RAPTOR GitHub](https://github.com/parthsarthi03/raptor)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/8kt5qbHeTP1_ELY_YKwonA)

---

## 2024.02

### 5. T-RAG【私人顾问】

**简介：** T-RAG（树状检索增强生成）结合RAG与微调的开源LLM，使用树结构来表示组织内的实体层次结构增强上下文，利用本地托管的开源模型来解决数据隐私问题，同时解决推理延迟、令牌使用成本以及区域和地理可用性问题。

- **时间：** 02.12
- **论文：** [T-RAG: Lessons from the LLM Trenches](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/ytEkDAuxK1tLecbFxa6vfA)

---

## 2024.03

### 6. RAT【思考者】

**简介：** RAT（检索增强思维）在生成初始零样本思维链（CoT）后，利用与任务查询、当前和过去思维步骤相关的检索信息逐个修订每个思维步骤，RAT可显著提高各种长时生成任务上的性能。

- **时间：** 03.08
- **论文：** [RAT: Retrieval Augmented Thoughts Elicit Context-Aware Reasoning in Long-Horizon Generation](链接)
- **项目：** [RAT GitHub](https://github.com/CraftJarvis/RAT)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/TqmY4ouDuloE2v-iSJB_-Q)

### 7. RAFT【开卷高手】

**简介：** RAFT旨在提高模型在特定领域内的“开卷”环境中回答问题的能力，通过训练模型忽略无关文档，并逐字引用相关文档中的正确序列来回答问题，结合思维链式响应，显著提升了模型的推理能力。

- **时间：** 03.15
- **论文：** [RAFT: Adapting Language Model to Domain Specific RAG](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/PPaviBpF8hdviqml3kPGIQ)

### 8. Adaptive-RAG【因材施教】

**简介：** Adaptive-RAG根据查询的复杂程度动态选择最适合的检索增强策略，从最简单到最复杂的策略中动态地为LLM选择最合适的策略。这个选择过程通过一个小语言模型分类器来实现，预测查询的复杂性并自动收集标签以优化选择过程。这种方法提供了一种平衡的策略，能够在迭代式和单步检索增强型 LLMs 以及无检索方法之间无缝适应，以应对一系列查询复杂度。

- **时间：** 03.21
- **论文：** [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](链接)
- **项目：** [Adaptive-RAG GitHub](https://github.com/starsuzi/Adaptive-RAG)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/sxAu8xahY-GthS--nfgmjg)

### 9. HippoRAG【海马体】

**简介：** HippoRAG是一种新颖的检索框架，其灵感来源于人类长期记忆的海马体索引理论，旨在实现对新经验更深入、更高效的知识整合。HippoRAG协同编排 LLMs、知识图谱和个性化PageRank算法，以模拟新皮层和海马体在人类记忆中的不同角色。

- **时间：** 03.23
- **论文：** [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](链接)
- **项目：** [HippoRAG GitHub](https://github.com/OSU-NLP-Group/HippoRAG)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/zhDw4SxX1UpnczEC3XyHGA)

### 10. RAE【智能编辑】

**简介：** RAE（多跳问答检索增强模型编辑框架）首先检索经过编辑的事实，然后通过上下文学习来优化语言模型。基于互信息最大化的检索方法利用大型语言模型的推理能力来识别传统基于相似性的搜索可能会错过的链式事实。此外框架包括一种修剪策略，以从检索到的事实中消除冗余信息，这提高了编辑准确性并减轻了幻觉问题。

- **时间：** 03.28
- **论文：** [Retrieval-enhanced Knowledge Editing in Language Models for Multi-Hop Question Answering](链接)
- **项目：** [RAE GitHub](https://github.com/sycny/RAE)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/R0N8yexAlXetFyCS-W2dvg)

---

## 2024.04

### 11. RAGCache【仓储员】

**简介：** RAGCache是一种为RAG量身定制的新型多级动态缓存系统，它将检索到的知识的中间状态组织在知识树中，并在GPU和主机内存层次结构中进行缓存。RAGCache提出了一种考虑到LLM推理特征和RAG检索模式的替换策略。它还动态地重叠检索和推理步骤，以最小化端到端延迟。

- **时间：** 04.18
- **论文：** [RAGCache: Efficient Knowledge Caching for Retrieval-Augmented Generation](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/EOf51zoycmUCKkIo8rPsZw)

### 12. GraphRAG【社区摘要】

**简介：** GraphRAG分两个阶段构建基于图的文本索引：首先从源文档中推导出实体知识图，然后为所有紧密相关实体的组预生成社区摘要。给定一个问题，每个社区摘要用于生成部分响应，然后在向用户的最终响应中再次总结所有部分响应。

- **时间：** 04.24
- **论文：** [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](链接)
- **项目：** [GraphRAG GitHub](https://github.com/microsoft/graphrag)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/I_-rpMNVoQz-KvUlgQH-2w)

---

## 2024.05

### 13. R4【编排大师】

**简介：** R4（Reinforced Retriever-Reorder-Responder）用于为检索增强型大语言模型学习文档排序，从而在大语言模型的大量参数保持冻结的情况下进一步增强其生成能力。重排序学习过程根据生成响应的质量分为两个步骤：文档顺序调整和文档表示增强。具体来说，文档顺序调整旨在基于图注意力学习将检索到的文档排序组织到开头、中间和结尾位置，以最大化响应质量的强化奖励。文档表示增强通过文档级梯度对抗学习进一步细化质量较差的响应的检索文档表示。

- **时间：** 05.04
- **论文：** [R4: Reinforced Retriever-Reorder-Responder for Retrieval-Augmented Large Language Models](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/Lsom93jtIr4Pv7DjpQuiDQ)

### 14. IM-RAG【自言自语】

**简介：** IM-RAG通过学习内部独白（Inner Monologues）来连接IR系统与LLMs，从而支持多轮检索增强生成。该方法将信息检索系统与大型语言模型相整合，通过学习内心独白来支持多轮检索增强生成。在内心独白过程中，大型语言模型充当核心推理模型，它既可以通过检索器提出查询以收集更多信息，也可以基于对话上下文提供最终答案。我们还引入了一个优化器，它能对检索器的输出进行改进，有效地弥合推理器与能力各异的信息检索模块之间的差距，并促进多轮通信。整个内心独白过程通过强化学习（RL）进行优化，在此过程中还引入了一个进展跟踪器来提供中间步骤奖励，并且答案预测会通过监督微调（SFT）进一步单独优化。

- **时间：** 05.15
- **论文：** [IM-RAG: Multi-Round Retrieval-Augmented Generation Through Learning Inner Monologues](链接)
- **项目：** [IM-RAG GitHub](https://github.com/Cinnamon/kotaemon)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/O6cNeBAT5f_nQM5hRaQUnw)

### 15. AntGroup-GraphRAG【百家之长】

**简介：** 蚂蚁TuGraph团队基于DB-GPT构建的开源GraphRAG框架，兼容了向量、图谱、全文等多种知识库索引底座，支持低成本的知识抽取、文档结构图谱、图社区摘要与混合检索以解决QFS问答问题。另外也提供了关键词、向量和自然语言等多样化的检索能力支持。

- **时间：** 05.16
- **项目：** [DB-GPT GitHub](https://github.com/eosphoros-ai/DB-GPT)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/LfhAY91JejRm_A6sY6akNA)

### 16. Kotaemon【乐高】

**简介：** 一个开源的干净且可定制的RAG UI，用于构建和定制自己的文档问答系统。既考虑了最终用户的需求，也考虑了开发者的需求。

- **时间：** 05.15
- **项目：** [Kotaemon GitHub](https://github.com/Cinnamon/kotaemon)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/SzoE2Hb82a6yUU7EcfF5Hg)

### 17. FlashRAG【百宝箱】

**简介：** FlashRAG是一个高效且模块化的开源工具包，旨在帮助研究人员在统一框架内重现现有的RAG方法并开发他们自己的RAG算法。我们的工具包实现了12种先进的RAG方法，并收集和整理了32个基准数据集。

- **时间：** 05.22
- **论文：** [FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research](链接)
- **项目：** [FlashRAG GitHub](https://github.com/RUC-NLPIR/FlashRAG)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/vvOdcARaU1LD6KgcdShhoA)

### 18. GRAG【侦探】

**简介：** 传统RAG模型在处理复杂的图结构数据时忽视了文本之间的联系和数据库的拓扑信息，从而导致了性能瓶颈。GRAG通过强调子图结构的重要性，显著提升了检索和生成过程的性能并降低幻觉。

- **时间：** 05.26
- **论文：** [GRAG: Graph Retrieval-Augmented Generation](链接)
- **项目：** [GRAG GitHub](https://github.com/HuieL/GRAG)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/xLVaFVr7rnYJq0WZLsFVMw)

### 19. Camel-GraphRAG【左右开弓】

**简介：** Camel-GraphRAG依托Mistral模型提供支持，从给定的内容中提取知识并构建知识结构，然后将这些信息存储在 Neo4j图数据库中。随后采用一种混合方法，将向量检索与知识图谱检索相结合，来查询和探索所存储的知识。

- **时间：** 05.27
- **项目：** [Camel GitHub](https://github.com/camel-ai/camel)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/DhnAd-k-CtdGFVrwGat90w)

### 20. G-RAG【串门神器】

**简介：** RAG 在处理文档与问题上下文的关系时仍存在挑战，当文档与问题的关联性不明显或仅包含部分信息时，模型可能无法有效利用这些文档。此外，如何合理推断文档之间的关联也是一个重要问题。G-RAG实现了RAG检索器和阅读器之间基于图神经网络（GNN）的重排器。该方法结合了文档之间的连接信息和语义信息（通过抽象语义表示图），为 RAG 提供了基于上下文的排序器。

- **时间：** 05.28
- **论文：** [Don't Forget to Connect! Improving RAG with Graph-based Reranking](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/e6sRpYFDTQ7w7ituIjyovQ)

### 21. LLM-Graph-Builder【搬运工】

**简介：** Neo4j开源的基于LLM提取知识图谱的生成器，可以把非结构化数据转换成Neo4j中的知识图谱。利用大模型从非结构化数据中提取节点、关系及其属性。

- **时间：** 05.29
- **项目：** [LLM-Graph-Builder GitHub](https://github.com/neo4j-labs/llm-graph-builder)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/9Jy11WH7UgrW37281XopiA)

---

## 2024.06

### 22. MRAG【八爪鱼】

**简介：** 现有的 RAG 解决方案并未专注于可能需要获取内容差异显著的多个文档的查询。此类查询经常出现，但具有挑战性，因为这些文档的嵌入在嵌入空间中可能相距较远，使得难以全部检索到它们。本文介绍了多头 RAG（MRAG），这是一种新颖的方案，旨在通过一个简单而强大的想法来填补这一空白：利用 Transformer 多头注意力层的激活，而非解码器层，作为获取多方面文档的键。其驱动动机是不同的注意力头可以学习捕获不同的数据方面。利用相应的激活会产生代表数据项和查询各个层面的嵌入，从而提高复杂查询的检索准确性。

- **时间：** 06.07
- **论文：** [Multi-Head RAG: Solving Multi-Aspect Problems with LLMs](链接)
- **项目：** [MRAG GitHub](https://github.com/spcl/MRAG)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/WFYnF5UDlmwYsWz_BMtIYA)

### 23. PlanRAG【战略家】

**简介：** PlanRAG研究如何利用大型语言模型解决复杂数据分析决策问题的方案，通过定义决策问答（Decision QA）任务，即根据决策问题Q、业务规则R和数据库D，确定最佳决策d。PlanRAG首先生成决策计划，然后检索器生成数据分析的查询。

- **时间：** 06.18
- **论文：** [PlanRAG: A Plan-then-Retrieval Augmented Generation for Generative Large Language Models as Decision Makers](链接)
- **项目：** [PlanRAG GitHub](https://github.com/myeon9h/PlanRAG)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/q3x2jOFFibyMXHA57sGx3w)

### 24. FoRAG【作家】

**简介：** FoRAG提出了一种新颖的大纲增强生成器，在第一阶段生成器使用大纲模板，根据用户查询和上下文草拟答案大纲，第二阶段基于生成的大纲扩展每个观点，构建最终答案。同时提出一种基于精心设计的双精细粒度RLHF框架的事实性优化方法，通过在事实性评估和奖励建模两个核心步骤中引入细粒度设计，提供了更密集的奖励信号。

- **时间：** 06.19
- **论文：** [FoRAG: Factuality-optimized Retrieval Augmented Generation for Web-enhanced Long-form Question Answering](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/7uqZ5U10Ec2Pa7akCLCJEA)

### 25. Multi-Meta-RAG【元筛选器】

**简介：** Multi-Meta-RAG使用数据库过滤和LLM提取的元数据来改进RAG从各种来源中选择与问题相关的相关文档。

- **时间：** 06.19
- **论文：** [Multi-Meta-RAG: Improving RAG for Multi-Hop Queries using Database Filtering with LLM-Extracted Metadata](链接)
- **项目：** [Multi-Meta-RAG GitHub](https://github.com/mxpoliakov/multi-meta-rag)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/Jf3qdFR-o_A4FXwmOOZ3pg)

## 2024.07

### 26. RankRAG【全能选手】

**简介：** RankRAG通过指令微调单一的LLM，使其同时具备上下文排名和答案生成的双重功能。通过在训练数据中加入少量排序数据，经过指令微调的大语言模型效果出奇地好，甚至超过了现有的专家排序模型，包括在大量排序数据上专门微调的相同大语言模型。这种设计不仅简化了传统RAG系统中多模型的复杂性，还通过共享模型参数增强了上下文的相关性判断和信息的利用效率。

- **时间：** 07.02
- **论文：** [RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/BZDXCTKSKLOwDv1j8_75_Q)

### 27. GraphRAG-Local-UI【改装师】

**简介：** GraphRAG-Local-UI是基于Microsoft的GraphRAG的本地模型适配版本，具有丰富的交互式用户界面生态系统。

- **时间：** 07.14
- **项目：** [GraphRAG-Local-UI GitHub](https://github.com/severian42/GraphRAG-Local-UI)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/DLvF7YpU3IfWvnu9ZyiBIA)

### 28. ThinkRAG【小秘书】

**简介：** ThinkRAG大模型检索增强生成系统，可以轻松部署在笔记本电脑上，实现本地知识库智能问答。

- **时间：** 07.15
- **项目：** [ThinkRAG GitHub](https://github.com/wzdavid/ThinkRAG)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/VmnVwDyi0i6qkBEzLZlERQ)

### 29. Nano-GraphRAG【轻装上阵】

**简介：** Nano-GraphRAG是一个更小、更快、更简洁的GraphRAG，同时保留了核心功能。

- **时间：** 07.25
- **项目：** [Nano-GraphRAG GitHub](https://github.com/gusye1234/nano-graphrag)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/pnyhz0jA4jgLndMUM9IU1g)

---

## 2024.08

### 30. RAGFlow-GraphRAG【导航员】

**简介：** RAGFlow借鉴了GraphRAG的实现，在文档预处理阶段，引入知识图谱构建作为可选项，服务于QFS问答场景，并引入了实体去重、Token优化等改进。

- **时间：** 08.02
- **项目：** [RAGFlow GitHub](https://github.com/infiniflow/ragflow)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/c5-0dCWI0bIa2zHagM1w0w)

### 31. Medical-Graph-RAG【数字医生】

**简介：** MedGraphRAG是一个框架，旨在解决在医学中应用LLM的挑战。它使用基于图谱的方法来提高诊断准确性、透明度并集成到临床工作流程中。该系统通过生成由可靠来源支持的响应来提高诊断准确性，解决了在大量医疗数据中维护上下文的困难。

- **时间：** 08.08
- **论文：** [Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation](链接)
- **项目：** [Medical-Graph-RAG GitHub](https://github.com/SuperMedIntel/Medical-Graph-RAG)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/5mX-hCyFdve98H01x153Eg)

### 32. HybridRAG【中医合方】

**简介：** 一种基于知识图谱RAG技术（GraphRAG）和VectorRAG技术相结合的新方法，称为HybridRAG，以增强从金融文档中提取信息的问答系统。该方法被证明能够生成准确且与上下文相关的答案。在检索和生成阶段，HybridRAG在检索准确性和答案生成方面优于传统的VectorRAG和GraphRAG。

- **时间：** 08.09
- **论文：** [HybridRAG: Integrating Knowledge Graphs and Vector Retrieval Augmented Generation for Efficient Information Extraction](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/59e_bEcxGkM4N0GeCTTF4w)

### 33. W-RAG【进化搜索】

**简介：** 开放域问答中的弱监督密集检索技术，利用大型语言模型的排序能力为训练密集检索器创建弱标注数据。通过评估大型语言模型基于问题和每个段落生成正确答案的概率，对通过BM25检索到的前K个段落进行重新排序。排名最高的段落随后被用作密集检索的正训练示例。

- **时间：** 08.15
- **论文：** [W-RAG: Weakly Supervised Dense Retrieval in RAG for Open-domain Question Answering](链接)
- **项目：** [W-RAG GitHub](https://github.com/jmnian/weak_label_for_rag)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/JqT1wteHC43h2cPlXmPOFg)

### 34. RAGChecker【质检员】

**简介：** RAGChecker的诊断工具为RAG系统提供细粒度、全面、可靠的诊断报告，并为进一步提升性能提供可操作的方向。它不仅能评估系统的整体表现，还能深入分析检索和生成两大核心模块的性能。

- **时间：** 08.15
- **论文：** [RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation](链接)
- **项目：** [RAGChecker GitHub](https://github.com/amazon-science/RAGChecker)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/x4o7BinnwvTsOa2_hegcrQ)

### 35. Meta-Knowledge-RAG【学者】

**简介：** Meta-Knowledge-RAG（MK Summary）引入了一种新颖的以数据为中心的RAG工作流程，将传统的“检索-读取”系统转变为更先进的“准备-重写-检索-读取”框架，以实现对知识库的更高领域专家级理解。我们的方法依赖于为每个文档生成元数据和合成的问题与答案，以及为基于元数据的文档集群引入元知识摘要的新概念。所提出的创新实现了个性化的用户查询增强和跨知识库的深度信息检索。

- **时间：** 08.16
- **论文：** [Meta Knowledge for Retrieval Augmented Large Language Models](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/twFVKQDTRZTGvDeYA8-c0A)

### 36. CommunityKG-RAG【社群探索】

**简介：** CommunityKG-RAG是一种新颖的零样本框架，它将知识图谱中的社区结构与RAG系统相结合，以增强事实核查过程。CommunityKG-RAG能够在无需额外训练的情况下适应新的领域和查询，它利用知识图谱中社区结构的多跳性质，显著提高信息检索的准确性和相关性。

- **时间：** 08.16
- **论文：** [CommunityKG-RAG: Leveraging Community Structures in Knowledge Graphs for Advanced Retrieval-Augmented Generation in Fact-Checking](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/ixKV-PKf8ohqZDCTN9jLZQ)

### 37. TC-RAG【记忆术士】

**简介：** 通过引入图灵完备的系统来管理状态变量，从而实现更高效、准确的知识检索。通过利用具有自适应检索、推理和规划能力的记忆堆栈系统，TC-RAG不仅确保了检索过程的受控停止，还通过Push和Pop操作减轻了错误知识的积累。

- **时间：** 08.17
- **论文：** [TC-RAG: Turing-Complete RAG's Case study on Medical LLM Systems](链接)
- **项目：** [TC-RAG GitHub](https://github.com/Artessay/TC-RAG)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/9VhIC5sJP_6nh_Ppfsb6UQ)

### 38. RAGLAB【竞技场】

**简介：** RAGLAB是一个模块化、研究导向的开源库，重现了6种算法并构建了全面的研究生态。借助RAGLAB，我们在10个基准上公平对比6种算法，助力研究人员高效评估和创新算法。

- **时间：** 08.21
- **论文：** [RAGLAB: A Modular and Research-Oriented Unified Framework for Retrieval-Augmented Generation](链接)
- **项目：** [RAGLab GitHub](https://github.com/fate-ubw/RAGLab)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/WSk0zdWZRXMVvm4-_HiFRw)

---

## 2024.09

### 39. MemoRAG【过目不忘】

**简介：** MemoRAG是一个创新的检索增强生成（RAG）框架，构建在一个高效的超长记忆模型之上。与主要处理具有明确信息需求查询的标准RAG不同，MemoRAG利用其记忆模型实现对整个数据库的全局理解。通过从记忆中召回特定于查询的线索，MemoRAG增强了证据检索，从而产生更准确且具有丰富上下文的响应生成。

- **时间：** 09.01
- **项目：** [MemoRAG GitHub](https://github.com/qhjqhj00/MemoRAG)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/88FTElcYf5PIgHN0J8R7DA)

### 40. OP-RAG【注意力管理】

**简介：** LLM中的极长语境会导致对相关信息的关注度降低，并导致答案质量的潜在下降。我们提出了一种顺序保留检索增强生成机制OP-RAG，显著提高了RAG在长上下文问答应用中的性能。

- **时间：** 09.03
- **论文：** [In Defense of RAG in the Era of Long-Context Language Models](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/WLaaniD7RRgN0h2OCztDcQ)

### 41. AgentRE【智能抽取】

**简介：** AgentRE通过整合大型语言模型的记忆、检索和反思能力，有效应对复杂场景关系抽取中关系类型多样以及单个句子中实体之间关系模糊的挑战。AgentRE包含三大模块，助力代理高效获取并处理信息，显著提升关系抽取性能。

- **时间：** 09.03
- **论文：** [AgentRE: An Agent-Based Framework for Navigating Complex Information Landscapes in Relation Extraction](链接)
- **项目：** [AgentRE GitHub](https://github.com/Lightblues/AgentRE)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/_P_3H3uyIWjgaCF_FczsDg)

## 2024.09

### 43. GraphInsight【图谱解读】

**简介：** GraphInsight旨在提升LLMs对宏观和微观层面图形信息理解的新框架。它基于两大关键策略：1）将关键图形信息置于LLMs记忆性能较强的位置；2）借鉴检索增强生成（RAG）的思想，对记忆性能较弱的区域引入轻量级外部知识库。此外，GraphInsight探索将这两种策略整合到LLM代理过程中，以应对需要多步推理的复杂图任务，使AI既能把握全局又不遗漏细节。

- **时间：** 09.05
- **论文：** [GraphInsight: Unlocking Insights in Large Language Models for Graph Structure Understanding](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/xDKTBtso3ONCGyskvmfAcg)

### 44. LA-RAG【方言通】

**简介：** LA-RAG是一种基于LLM的ASR的新型检索增强生成（RAG）范式。LA-RAG利用细粒度标记级语音数据存储和语音到语音检索机制，通过LLM的上下文学习(ICL)功能，提高语音识别(ASR)的准确性。它不仅能准确识别标准普通话，还能处理带有地方特色的口音，实现与不同地区用户的无障碍交流。

- **时间：** 09.13
- **论文：** [LA-RAG: Enhancing LLM-based ASR Accuracy with Retrieval-Augmented Generation](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/yrmtBqP4bmQ2wYZM7F24Yg)

### 45. SFR-RAG【精简检索】

**简介：** SFR-RAG是一个经过指令微调的小型语言模型，重点是基于上下文的生成和最小化幻觉。通过专注于在保持高性能的同时减少参数数量，SFR-RAG模型包含函数调用功能，使其能够与外部工具动态交互以检索高质量的上下文信息，确保回答既准确又高效。

- **时间：** 09.16
- **论文：** [SFR-RAG: Towards Contextually Faithful LLMs](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/rArOICbHpkmFPR5UoBIi5A)

### 46. FlexRAG【压缩专家】

**简介：** FlexRAG通过在LLM编码之前对检索到的上下文进行压缩，生成紧凑的嵌入。这些压缩后的嵌入经过优化以提升下游RAG的性能。FlexRAG的关键特性是其灵活性，能够支持不同的压缩比，并选择性地保留重要上下文。得益于这些技术设计，FlexRAG在显著降低运行成本的同时，实现了卓越的生成质量。

- **时间：** 09.24
- **论文：** [Lighter And Better: Towards Flexible Context Adaptation For Retrieval Augmented Generation](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/heYbLVQHeykD1EbqH8PSZw)

### 47. CoTKR【图谱翻译】

**简介：** CoTKR（Chain-of-Thought Enhanced Knowledge Rewriting）通过交替生成推理路径和相应知识，克服了单步知识改写的限制。为了弥合知识改写器和问答模型之间的偏好差异，提出了从问答反馈中对齐偏好的训练策略，利用QA模型的反馈进一步优化知识改写器。这一方法使得模型能够深入理解知识的来龙去脉，逐步讲解复杂问题。

- **时间：** 09.29
- **论文：** [CoTKR: Chain-of-Thought Enhanced Knowledge Rewriting for Complex Knowledge Graph Question Answering](链接)
- **项目：** [CoTKR GitHub](https://github.com/wuyike2000/CoTKR)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/lCHxLxRP96Y3mofDVjKY9w)

---

## 2024.10

### 48. Open-RAG【智囊团】

**简介：** Open-RAG通过开源大语言模型提高RAG中的推理能力，将任意密集的大语言模型转换为参数高效的稀疏专家混合（MoE）模型。该模型能够处理复杂的推理任务，包括单跳和多跳查询，独特地训练模型应对看似相关但具有误导性的干扰项，使其既能独立思考又能协同工作，还能分辨真假信息。

- **时间：** 10.02
- **论文：** [Open-RAG: Enhanced Retrieval-Augmented Reasoning with Open-Source Large Language Models](链接)
- **项目：** [Open-RAG GitHub](https://github.com/ShayekhBinIslam/openrag)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/H0_THczQ3UWCkSnnk-vveQ)

### 49. TableRAG【Excel专家】

**简介：** TableRAG专为表格理解设计了检索增强生成框架，通过查询扩展结合Schema和单元格检索，能够在提供信息给语言模型之前精准定位关键数据，从而实现更高效的数据编码和精确检索。它大幅缩短了提示长度，减少了信息丢失，使模型能够快速定位和提取所需的关键信息。

- **时间：** 10.07
- **论文：** [TableRAG: Million-Token Table Understanding with Language Models](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/n0iu6qOufc1izlzuRjQO6g)

### 50. LightRAG【蜘蛛侠】

**简介：** LightRAG将图结构融入文本索引和检索过程中，采用双层检索系统，从低级和高级知识发现中增强全面的信息检索。通过将图结构与向量表示相结合，便于高效检索相关实体及其关系，显著提高响应时间，同时保持上下文相关性。增量更新算法确保了新数据的及时整合，使系统能够在快速变化的数据环境中保持有效性和响应性。

- **时间：** 10.08
- **论文：** [LightRAG: Simple and Fast Retrieval-Augmented Generation](链接)
- **项目：** [LightRAG GitHub](https://github.com/HKUDS/LightRAG)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/1QKdgZMN55zD6X6xWSiTJw)

### 51. AstuteRAG【明智判官】

**简介：** Astute RAG通过适应性地从LLMs内部知识中提取信息，结合外部检索结果，并根据信息的可靠性来最终确定答案，从而提高系统的鲁棒性和可信度。它对外部信息保持警惕，不轻信检索结果，善用自身积累的知识，甄别信息真伪，像资深法官一样，权衡多方证据做出判断。

- **时间：** 10.09
- **论文：** [Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/Y8ozl3eH1osJTNOSuu4v1w)

### 52. TurboRAG【速记高手】

**简介：** TurboRAG通过离线预计算和存储文档的KV缓存来优化RAG系统的推理范式。与传统方法不同，TurboRAG在每次推理时不再计算这些KV缓存，而是检索预先计算的缓存以进行高效的预填充，从而消除了重复在线计算的需要。这种方法显著减少了计算开销，加快了响应时间，同时保持了准确性。

- **时间：** 10.10
- **论文：** [TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text](链接)
- **项目：** [TurboRAG GitHub](https://github.com/MooreThreads/TurboRAG)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/lanZ8cIEnIt12tFt4d-xzw)

### 53. StructRAG【收纳师】

**简介：** StructRAG受人类在处理知识密集型推理时将原始信息转换为结构化知识的认知理论启发，提出了一种混合信息结构化机制。该机制根据任务的特定要求以最合适的格式构建和利用结构化知识。通过模仿类人的思维过程，StructRAG提高了LLM在知识密集型推理任务上的表现。

- **时间：** 10.11
- **论文：** [StructRAG: Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Structurization](链接)
- **项目：** [StructRAG GitHub](https://github.com/Li-Z-Q/StructRAG)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/9UQOozHNHDRade5b6Onr6w)

### 54. VisRAG【火眼金睛】

**简介：** VisRAG通过构建基于视觉-语言模型(VLM)的RAG流程，直接将文档作为图像嵌入并检索，避免了解析过程中的信息损失，更全面地保留了原始文档的信息。实验显示，VisRAG在检索和生成阶段均超越传统RAG，端到端性能提升达25-39%。VisRAG不仅有效利用训练数据，还展现出强大的泛化能力，成为多模态文档RAG的理想选择。

- **时间：** 10.14
- **论文：** [VisRAG: Vision-based Retrieval-augmented Generation on Multi-modality Documents](链接)
- **项目：** [VisRAG GitHub](https://github.com/openbmb/visrag)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/WB23pwJD-JV95ZlpB3bUew)

### 55. AGENTiGraph【知识管家】

**简介：** AGENTiGraph是通过自然语言交互进行知识管理的平台。它集成了知识提取、整合和实时可视化。AGENTiGraph采用多智能体架构来动态解释用户意图、管理任务和集成新知识，确保能够适应不断变化的用户需求和数据上下文。就像一个善于对话的图书管理员，通过日常交流帮你整理和展示知识。

- **时间：** 10.15
- **论文：** [AGENTiGraph: An Interactive Knowledge Graph Platform for LLM-based Chatbots Utilizing Private Data](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/iAlcxjXHlz7xfwVd4lpQ-g)

## 2024.10

### 56. RuleRAG【循规蹈矩】

**简介：** RuleRAG提出了基于语言模型的规则引导检索增强生成方法。该方法明确引入符号规则作为上下文学习（RuleRAG - ICL）的示例，以引导检索器按照规则方向检索逻辑相关的文档，并统一引导生成器在同一组规则的指导下生成有依据的答案。此外，查询和规则的组合可进一步用作有监督的微调数据，用以更新检索器和生成器（RuleRAG - FT），从而实现更好的基于规则的指令遵循能力，进而检索到更具支持性的结果并生成更可接受的答案。就像带新人入职，先给本员工手册，不是漫无目的地学，而是像个严格的老师，先把规矩和范例都讲明白，然后再让学生自己动手。

- **时间：** 10.15
- **论文：** [RuleRAG: Rule-guided retrieval-augmented generation with language models for question answering](链接)
- **项目：** [RuleRAG_ICL_FT](https://github.com/chenzhongwu20/RuleRAG_ICL_FT)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/GNLvKG8ZgJzzNsWyVbSSig)

### 57. Class-RAG【法官】

**简介：** Class-RAG是一种内容审核分类器，对生成式AI的安全性至关重要。安全与不安全内容间的细微差别常令人难以区分。随着技术广泛应用，持续微调模型以应对风险变得愈发困难且昂贵。为此，Class-RAG方法通过动态更新检索库，实现即时风险缓解。与传统微调模型相比，Class-RAG更具灵活性与透明度，且在分类与抗攻击方面表现更佳。研究还表明，扩大检索库能有效提升审核性能，成本低廉。它就像经验老到的法官，手握活页法典，随时翻阅最新案例，让判决既有温度又有尺度。

- **时间：** 10.18
- **论文：** [Class-RAG: Content Moderation with Retrieval Augmented Generation](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/4AfZodMGJ5JQ2NUCFt3eqQ)

### 58. Self-RAG【反思者】

**简介：** Self-RAG通过检索和自我反思来提升语言模型的质量和准确性。该框架训练一个单一的语言模型，能够按需自适应地检索文档，并使用被称为“反思标记”的特殊标记来对检索到的文档及其自身生成的内容进行生成和反思。生成反思标记使得语言模型在推理阶段具备可控性，使其能够根据不同的任务要求调整自身行为。就像一个谨慎的学者，在回答问题时，不仅查阅资料，还不断思考和检查自己的答案是否准确完整。

- **时间：** 10.23
- **论文：** [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](链接)
- **项目：** [Self-RAG GitHub](https://github.com/AkariAsai/self-rag)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/y-hN17xFyODxzTIfEfm1Vg)

### 59. SimRAG【自学成才】

**简介：** SimRAG是一种自训练方法，使大型语言模型具备问答和问题生成的联合能力，以适应特定领域。模型首先在指令遵循、问答和搜索相关数据上进行微调，然后促使同一模型从无标签语料库中生成各种与领域相关的问题，并采用额外的过滤策略来保留高质量的合成示例。通过利用这些合成示例，模型可以提高其在特定领域的RAG任务性能。就像学生通过反复做习题来熟悉专业知识一样，模型通过自我提问和回答，不断提升专业知识储备。

- **时间：** 10.23
- **论文：** [SimRAG: Self-Improving Retrieval-Augmented Generation for Adapting Large Language Models to Specialized Domains](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/pR-W_bQEA4nM86YsVTThtA)

### 60. ChunkRAG【摘抄达人】

**简介：** ChunkRAG提出了基于LLM驱动的块过滤方法，通过在块级别评估和过滤检索到的信息来增强RAG系统。该方法采用语义分块将文档划分为连贯的部分，并利用基于大语言模型的相关性评分来评估每个块与用户查询的匹配程度。通过在生成阶段之前过滤掉不太相关的块，显著减少了幻觉并提高了事实准确性。就像一个摘抄达人，先把长文章分成小段落，再用专业眼光挑出最相关的片段，既不遗漏重点，又不被无关内容干扰。

- **时间：** 10.23
- **论文：** [ChunkRAG: Novel LLM-Chunk Filtering Method for RAG Systems](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/Pw7_vQ9bhdDFTmoVwxGCyg)

### 61. FastGraphRAG【雷达】

**简介：** FastGraphRAG提供了一个高效、可解释且精度高的快速图检索增强生成框架。它将PageRank算法应用于知识图谱的遍历过程，快速定位最相关的知识节点。通过计算节点的重要性得分，PageRank使GraphRAG能够更智能地筛选和排序知识图谱中的信息。这就像为GraphRAG装上了一个“重要性雷达”，能够在浩如烟海的数据中快速定位关键信息。

- **时间：** 10.23
- **项目：** [FastGraphRAG GitHub](https://github.com/circlemind-ai/fast-graphrag)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/uBcYaO5drTUabcCXh3bzjA)

### 62. AutoRAG【调音师】

**简介：** AutoRAG框架能够自动为给定数据集识别合适的RAG模块，并探索和逼近该数据集的RAG模块的最优组合。通过系统评估不同的RAG设置来优化技术选择，该框架类似于传统机器学习中的AutoML实践，通过广泛实验来优化RAG技术的选择，提高RAG系统的效率和可扩展性。就像一位经验丰富的调音师，自动尝试各种RAG组合，最终找到最和谐的“演奏方案”。

- **时间：** 10.28
- **论文：** [AutoRAG: Automated Framework for optimization of Retrieval Augmented Generation Pipeline](链接)
- **项目：** [AutoRAG_ARAGOG_Paper](https://github.com/Marker-Inc-Korea/AutoRAG_ARAGOG_Paper)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/96r6y3cNmLRL2Z0W78X1OQ)

### 63. Plan×RAG【项目经理】

**简介：** Plan×RAG是一个新颖的框架，将现有RAG框架的“检索-推理”范式扩充为“计划-检索-推理”范式。Plan×RAG将推理计划制定为有向无环图（DAG），将查询分解为相互关联的原子子查询。答案生成遵循DAG结构，通过并行检索和生成显著提高效率。虽然最先进的RAG解决方案需要大量的数据生成和语言模型的微调，Plan×RAG纳入了冻结的语言模型作为即插即用的专家来生成高质量的答案。它就像一个项目经理，先规划后行动，把大任务分解成小任务，安排多个“专家”并行工作。

- **时间：** 10.28
- **论文：** [Plan×RAG: Planning-guided Retrieval Augmented Generation](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/I_-NDGzd7d8l4zjRfCsvDQ)

### 64. SubgraphRAG【定位仪】

**简介：** SubgraphRAG扩展了基于知识图谱的RAG框架，通过检索子图并利用LLM进行推理和答案预测。将轻量级多层感知器与并行三元组评分机制相结合，实现高效灵活的子图检索，同时编码有向结构距离以提高检索有效性。检索到的子图大小可以灵活调整，以匹配查询需求和下游LLM的能力。这种设计在模型复杂性和推理能力之间取得了平衡，实现了可扩展且通用的检索过程。就像一个定位仪，不是漫无目的地大海捞针，而是精准绘制一张小型知识地图，让AI能快速找到答案。

- **时间：** 10.28
- **论文：** [Simple is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation](链接)
- **项目：** [SubgraphRAG GitHub](https://github.com/Graph-COM/SubgraphRAG)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/ns22XLKsABly7RjpSjQ_Fw)

---

## 2024.11

### 65. RuRAG【炼金术士】

**简介：** RuRAG旨在通过将大量离线数据自动蒸馏成可解释的一阶逻辑规则，并注入大型语言模型中，以提升其推理能力。该框架使用蒙特卡洛树搜索（MCTS）来发现逻辑规则，并将这些规则转化为自然语言，实现针对LLM下游任务的知识注入和无缝集成。该论文在公共和私有工业任务上评估了该框架的有效性，证明了其在多样化任务中增强LLM能力的潜力。就像一个炼金术士，能将海量数据提炼成清晰的逻辑规则，并用通俗易懂的语言表达出来。

- **时间：** 11.04
- **论文：** [RuAG: Learned-rule-augmented Generation for Large Language Models](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/A4vjN1eJr7hJd75UH0kuXA)

### 66. RAGViz【透视眼】

**简介：** RAGViz提供了对检索文档和模型注意力的可视化，帮助用户理解生成的标记与检索文档之间的交互，可用于诊断和可视化RAG系统。它让RAG系统变得透明，能够看见模型在读哪句话，哪里不对一目了然。

- **时间：** 11.04
- **论文：** [RAGViz: Diagnose and Visualize Retrieval-Augmented Generation](链接)
- **项目：** [RAGViz GitHub](https://github.com/cxcscmu/RAGViz)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/ZXvAWDhqKRPq1u9NTfYFnQ)

### 67. AgenticRAG【智能助手】

**简介：** AgenticRAG描述了基于AI智能体实现的RAG。它将AI智能体纳入RAG流程中，以协调其组件并执行超出简单信息检索和生成的额外行动，以克服非智能体流程的局限性。不再是简单的查找复制，而是配备了一个能当机要秘书的助手，像个得力的行政官，不光会查资料，还知道什么时候该采取额外的行动。

- **时间：** 11.05
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/Sa6vtb1pDKo1we1cSMn9oQ)

### 68. HtmlRAG【排版师】

**简介：** HtmlRAG在RAG中使用HTML而不是纯文本作为检索知识的格式。在对外部文档中的知识进行建模时，HTML比纯文本更好，大多数LLM具有强大的理解HTML的能力。HtmlRAG提出了HTML清理、压缩和修剪策略，以缩短HTML，同时最小化信息损失。它就像一个挑剔的美编，认为光有内容不够，还得讲究排版，这样重点才能一目了然。

- **时间：** 11.05
- **论文：** [HtmlRAG: HTML is Better Than Plain Text for Modeling Retrieved Knowledge in RAG Systems](链接)
- **项目：** [HtmlRAG GitHub](https://github.com/plageon/HtmlRAG)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/1X6k9QI71BIyQ4IELQxOlA)

### 69. M3DocRAG【感官达人】

**简介：** M3DocRAG是一种新颖的多模态RAG框架，能够灵活适应各种文档上下文（封闭域和开放域）、问题跳转（单跳和多跳）和证据模式（文本、图表、图形等）。M3DocRAG使用多模态检索器和多模态语言模型查找相关文档并回答问题，因此可以有效地处理单个或多个文档，同时保留视觉信息。就像一个全能选手，图片能看懂，文字能理解，各种挑战都难不倒。

- **时间：** 11.07
- **论文：** [M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding](链接)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/a9tDj6BmIZHs2vTFXKSPcA)

### 70. KAG【逻辑大师】

**简介：** KAG的设计目的是充分利用知识图谱和向量检索的优势，应对RAG中向量相似性与知识推理的相关性之间的差距，以及对知识逻辑（如数值、时间关系、专家规则等）不敏感的问题。KAG通过五个关键方面双向增强大型语言模型和知识图谱，提高生成和推理性能。就像一个严谨的数学老师，不仅要知道答案是什么，还得解释清楚答案是怎么一步步推导出来的。

- **时间：** 11.10
- **论文：** [KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation](链接)
- **项目：** [KAG GitHub](https://github.com/OpenSPG/KAG)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/oOzFBHS_B7FST6YKynD1GA)

### 71. FILCO【筛选师】

**简介：** FILCO通过基于词法和信息论的方法识别有用的上下文，以及训练上下文过滤模型，来过滤检索到的上下文，提高提供给生成器的上下文质量。它就像一个严谨的编辑，善于从大量文本中识别并保留最有价值的信息，确保传递给AI的每段内容都精准且相关。

- **时间：** 11.14
- **论文：** [Learning to Filter Context for Retrieval-Augmented Generation](链接)
- **项目：** [FILCO GitHub](https://github.com/zorazrw/filco)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/93CdvD8FLZjaA7E724bf7g)

### 72. LazyGraphRAG【精算师】

**简介：** LazyGraphRAG是一种新型的图谱增强检索增强生成（RAG）方法。该方法显著降低了索引和查询成本，同时在回答质量上保持或超越竞争对手，使其在多种用例中具有高度的可扩展性和高效性。LazyGraphRAG推迟了对LLM的使用，在索引阶段仅使用轻量级的NLP技术处理文本，将LLM的调用延迟到查询时。这种“懒惰”的策略避免了前期高昂的索引成本，实现了高效的资源利用。就像一个精算师，能省一步是一步，把贵的大模型用在刀刃上。

- **时间：** 11.25
- **项目：** [LazyGraphRAG GitHub](https://github.com/microsoft/graphrag)
- **参考：** [微信公众号链接](https://mp.weixin.qq.com/s/kDUcg5CzRcL7lTGllv-GKA)

