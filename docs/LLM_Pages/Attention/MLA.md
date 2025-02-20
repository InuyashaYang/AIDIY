

### **Stage 1: Query路径处理**
1. **低秩投影**  
   $h \xrightarrow{W_{\text{dq}} \in \mathbb{R}^{d_{\text{model}} \times r_q}} q_{\text{lowrank}} \xrightarrow{W_{\text{uq}} \in \mathbb{R}^{r_q \times (n_h \cdot d_h)}} q_{\text{restore}}$  
   - $r_q$: Query低秩维度（q_lora_rank）
   - $d_h = d_{\text{nope}} + d_{\text{rope}}$: 每个注意力头的总维度

2. **维度拆分**  
   $q_{\text{restore}} \longrightarrow \underset{\text{内容分量}}{\boxed{q^C \in \mathbb{R}^{n_h \times d_{\text{nope}}}}} \oplus \underset{\text{位置分量}}{\boxed{q^R \in \mathbb{R}^{n_h \times d_{\text{rope}}}}}$

---

### **Stage 2: Key-Value路径处理**
1. **联合投影**  
   $h \xrightarrow{W_{kv_a} \in \mathbb{R}^{d_{\text{model}} \times (d_c + d_{\text{rope}})}} \text{compressed\_kv\_full}$

2. **首次拆分**  
   $\text{compressed\_kv\_full} \longrightarrow \underset{\text{压缩KV}}{\boxed{c^{kv} \in \mathbb{R}^{d_c}}} \oplus \underset{\text{位置键}}{\boxed{k^{R} \in \mathbb{R}^{d_{\text{rope}}}}}$

3. **低秩重建**  
   $c^{kv} \xrightarrow{W_{kv_b} \in \mathbb{R}^{d_c \times n_h \cdot (d_{\text{nope}} + d_v)}} \text{rebuilt\_kv}$

4. **二次拆分**  
   $\text{rebuilt\_kv} \longrightarrow \underset{\text{内容键}}{\boxed{k^C \in \mathbb{R}^{n_h \times d_{\text{nope}}}}} \oplus \underset{\text{内容值}}{\boxed{v^C \in \mathbb{R}^{n_h \times d_v}}}$

---

### **Stage 3: 位置编码处理**
1. **RoPE应用**  
   $\begin{cases}
   q^R \xrightarrow{\text{RoPE}} \tilde{q}^R \\
   k^R \xrightarrow{\text{RoPE}} \tilde{k}^R 
   \end{cases}$

---

### **Stage 4: 最终聚合**
| 组件 | 构建公式 | 维度 |
|------|----------|------|
| **Query** | $Q = \text{Concat}(q^C, \tilde{q}^R)$ | $\mathbb{R}^{n_h \times d_h}$ |
| **Key** | $K = \text{Concat}(k^C, \tilde{k}^R)$ | $\mathbb{R}^{n_h \times d_h}$ |
| **Value** | $V = v^C$ | $\mathbb{R}^{n_h \times d_v}$ |

---

### **Stage 5: 注意力计算**
1. **注意力权重**  
   $\text{AttnWeights} = \text{Softmax}\left( \frac{QK^\top}{\sqrt{d_h}} \right)$

2. **上下文输出**  
   $\text{Output} = \text{AttnWeights} \cdot V$
