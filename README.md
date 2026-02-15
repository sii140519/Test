# 一维稳态热传导 Graph Transformer 示例

这个项目提供了一个用于求解**一维稳态热传导方程**的 Graph Transformer 程序设计与实现模板。

方程形式：

\[
-\frac{d}{dx}\left(k\frac{dT}{dx}\right)=q,\quad x\in[0,L]
\]

在常系数 \(k\) 与均匀热源 \(q\) 条件下，配合 Dirichlet 边界 \(T(0)=T_l, T(L)=T_r\)，解析解为：

\[
T(x)=-\frac{q}{2k}x^2 + C_1x + C_2
\]

代码通过随机采样 \(k, q, T_l, T_r\) 生成图数据（节点为离散空间点），并训练 Graph Transformer 回归每个节点温度。

## 目录结构

- `src/heat_graph_transformer/data.py`：数据生成、数据集定义。
- `src/heat_graph_transformer/model.py`：Graph Transformer 模型。
- `src/heat_graph_transformer/train.py`：训练入口。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch
PYTHONPATH=src python3 -m heat_graph_transformer.train --epochs 30 --batch-size 32
```

## 设计要点

1. **图构建**
   - 节点：一维网格点 \(x_i\)
   - 节点特征：`[x, k, q, T_l, T_r, is_left, is_right]`
   - 边：全连接（通过注意力隐式建模）

2. **模型结构**
   - 输入编码层：MLP 将节点特征映射到隐藏维度
   - 多层 Graph Transformer Block：
     - 多头自注意力（带相对距离偏置）
     - 残差连接 + LayerNorm
     - 前馈网络
   - 输出头：逐节点回归温度值

3. **训练目标**
   - 监督损失：节点温度 MSE

4. **可扩展方向**
   - 变量导热系数 \(k(x)\)
   - Neumann/Robin 边界条件
   - 物理约束损失（PDE 残差）
