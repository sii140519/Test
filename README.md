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

---

## 二维弹性静力有限元程序（矩形板）

已新增 `src/fem2d_elastic_static.py`，用于你描述的工况：

- 几何：`2 x 1` 矩形
- 网格：`0.1 x 0.1` 正方形 Q4 单元
- 边界条件：左端全固定
- 载荷：右端下方节点施加 `Fy = -1000 N`

### 运行方式

```bash
python3 src/fem2d_elastic_static.py
```

若希望自动生成结果图（变形网格 + von Mises 应力云图）：

```bash
python3 src/fem2d_elastic_static.py --plot --plot-dir outputs --disp-scale 1000
```

可选参数示例：

```bash
python3 src/fem2d_elastic_static.py --E 2.1e11 --nu 0.3 --thickness 0.01 --force-y -1000
```

程序会输出：

- 节点数与单元数
- 最大位移
- 右下角加载点位移
- 左端约束反力合力（用于平衡校核）
- 单元中心 von Mises 应力范围
- （可选）`outputs/deformed_mesh.png`、`outputs/von_mises.png`

---

## Level 1 三维钢架梁系程序（steel3d）

已新增 `src/steel3d`，实现基础 3D 梁单元线性静力分析：

- 2 节点 12DOF Euler-Bernoulli 3D 梁单元
- 节点荷载 + 构件整跨均布荷载（局部 `qy/qz`）
- 固定/铰接（平动约束，转角自由）边界
- 稀疏矩阵组装 + `scipy.sparse.linalg.spsolve`
- 端力恢复：`N, Vy, Vz, T, My, Mz`

运行示例门式刚架：

```bash
pip install numpy scipy
PYTHONPATH=src python3 examples/portal_frame_3d.py
```
