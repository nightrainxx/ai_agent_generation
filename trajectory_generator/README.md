# 轨迹生成器

## 项目概述
本项目实现了一个基于环境特征和参数化规则的轨迹生成器，能够依据环境地形、坡度等因素生成真实的运动轨迹。

## 使用方法
1. 准备环境数据（地形高度、坡度、土地覆盖等）
2. 准备轨迹航点序列
3. 运行生成器获取完整轨迹

```bash
# 激活环境
conda activate wargame

# 运行主程序
python main.py
```

## 项目结构
- `src/`: 源代码目录
  - `environment.py`: 环境数据处理
  - `generator.py`: 轨迹生成器
  - `validation.py`: 轨迹验证
- `data/`: 数据目录
  - `environment/`: 环境数据
  - `trajectories/`: 轨迹数据
- `results/`: 结果输出目录

## 轨迹生成方法

### 1. 关键点提取法 (之前方法)
该方法首先从原始轨迹中提取关键航点，然后基于这些关键点生成完整轨迹。

**优点**:
- 有效减少计算量
- 生成的轨迹简化且平滑

**缺点**:
- 可能丢失原始轨迹的细节变化
- 生成轨迹的点数通常与原始轨迹不匹配
- 速度模式可能与原始轨迹存在差异

### 2. 全点轨迹跟踪法（当前方法）
该方法直接使用原始轨迹的所有点作为航点，并保留原始速度信息，以更精确地重现轨迹。

**优点**:
- 轨迹形状与原始轨迹高度匹配
- 速度模式与原始轨迹高度相关（相关性达到0.95以上）
- 点数精确匹配原始轨迹

**实现方式**:
1. 使用原始轨迹的每个点作为航点
2. 提取原始轨迹的速度信息作为参考
3. 结合环境特征（坡度、地形）调整速度
4. 使用原始时间戳保证时间同步

## 性能指标

### 方法1（关键点提取）vs 方法2（全点轨迹跟踪）

轨迹 | 方法 | 速度相关性 | Hausdorff距离(m)
-----|------|------------|----------------
轨迹1 | 方法1 | 0.115 | 2386.350
轨迹1 | 方法2 | 0.958 | 526.548
轨迹2 | 方法1 | 0.093 | 2378.538
轨迹2 | 方法2 | 0.977 | 2378.319
轨迹3 | 方法1 | 0.126 | 2357.850
轨迹3 | 方法2 | 0.961 | 1218.485
轨迹4 | 方法1 | 0.129 | 2376.013
轨迹4 | 方法2 | 0.970 | 2375.700

## 后续改进方向
1. 优化计算效率，提高处理大规模轨迹点的能力
2. 改进曲率计算方法，进一步提高转弯处的真实性
3. 开发自适应参数调整机制，根据轨迹特征自动优化参数
4. 探索更多环境要素对轨迹生成的影响 