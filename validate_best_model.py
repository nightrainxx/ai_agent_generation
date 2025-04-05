#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最佳模型验证脚本
- 使用最佳线性模型和残差分布
- 使用真实的坡度和土地覆盖数据
- 在原始轨迹上验证模型效果
- 比较生成速度与原始速度的差异
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import glob
import seaborn as sns
from matplotlib.font_manager import FontProperties
import rasterio  # 添加 rasterio 库
from rasterio.windows import Window
from rasterio.transform import Affine

# 设置中文字体
try:
    chinese_font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
except:
    chinese_font = FontProperties()

# 设置更大的字体大小
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

# 目录设置
data_dir = "trajectory_generator/data/trajectories"
env_dir = "trajectory_generator/data/environment" # 环境数据目录
output_dir = "model_validation"
analysis_dir = "analysisi_result"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# --- 文件路径 ---
slope_file = os.path.join(env_dir, "slope_aligned.tif")
landcover_file = os.path.join(env_dir, "landcover_aligned.tif")

# 统一使用10秒窗口
BEST_WINDOW = "10秒"
LAND_BEST_WINDOWS = {
    "林地": "10秒",
    "灌木地": "10秒",
    "水体": "10秒"
}

# 线性模型参数 (从之前分析中获取)
LINEAR_MODELS = {
    "林地": {"slope": -0.0145, "intercept": 3.32},
    "灌木地": {"slope": -0.0096, "intercept": 4.37},
    "水体": {"slope": -0.0004, "intercept": 1.40}
}

# 残差分布参数 (从之前分析中获取)
RESIDUAL_MODELS = {
    "林地": {"type": "laplace", "params": {"loc": -0.0030, "scale": 0.4091}},
    "灌木地": {"type": "t", "params": {"df": 3.62, "loc": 0.036, "scale": 0.457}},
    "水体": {"type": "gmm", "params": {
        "means": [-0.8065, 0.5541],
        "weights": [0.4072, 0.5928],
        "covariances": [0.3710, 0.2694]
    }}
}

# 速度限制
MIN_SPEED = 1.0
MAX_SPEED = 8.0

# 土地覆盖代码映射 (根据需要调整)
LANDCOVER_MAP = {
    20: "林地",
    40: "灌木地",
    60: "水体"
}
DEFAULT_LANDCOVER = "其他"

def load_trajectory_data(file_path):
    """加载轨迹数据"""
    try:
        df = pd.read_csv(file_path)
        # 确保坐标列是数值类型
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df.dropna(subset=['latitude', 'longitude'], inplace=True)
        return df
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None

def get_raster_value(raster_path, coordinates):
    """从栅格文件中获取指定坐标的值"""
    values = []
    try:
        with rasterio.open(raster_path) as src:
            # coordinates 应该是 [(lon1, lat1), (lon2, lat2), ...]
            # rasterio.sample 返回一个生成器，需要转换为列表
            sampled_values = list(src.sample(coordinates))
            # 每个样本是一个包含单个值的数组，例如 [value]
            values = [val[0] if len(val) > 0 else np.nan for val in sampled_values]
    except Exception as e:
        print(f"读取栅格文件 {raster_path} 时出错: {e}")
        # 如果出错，返回NaN列表
        values = [np.nan] * len(coordinates)
    return values

# 替换旧的随机函数
def get_landcover_data(lats, lons):
    """获取批量坐标点的土地覆盖类型数据"""
    coordinates = list(zip(lons, lats)) # rasterio.sample 需要 (lon, lat) 格式
    landcover_codes = get_raster_value(landcover_file, coordinates)
    # 转换为类型名称，未知代码或NaN映射到默认值
    landcover_types = [LANDCOVER_MAP.get(code, DEFAULT_LANDCOVER) if pd.notna(code) else DEFAULT_LANDCOVER for code in landcover_codes]
    return landcover_types

def get_slope_data(lats, lons):
    """获取批量坐标点的坡度数据"""
    coordinates = list(zip(lons, lats)) # rasterio.sample 需要 (lon, lat) 格式
    slopes = get_raster_value(slope_file, coordinates)
    # 处理可能的NaN值（例如坐标在栅格范围外）
    slopes = [s if pd.notna(s) else 0 for s in slopes] # 默认坡度为0
    return slopes

def generate_residual(land_type):
    """根据地形类型生成随机残差"""
    if land_type not in RESIDUAL_MODELS:
        # 对于未知或"其他"地类，使用小的随机噪声
        return np.random.normal(0, 0.1)
        
    model = RESIDUAL_MODELS[land_type]
    
    if model["type"] == "laplace":
        return np.random.laplace(loc=model["params"]["loc"], 
                               scale=model["params"]["scale"])
    elif model["type"] == "t":
        return stats.t.rvs(df=model["params"]["df"], 
                         loc=model["params"]["loc"], 
                         scale=model["params"]["scale"])
    elif model["type"] == "gmm":
        # 从GMM模型生成
        component = np.random.choice(len(model["params"]["weights"]), 
                                   p=model["params"]["weights"])
        mean = model["params"]["means"][component]
        std = np.sqrt(model["params"]["covariances"][component])
        return np.random.normal(mean, std)
    else:
        # 默认使用正态分布
        return np.random.normal(0, 0.1)

# landcover_to_type 函数不再需要，直接使用 LANDCOVER_MAP

def calculate_speed(slope, land_type):
    """计算速度 (线性模型 + 残差)"""
    # 获取线性模型参数，未知类型使用默认值
    model = LINEAR_MODELS.get(land_type, {"slope": -0.02, "intercept": 4.0})
    
    # 计算基础速度
    base_speed = model["slope"] * slope + model["intercept"]
    
    # 生成随机残差
    residual = generate_residual(land_type)
    
    # 添加残差
    speed = base_speed + residual
    
    # 确保速度在合理范围内
    speed = max(MIN_SPEED, min(speed, MAX_SPEED))
    
    return speed, base_speed, residual

def moving_average(data, window_size):
    """计算移动平均"""
    if window_size <= 1:
        return data
    
    weights = np.ones(window_size) / window_size
    # 使用 'same' 模式保持长度一致，边缘用部分窗口计算
    # 注意：这可能引入边缘效应，但避免了长度不匹配问题
    smoothed = np.convolve(data, weights, mode='same')
    
    # 修复边缘，使用部分窗口
    for i in range(window_size // 2):
         smoothed[i] = np.mean(data[:i+window_size//2+1])
         smoothed[-(i+1)] = np.mean(data[-(i+window_size//2+1):])
         
    return smoothed

def aggregate_trajectory(df, window_size):
    """按窗口大小聚合轨迹数据"""
    # 确定聚合因子
    if window_size.endswith('秒'):
        seconds = int(window_size.replace('秒', ''))
    else:
        seconds = int(window_size)
        
    # 计算聚合因子 (假设采样频率为1秒)
    agg_factor = max(1, seconds)
    
    # 按聚合因子对数据进行采样
    agg_df = df.iloc[::agg_factor].copy().reset_index(drop=True)
    
    # 使用移动平均平滑速度 (保持长度)
    if agg_factor > 1 and len(df) >= agg_factor:
         agg_df['velocity_north_ms'] = moving_average(df['velocity_north_ms'].values, agg_factor)[::agg_factor]
         agg_df['velocity_east_ms'] = moving_average(df['velocity_east_ms'].values, agg_factor)[::agg_factor]
         # 重新计算聚合后的原始速度
         agg_df['original_speed'] = np.sqrt(agg_df['velocity_north_ms']**2 + agg_df['velocity_east_ms']**2)
    else:
         # 如果不聚合或数据太少，直接使用原始速度
         agg_df['original_speed'] = np.sqrt(agg_df['velocity_north_ms']**2 + agg_df['velocity_east_ms']**2)

    return agg_df

def validate_trajectory(traj_file):
    """验证单个轨迹"""
    print(f"\n验证轨迹: {os.path.basename(traj_file)}")
    
    # 加载轨迹数据
    df = load_trajectory_data(traj_file)
    if df is None or df.empty:
        print("加载轨迹失败或轨迹为空")
        return None
    
    # 检查必要的列是否存在
    required_cols = ['latitude', 'longitude', 'velocity_north_ms', 'velocity_east_ms']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"缺少必要的列: {missing_cols}")
        return None
    
    # 获取轨迹ID
    traj_id = os.path.splitext(os.path.basename(traj_file))[0].split('_')[-1]
    
    # 创建结果数据框
    results = []
    
    # --- 聚合 --- 
    window = BEST_WINDOW  # 固定为10秒
    agg_df = aggregate_trajectory(df, window)
    if agg_df.empty:
        print("聚合后轨迹为空")
        return None

    # --- 获取真实环境数据 --- 
    lats = agg_df['latitude'].tolist()
    lons = agg_df['longitude'].tolist()
    slope_values = get_slope_data(lats, lons)
    landcover_types = get_landcover_data(lats, lons)

    agg_df['slope'] = slope_values
    agg_df['landcover_type'] = landcover_types

    # --- 生成速度 --- 
    generated_speeds = []
    base_speeds = []
    residuals = []

    for i, row in agg_df.iterrows():
        slope = row['slope']
        land_type = row['landcover_type']
        
        # 计算速度
        speed, base_speed, residual = calculate_speed(slope, land_type)
        generated_speeds.append(speed)
        base_speeds.append(base_speed)
        residuals.append(residual)

    # 添加到聚合后的数据框
    agg_df['generated_speed'] = generated_speeds
    agg_df['base_speed'] = base_speeds
    agg_df['residual'] = residuals

    # --- 评估 --- 
    # 确保原始速度和生成速度长度一致且非空
    if len(agg_df['original_speed']) == 0 or len(agg_df['generated_speed']) == 0:
        print("原始速度或生成速度为空，无法评估")
        return None
    if len(agg_df['original_speed']) != len(agg_df['generated_speed']):
        print("原始速度和生成速度长度不匹配，无法评估")
        # 尝试找出问题
        print(f"原始速度长度: {len(agg_df['original_speed'])}, 生成速度长度: {len(agg_df['generated_speed'])}")
        return None

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    r2 = r2_score(agg_df['original_speed'], agg_df['generated_speed'])
    mae = mean_absolute_error(agg_df['original_speed'], agg_df['generated_speed'])
    rmse = np.sqrt(mean_squared_error(agg_df['original_speed'], agg_df['generated_speed']))

    # 计算相关系数
    corr_matrix = np.corrcoef(agg_df['original_speed'], agg_df['generated_speed'])
    # 检查结果是否为标量（如果只有一个数据点）
    if corr_matrix.ndim > 1:
        corr = corr_matrix[0, 1]
    else:
        corr = np.nan # 或者 1.0，取决于如何定义单个点的相关性

    # 计算平均速度差异 (避免除以0)
    valid_speeds = agg_df['original_speed'] > 0.1
    if valid_speeds.any():
        mean_diff = np.mean(agg_df.loc[valid_speeds, 'generated_speed'] - agg_df.loc[valid_speeds, 'original_speed'])
        mean_pct_diff = np.mean(np.abs(agg_df.loc[valid_speeds, 'generated_speed'] - agg_df.loc[valid_speeds, 'original_speed']) / agg_df.loc[valid_speeds, 'original_speed']) * 100
    else:
        mean_diff = np.nan
        mean_pct_diff = np.nan

    results.append({
        'trajectory': traj_id,
        'window': window,
        'r2': r2,
        'correlation': corr,
        'mae': mae,
        'rmse': rmse,
        'mean_diff': mean_diff,
        'mean_pct_diff': mean_pct_diff,
        'df': agg_df
    })
    
    # --- 绘图 --- 
    # (绘图代码保持不变，但现在使用真实数据)
    plt.figure(figsize=(18, 12))
    
    # 1. 速度时间序列
    plt.subplot(2, 2, 1)
    time_index = np.arange(len(agg_df))  # 使用简单索引作为x轴
    plt.plot(time_index, agg_df['original_speed'], 'b-', linewidth=2.5, label='原始速度')
    plt.plot(time_index, agg_df['generated_speed'], 'r-', linewidth=2.5, label='生成速度')
    plt.plot(time_index, agg_df['base_speed'], 'g--', linewidth=1.5, label='基础速度(无残差)')
    plt.title(f'轨迹{traj_id} - {window}窗口速度时间序列', fontproperties=chinese_font, fontsize=22)
    plt.xlabel('采样点', fontproperties=chinese_font, fontsize=18)
    plt.ylabel('速度 (m/s)', fontproperties=chinese_font, fontsize=18)
    plt.legend(prop=chinese_font, fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # 2. 地形类型随时间变化曲线
    ax2 = plt.subplot(2, 2, 2)
    land_type_codes = {"林地": 1, "灌木地": 2, "水体": 3, DEFAULT_LANDCOVER: 0}
    land_colors = {"林地": 'green', "灌木地": 'orange', "水体": 'blue', DEFAULT_LANDCOVER: 'gray'}
    
    # 绘制地形类型随时间变化
    plotted_types = set()
    for i, land_type in enumerate(land_type_codes.keys()):
        mask = agg_df['landcover_type'] == land_type
        if np.any(mask):
            label = land_type if land_type not in plotted_types else ""
            plt.scatter(time_index[mask], 
                      [land_type_codes[land_type]] * np.sum(mask), 
                      c=land_colors[land_type], 
                      s=100, 
                      alpha=0.7,
                      label=label)
            plotted_types.add(land_type)
    
    plt.title('地形类型随时间变化', fontproperties=chinese_font, fontsize=22)
    plt.xlabel('采样点', fontproperties=chinese_font, fontsize=18)
    plt.yticks(list(land_type_codes.values()), list(land_type_codes.keys()), fontproperties=chinese_font, fontsize=16)
    plt.legend(prop=chinese_font, fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # 3. 速度-坡度散点图
    plt.subplot(2, 2, 3)
    plotted_types = set()
    for land_type in land_colors.keys():
        mask = agg_df['landcover_type'] == land_type
        if np.any(mask):
            label_orig = f'{land_type}-原始' if land_type not in plotted_types else ""
            label_gen = f'{land_type}-生成' if land_type not in plotted_types else ""
            plt.scatter(agg_df.loc[mask, 'slope'], 
                      agg_df.loc[mask, 'original_speed'], 
                      c=land_colors[land_type], 
                      alpha=0.5, 
                      s=80,
                      marker='o',
                      label=label_orig)
            plt.scatter(agg_df.loc[mask, 'slope'], 
                      agg_df.loc[mask, 'generated_speed'], 
                      c=land_colors[land_type], 
                      alpha=0.5, 
                      s=80,
                      marker='x',
                      label=label_gen)
            plotted_types.add(land_type)
    
    # 添加线性模型曲线
    plotted_models = set()
    for land_type, model in LINEAR_MODELS.items():
        label_model = f'{land_type}模型' if land_type not in plotted_models else ""
        x_range = np.linspace(-30, 30, 100)
        y_pred = model['slope'] * x_range + model['intercept']
        plt.plot(x_range, y_pred, 
               c=land_colors.get(land_type, 'black'), 
               linestyle='--', 
               linewidth=2,
               label=label_model)
        plotted_models.add(land_type)
    
    plt.title('速度-坡度关系', fontproperties=chinese_font, fontsize=22)
    plt.xlabel('坡度', fontproperties=chinese_font, fontsize=18)
    plt.ylabel('速度 (m/s)', fontproperties=chinese_font, fontsize=18)
    plt.legend(prop=chinese_font, fontsize=14, loc='best')
    plt.grid(True, alpha=0.3)
    
    # 4. 速度分布直方图
    plt.subplot(2, 2, 4)
    bins = np.linspace(0, 10, 20)
    plt.hist(agg_df['original_speed'], bins=bins, alpha=0.5, label='原始速度分布', color='blue')
    plt.hist(agg_df['generated_speed'], bins=bins, alpha=0.5, label='生成速度分布', color='red')
    plt.title('速度分布对比', fontproperties=chinese_font, fontsize=22)
    plt.xlabel('速度 (m/s)', fontproperties=chinese_font, fontsize=18)
    plt.ylabel('频次', fontproperties=chinese_font, fontsize=18)
    plt.legend(prop=chinese_font, fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = (
        f"R² = {r2:.4f}\n"
        f"相关系数 = {corr:.4f}\n"
        f"MAE = {mae:.4f} m/s\n"
        f"RMSE = {rmse:.4f} m/s\n"
        f"平均相对误差 = {mean_pct_diff:.2f}%"
    )
    plt.figtext(0.5, 0.01, stats_text, ha='center', fontproperties=chinese_font, fontsize=16, 
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'轨迹{traj_id}模型验证 (聚合窗口: {window})', fontproperties=chinese_font, fontsize=24)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"轨迹{traj_id}_{window}_综合分析.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 额外绘制地形分类残差分布
    plt.figure(figsize=(18, 12))
    plotted_types = set()
    for i, land_type in enumerate(["林地", "灌木地", "水体"]):
        land_df = agg_df[agg_df['landcover_type'] == land_type]
        if len(land_df) > 0:
            ax_res = plt.subplot(2, 2, i+1)
            residuals = land_df['residual']
            plt.hist(residuals, bins=20, alpha=0.7, density=True, color=land_colors[land_type], label=f'{land_type}实际残差')
            
            # 拟合分布曲线
            x_range = np.linspace(min(residuals) - 1, max(residuals) + 1, 1000)
            label_dist = ""
            if land_type == "林地":
                loc = RESIDUAL_MODELS[land_type]["params"]["loc"]
                scale = RESIDUAL_MODELS[land_type]["params"]["scale"]
                pdf = stats.laplace.pdf(x_range, loc=loc, scale=scale)
                label_dist = f'拉普拉斯(loc={loc:.2f}, scale={scale:.2f})'
            elif land_type == "灌木地":
                df_t = RESIDUAL_MODELS[land_type]["params"]["df"]
                loc = RESIDUAL_MODELS[land_type]["params"]["loc"]
                scale = RESIDUAL_MODELS[land_type]["params"]["scale"]
                pdf = stats.t.pdf(x_range, df=df_t, loc=loc, scale=scale)
                label_dist = f't分布(df={df_t:.1f}, loc={loc:.2f}, scale={scale:.2f})'
            elif land_type == "水体":
                means = RESIDUAL_MODELS[land_type]["params"]["means"]
                weights = RESIDUAL_MODELS[land_type]["params"]["weights"]
                covariances = RESIDUAL_MODELS[land_type]["params"]["covariances"]
                gmm_pdf = np.zeros_like(x_range)
                for j in range(len(means)):
                    pdf_j = stats.norm.pdf(x_range, loc=means[j], scale=np.sqrt(covariances[j]))
                    gmm_pdf += weights[j] * pdf_j
                pdf = gmm_pdf
                label_dist = 'GMM分布'
            else:
                pdf = None

            if pdf is not None:
                 plt.plot(x_range, pdf, 'r-', linewidth=2, label=f'{label_dist}模型')
            
            plt.title(f'{land_type}残差分布 (n={len(land_df)})', fontproperties=chinese_font, fontsize=22)
            plt.xlabel('残差值', fontproperties=chinese_font, fontsize=18)
            plt.ylabel('概率密度', fontproperties=chinese_font, fontsize=18)
            plt.legend(prop=chinese_font, fontsize=14)
            plt.grid(True, alpha=0.3)
            plotted_types.add(land_type)
    
    # 残差时间序列图
    plt.subplot(2, 2, 4)
    plotted_types_ts = set()
    for land_type in land_colors.keys():
        mask = agg_df['landcover_type'] == land_type
        if np.any(mask):
            label = land_type if land_type not in plotted_types_ts else ""
            plt.scatter(time_index[mask], agg_df.loc[mask, 'residual'], 
                      c=land_colors[land_type], alpha=0.7, s=50, label=label)
            plotted_types_ts.add(land_type)
            
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title('残差时间序列', fontproperties=chinese_font, fontsize=22)
    plt.xlabel('采样点', fontproperties=chinese_font, fontsize=18)
    plt.ylabel('残差值', fontproperties=chinese_font, fontsize=18)
    plt.legend(prop=chinese_font, fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'轨迹{traj_id}残差分析 (聚合窗口: {window})', fontproperties=chinese_font, fontsize=24)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"轨迹{traj_id}_{window}_残差分析.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def analyze_all_trajectories():
    """分析所有轨迹"""
    # 检查环境文件是否存在
    if not os.path.exists(slope_file) or not os.path.exists(landcover_file):
        print(f"错误：环境数据文件不存在于 {env_dir}")
        print("请确保 slope_aligned.tif 和 landcover_aligned.tif 文件存在。")
        # 可以选择创建示例数据或直接退出
        # print("正在创建示例环境数据...")
        # create_sample_environment_data() # 如果需要
        return

    # 查找所有轨迹文件
    trajectory_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    # 如果没有找到轨迹文件，创建一个示例轨迹
    if not trajectory_files:
        print("未找到轨迹文件，创建示例轨迹...")
        create_sample_trajectory()
        trajectory_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    # 验证所有轨迹
    all_results = []
    for file_path in trajectory_files:
        results = validate_trajectory(file_path)
        if results:
            all_results.extend(results)
    
    if not all_results:
        print("未能成功验证任何轨迹。")
        return
        
    # 创建汇总报告
    create_summary_report(all_results)

def create_sample_trajectory():
    """创建示例轨迹用于测试"""
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.join(data_dir, 'sample_trajectory.csv')), exist_ok=True)
    
    # 创建时间序列
    n_points = 1000
    timestamps = np.arange(0, n_points) * 1000 + 1637841694000  # 模拟真实时间戳
    
    # 创建坐标 (模拟经纬度 - 注意: 这些坐标需要落在示例栅格范围内)
    # 假设示例栅格覆盖 [lon_min, lat_min] 到 [lon_max, lat_max]
    # 这里使用随机游走，需要确保它与你的栅格范围大致匹配
    # TODO: 根据你的栅格范围调整起始点和步长
    start_lat, start_lon = 8846764.0, -981867.0 # 需要是栅格内的点
    lat_step, lon_step = 0.1, 0.1 # 调整步长以匹配栅格分辨率
    latitude = start_lat + np.cumsum(np.random.normal(0, lat_step, n_points))
    longitude = start_lon + np.cumsum(np.random.normal(0, lon_step, n_points))
    altitude = 330.0 + np.random.normal(0, 0.5, n_points)
    
    # 创建速度
    speeds = 3 + np.sin(np.linspace(0, 10, n_points)) + np.random.normal(0, 0.5, n_points)
    directions = np.linspace(0, 2*np.pi, n_points) + np.random.normal(0, 0.1, n_points)
    
    # 计算北向和东向速度
    velocity_north = speeds * np.cos(directions)
    velocity_east = speeds * np.sin(directions)
    velocity_down = np.random.normal(0, 0.3, n_points)
    
    # 创建数据框
    df = pd.DataFrame({
        'timestamp_ms': timestamps,
        'latitude': latitude,
        'longitude': longitude,
        'altitude_m': altitude,
        'velocity_north_ms': velocity_north,
        'velocity_east_ms': velocity_east,
        'velocity_down_ms': velocity_down,
        'velocity_2d_ms': speeds,
        'velocity_3d_ms': np.sqrt(speeds**2 + velocity_down**2)
    })
    
    # 保存为CSV
    df.to_csv(os.path.join(data_dir, 'sample_trajectory.csv'), index=False)
    print(f"示例轨迹已保存到: {os.path.join(data_dir, 'sample_trajectory.csv')}")

# --- 示例环境数据创建 (如果需要) ---
def create_sample_environment_data(rows=100, cols=100, resolution=1.0):
    """创建一个简单的示例环境栅格文件"""
    os.makedirs(env_dir, exist_ok=True)
    
    # 定义变换: 左上角坐标和分辨率
    # TODO: 根据你的实际坐标范围调整
    transform = Affine(resolution, 0.0, -981900, 0.0, -resolution, 8846800)
    
    # 创建坡度数据 (简单梯度)
    slope_data = np.fromfunction(lambda i, j: (i / rows * 30) + (j / cols * 10), (rows, cols), dtype=np.float32)
    slope_meta = {
        'driver': 'GTiff',
        'height': rows,
        'width': cols,
        'count': 1,
        'dtype': 'float32',
        'crs': 'EPSG:3857', # 假设为 Web Mercator，需要根据实际情况修改
        'transform': transform,
        'nodata': -9999
    }
    with rasterio.open(slope_file, 'w', **slope_meta) as dst:
        dst.write(slope_data, 1)
    print(f"示例坡度文件已创建: {slope_file}")
    
    # 创建土地覆盖数据 (随机分配)
    landcover_data = np.random.choice([20, 40, 60, 0], size=(rows, cols), p=[0.3, 0.4, 0.2, 0.1]).astype(np.uint8)
    landcover_meta = slope_meta.copy()
    landcover_meta['dtype'] = 'uint8'
    landcover_meta['nodata'] = 0 # 假设0是nodata或未知
    with rasterio.open(landcover_file, 'w', **landcover_meta) as dst:
        dst.write(landcover_data, 1)
    print(f"示例土地覆盖文件已创建: {landcover_file}")

def create_summary_report(all_results):
    """创建汇总报告"""
    report = ["# 最佳模型验证报告 (使用真实环境数据)", ""]
    
    # 检查结果是否为空
    if not all_results:
        report.append("未能生成任何有效的验证结果。")
        with open(os.path.join(output_dir, "模型验证报告.md"), 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        return
        
    # 统计所有窗口的结果
    windows = list(set(r['window'] for r in all_results))
    trajectories = list(set(r['trajectory'] for r in all_results))
    
    # 添加整体统计
    report.append("## 整体性能统计")
    report.append("")
    report.append("| 聚合窗口 | 平均R² | 平均相关系数 | 平均MAE | 平均RMSE | 平均误差(%) |")
    report.append("| -------- | ------ | ------------ | ------- | -------- | ----------- |")
    
    for window in windows:
        window_results = [r for r in all_results if r['window'] == window]
        # 过滤掉NaN值
        avg_r2 = np.nanmean([r['r2'] for r in window_results])
        avg_corr = np.nanmean([r['correlation'] for r in window_results])
        avg_mae = np.nanmean([r['mae'] for r in window_results])
        avg_rmse = np.nanmean([r['rmse'] for r in window_results])
        avg_pct_diff = np.nanmean([r['mean_pct_diff'] for r in window_results])
        
        report.append(f"| {window} | {avg_r2:.4f} | {avg_corr:.4f} | {avg_mae:.4f} | {avg_rmse:.4f} | {avg_pct_diff:.2f} |")
    
    report.append("")
    
    # 添加每个轨迹的统计
    report.append("## 轨迹特定性能")
    report.append("")
    
    for traj in trajectories:
        report.append(f"### 轨迹 {traj}")
        report.append("")
        report.append("| 聚合窗口 | R² | 相关系数 | MAE | RMSE | 平均误差(%) |")
        report.append("| -------- | -- | -------- | --- | ---- | ----------- |")
        
        for window in windows:
            traj_window_results = [r for r in all_results if r['trajectory'] == traj and r['window'] == window]
            if traj_window_results:
                r = traj_window_results[0]
                report.append(f"| {window} | {r['r2']:.4f} | {r['correlation']:.4f} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['mean_pct_diff']:.2f} |")
        
        report.append("")
    
    # 结论和建议
    best_window_results = [r for r in all_results if r['window'] == BEST_WINDOW]
    avg_best_r2 = np.nanmean([r['r2'] for r in best_window_results])
    avg_best_corr = np.nanmean([r['correlation'] for r in best_window_results])
    avg_best_pct_diff = np.nanmean([r['mean_pct_diff'] for r in best_window_results])
    
    report.append("## 结论和建议")
    report.append("")
    report.append(f"基于{BEST_WINDOW}聚合窗口和真实环境数据的验证结果：")
    report.append("")
    report.append(f"- 平均R²值: {avg_best_r2:.4f}")
    report.append(f"- 平均相关系数: {avg_best_corr:.4f}")
    report.append(f"- 平均相对误差: {avg_best_pct_diff:.2f}%")
    report.append("")
    
    if avg_best_r2 >= 0.7 and avg_best_corr >= 0.8 and avg_best_pct_diff <= 15:
        report.append("**结论**: 模型表现良好，能够较好地预测不同地形和坡度条件下的速度。")
    elif avg_best_r2 >= 0.5 and avg_best_corr >= 0.7 and avg_best_pct_diff <= 25:
        report.append("**结论**: 模型表现一般，能大致预测速度趋势，但精度有待提高，尤其是在某些地形或坡度变化区域。")
    else:
        report.append("**结论**: 模型表现不佳，生成速度与原始速度差异较大。需要进一步改进模型或检查数据。")
    
    report.append("")
    report.append("**可能原因与改进方向**:")
    report.append("")
    report.append("1. **坐标系问题**: 确认轨迹坐标 (`latitude`, `longitude`) 与环境栅格文件的坐标系 (`CRS`) 是否一致。如果不一致，需要进行转换。")
    report.append("2. **模型参数**: 当前线性模型和残差参数可能不适用于真实数据，需要基于真实坡度和地类重新拟合。")
    report.append("3. **数据聚合方法**: 10秒窗口可能过大，导致细节丢失。可以尝试更小的窗口（如之前分析建议的2-5秒）。移动平均方法也可能过度平滑。")
    report.append("4. **未考虑因素**: 模型未包含曲率、历史速度等因素，这些可能对速度有显著影响。")
    report.append("5. **土地覆盖映射**: 确认 `LANDCOVER_MAP` 中的代码与 `landcover_aligned.tif` 文件中的实际值匹配。")
    
    # 模型使用指南 (保持不变)
    report.append("")
    report.append("## 模型使用指南")
    # ... (省略)
    
    # 写入文件
    with open(os.path.join(output_dir, "模型验证报告.md"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

def main():
    """主函数"""
    print("开始模型验证 (使用真实环境数据)...")
    
    # 分析所有轨迹
    analyze_all_trajectories()
    
    print(f"模型验证完成！结果保存在 {output_dir} 目录中")

if __name__ == "__main__":
    main() 