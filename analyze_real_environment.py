#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析真实环境数据与轨迹速度的关系，重新拟合模型
使用有效坡度（考虑行进方向）进行分析
采用控制变量法和分箱统计

输入：
- 核心轨迹数据（core_trajectories）
- 真实环境数据（坡度、坡向、土地覆盖类型等）

输出：
- 控制变量后的速度-坡度关系分析
- 分箱统计结果
- 最佳模型参数
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import KBinsDiscretizer
import rasterio
from rasterio.sample import sample_gen
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False
# 增大图表字体大小
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

# 环境数据路径
ENV_DATA_DIR = 'trajectory_generator/data/environment'
SLOPE_FILE = os.path.join(ENV_DATA_DIR, 'slope_aligned.tif')
ASPECT_FILE = os.path.join(ENV_DATA_DIR, 'aspect_aligned.tif')
LANDCOVER_FILE = os.path.join(ENV_DATA_DIR, 'landcover_aligned.tif')
DEM_FILE = os.path.join(ENV_DATA_DIR, 'dem_aligned.tif')

# 核心轨迹数据路径
CORE_TRAJ_DIR = 'core_trajectories'

# 输出目录
OUTPUT_DIR = 'real_environment_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定义土地覆盖类型映射
LANDCOVER_MAPPING = {
    20: '林地',
    40: '灌木地',
    60: '水体',
    # 可以根据实际数据扩展
}

# 定义分析参数
SLOPE_BINS = np.arange(-30, 31, 5)  # 坡度分箱边界
ASPECT_BINS = np.arange(0, 361, 45)  # 坡向分箱边界（8个方向）
DEM_BINS = 5  # 高程分箱数量
CURVATURE_BINS = 5  # 曲率分箱数量

def calculate_angle_difference(angle1, angle2):
    """计算两个角度之间的夹角（0-180度）
    
    Args:
        angle1: 第一个角度（0-360度）
        angle2: 第二个角度（0-360度）
    
    Returns:
        float: 两个角度之间的夹角（0-180度）
    """
    # 确保角度在0-360度范围内
    angle1 = angle1 % 360
    angle2 = angle2 % 360
    
    # 计算夹角
    diff = abs(angle1 - angle2)
    if diff > 180:
        diff = 360 - diff
    
    return diff

def calculate_effective_slope(slope, aspect, heading):
    """计算有效坡度（考虑移动方向）
    
    Args:
        slope: 坡度（度）
        aspect: 坡向（度，0表示正北，顺时针增加）
        heading: 移动方向（度，0表示正北，顺时针增加）
    
    Returns:
        float: 有效坡度（度，正值表示上坡，负值表示下坡）
    """
    if pd.isna(slope) or pd.isna(aspect) or pd.isna(heading):
        return np.nan
        
    # 计算坡向与移动方向的夹角
    angle_diff = calculate_angle_difference(aspect, heading)
    
    # 计算有效坡度
    # 当移动方向与坡向夹角小于90度时为上坡
    # 当夹角大于90度时为下坡
    if angle_diff > 90:
        direction = -1  # 下坡
    else:
        direction = 1   # 上坡
        
    # 计算有效坡度：坡度 * cos(夹角) * 方向
    effective_slope = direction * slope * abs(np.cos(np.radians(angle_diff)))
    
    return effective_slope

def calculate_curvature(points):
    """计算轨迹曲率"""
    if len(points) < 3:
        return np.nan
    
    # 计算两个相邻向量
    v1 = points[1] - points[0]
    v2 = points[2] - points[1]
    
    # 计算曲率（使用叉积和点积）
    cross_product = np.cross(v1, v2)
    dot_product = np.dot(v1, v2)
    
    # 避免除以零
    if np.linalg.norm(v1) * np.linalg.norm(v2) == 0:
        return np.nan
        
    curvature = np.linalg.norm(cross_product) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return curvature

def get_env_data(df, raster_file):
    """从栅格文件中获取环境数据
    
    Args:
        df: 包含经纬度的DataFrame
        raster_file: 栅格文件路径
    
    Returns:
        list: 环境数据值列表
    """
    try:
        with rasterio.open(raster_file) as src:
            # 获取采样点
            coords = list(zip(df['longitude'], df['latitude']))
            
            # 采样环境数据
            values = []
            for val in sample_gen(src, coords):
                values.append(float(val[0]) if val[0] is not None else np.nan)
            
            return values
    except Exception as e:
        print(f"读取环境数据失败: {e}")
        return [np.nan] * len(df)

def load_trajectory(file_path):
    """加载轨迹数据
    
    Args:
        file_path: 轨迹文件路径
    
    Returns:
        DataFrame: 轨迹数据
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 确保必要的列存在
        required_cols = ['timestamp_ms', 'longitude', 'latitude']
        if not all(col in df.columns for col in required_cols):
            print(f"错误: 缺少必要的列 {required_cols}")
            return None
            
        # 转换时间戳
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
        
        # 计算移动方向
        dlon = df['longitude'].diff()
        dlat = df['latitude'].diff()
        df['heading'] = np.degrees(np.arctan2(dlon * np.cos(np.radians(df['latitude'])), dlat)) % 360
        df.loc[0, 'heading'] = df.loc[1, 'heading'] if len(df) > 1 else np.nan
        
        # 计算曲率
        points = np.column_stack([df['longitude'].values, df['latitude'].values])
        df['curvature'] = np.nan
        for i in range(1, len(df)-1):
            df.loc[i, 'curvature'] = calculate_curvature(points[i-1:i+2])
        
        return df
        
    except Exception as e:
        print(f"加载轨迹文件失败: {e}")
        return None

def calculate_speed(df):
    """计算速度
    
    Args:
        df: 轨迹数据DataFrame
    
    Returns:
        DataFrame: 添加了速度的数据
    """
    # 如果已有速度数据，优先使用
    if 'velocity_2d_ms' in df.columns:
        df['speed'] = df['velocity_2d_ms']
        return df
        
    # 计算相邻点的距离（米）
    df['dx'] = df['longitude'].diff() * 111320 * np.cos(np.radians(df['latitude']))
    df['dy'] = df['latitude'].diff() * 110540
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
    
    # 计算时间差（秒）
    df['dt'] = df['timestamp'].diff().dt.total_seconds()
    
    # 计算速度（米/秒）
    df['speed'] = df['distance'] / df['dt']
    
    # 处理第一个点
    df.loc[df.index[0], ['distance', 'dt', 'speed']] = 0
    
    # 移除异常值
    df.loc[df['speed'] > 20, 'speed'] = np.nan  # 速度超过20m/s视为异常
    
    return df

def analyze_controlled_relationship(df):
    """在控制土地覆盖类型和有效坡度区间的情况下分析速度分布
    
    Args:
        df: 包含土地覆盖类型、有效坡度和速度的DataFrame
    
    Returns:
        dict: 分析结果
    """
    results = {}
    
    # 对有效坡度进行分段
    slope_bins = [-30, -20, -10, -5, 0, 5, 10, 20, 30]
    df['slope_bin'] = pd.cut(df['effective_slope'], bins=slope_bins)
    
    # 按土地覆盖类型和坡度区间分组分析
    for landcover_code, landcover_name in LANDCOVER_MAPPING.items():
        landcover_mask = df['landcover'] == landcover_code
        if landcover_mask.sum() == 0:
            continue
            
        results[landcover_name] = []
        
        # 对每个坡度区间进行分析
        for slope_bin in df['slope_bin'].unique():
            if pd.isna(slope_bin):
                continue
                
            # 获取当前组的数据
            mask = landcover_mask & (df['slope_bin'] == slope_bin)
            group = df[mask]
            
            if len(group) < 30:  # 样本量太小的组跳过
                continue
            
            # 计算统计量
            stats = {
                'slope_bin': str(slope_bin),
                'slope_range': f"({slope_bin.left:.1f}, {slope_bin.right:.1f}]",
                'sample_size': len(group),
                'mean_speed': group['velocity_2d_ms'].mean(),
                'std_speed': group['velocity_2d_ms'].std(),
                'median_speed': group['velocity_2d_ms'].median(),
                'mean_slope': group['effective_slope'].mean()
            }
            
            results[landcover_name].append(stats)
    
    return results

def plot_controlled_analysis(results, output_prefix):
    """绘制控制变量分析结果"""
    # 为每种土地覆盖类型绘制箱线图
    for landcover_name, stats_list in results.items():
        if not stats_list:  # 跳过没有数据的地类
            continue
            
        # 转换为DataFrame便于绘图
        df = pd.DataFrame(stats_list)
        
        # 1. 速度-坡度关系图
        plt.figure(figsize=(15, 10))
        plt.errorbar(df['mean_slope'], df['mean_speed'], 
                    yerr=df['std_speed'], fmt='o-', 
                    label=f'平均速度±标准差')
        
        # 添加样本量标注
        for _, row in df.iterrows():
            plt.annotate(f'n={row["sample_size"]}', 
                        (row['mean_slope'], row['mean_speed']),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center')
        
        # 添加趋势线
        mask = df['sample_size'] >= 30
        if mask.sum() >= 2:
            X = df.loc[mask, 'mean_slope'].values.reshape(-1, 1)
            y = df.loc[mask, 'mean_speed'].values
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            x_range = np.array([df['mean_slope'].min(), df['mean_slope'].max()])
            y_pred = model.predict(x_range.reshape(-1, 1))
            plt.plot(x_range, y_pred, 'r--', 
                    label=f'趋势线: y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}')
        
        plt.title(f'{landcover_name}的速度-坡度关系')
        plt.xlabel('有效坡度（度）')
        plt.ylabel('速度（米/秒）')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'{output_prefix}_{landcover_name}_速度坡度关系.png', 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        # 2. 速度分布箱线图
        plt.figure(figsize=(15, 8))
        data = []
        labels = []
        for stats in stats_list:
            speeds = df.loc[df['slope_range'] == stats['slope_range'], 'mean_speed']
            if not speeds.empty:
                data.append(speeds)
                labels.append(f"{stats['slope_range']}\nn={stats['sample_size']}")
        
        if data:  # 确保有数据再绘图
            plt.boxplot(data, labels=labels)
            plt.title(f'{landcover_name}不同坡度区间的速度分布')
            plt.xlabel('有效坡度区间（度）')
            plt.ylabel('速度（米/秒）')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.savefig(f'{output_prefix}_{landcover_name}_速度分布.png', 
                        bbox_inches='tight', dpi=300)
        plt.close()

def analyze_binned_relationship(df, landcover_type=None):
    """分箱分析速度-坡度关系"""
    if landcover_type:
        df = df[df['landcover'].map(LANDCOVER_MAPPING) == landcover_type].copy()
    
    # 创建坡度分箱
    df['slope_bin'] = pd.cut(df['effective_slope'], bins=SLOPE_BINS)
    
    # 计算每个分箱的统计量
    stats = df.groupby('slope_bin').agg({
        'velocity_2d_ms': ['count', 'mean', 'std', 'median'],
        'effective_slope': 'mean'
    }).reset_index()
    
    # 重命名列
    stats.columns = ['slope_bin', 'count', 'mean_speed', 'std_speed', 
                    'median_speed', 'mean_slope']
    
    return stats

def plot_binned_analysis(stats, landcover_type, output_prefix):
    """绘制分箱分析结果"""
    plt.figure(figsize=(15, 10))
    
    # 绘制速度-坡度关系
    plt.errorbar(stats['mean_slope'], stats['mean_speed'], 
                yerr=stats['std_speed'], fmt='o-', 
                label=f'平均速度±标准差')
    
    # 添加样本量标注
    for i, row in stats.iterrows():
        plt.annotate(f'n={row["count"]}', 
                    (row['mean_slope'], row['mean_speed']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center')
    
    plt.title(f'{landcover_type}的速度-坡度关系（分箱分析）')
    plt.xlabel('有效坡度（度）')
    plt.ylabel('速度（米/秒）')
    plt.grid(True)
    plt.legend()
    
    # 添加趋势线
    mask = stats['count'] >= 30  # 只用样本量足够的箱子拟合趋势线
    if mask.sum() >= 2:
        X = stats.loc[mask, 'mean_slope'].values.reshape(-1, 1)
        y = stats.loc[mask, 'mean_speed'].values
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        x_range = np.array([stats['mean_slope'].min(), stats['mean_slope'].max()])
        y_pred = model.predict(x_range.reshape(-1, 1))
        plt.plot(x_range, y_pred, 'r--', 
                label=f'趋势线: y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}')
        plt.legend()
    
    plt.savefig(f'{output_prefix}_{landcover_type}_分箱分析.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def main():
    """主函数"""
    print("开始分析真实环境数据...")
    
    # 获取核心轨迹文件列表
    traj_files = []
    for root, dirs, files in os.walk(CORE_TRAJ_DIR):
        for file in files:
            if file.endswith('_core.csv'):
                traj_files.append(os.path.join(root, file))
    
    if not traj_files:
        print(f"错误: 未找到核心轨迹文件")
        return
    
    print(f"找到 {len(traj_files)} 个核心轨迹文件")
    
    # 存储所有轨迹的分析结果
    all_results = {}
    
    # 处理每个轨迹文件
    for traj_file in traj_files:
        print(f"\n处理轨迹文件: {os.path.basename(traj_file)}")
        
        try:
            # 加载轨迹数据
            df = load_trajectory(traj_file)
            if df is None:
                continue
            print(f"  加载轨迹数据成功，共 {len(df)} 行")
            
            # 获取环境数据
            df['slope'] = get_env_data(df, SLOPE_FILE)
            df['aspect'] = get_env_data(df, ASPECT_FILE)
            df['landcover'] = get_env_data(df, LANDCOVER_FILE)
            
            print(f"  环境数据获取成功:")
            print(f"    坡度数据: {df['slope'].notna().sum()} 个有效值")
            print(f"    坡向数据: {df['aspect'].notna().sum()} 个有效值")
            print(f"    土地覆盖: {df['landcover'].notna().sum()} 个有效值")
            
            # 计算有效坡度
            df['effective_slope'] = df.apply(
                lambda row: calculate_effective_slope(
                    row['slope'], 
                    row['aspect'], 
                    row['heading']
                ), 
                axis=1
            )
            print(f"  有效坡度计算成功: {df['effective_slope'].notna().sum()} 个有效值")
            
            # 控制变量分析
            print("\n  进行控制变量分析...")
            results = analyze_controlled_relationship(df)
            
            # 保存结果
            traj_id = os.path.basename(traj_file).split('_')[1]
            all_results[traj_id] = results
            
            # 绘制分析结果
            plot_controlled_analysis(
                results,
                os.path.join(OUTPUT_DIR, f'轨迹{traj_id}')
            )
            
        except Exception as e:
            print(f"处理文件 {traj_file} 时出错: {e}")
            import traceback
            print(traceback.format_exc())
            continue
    
    # 保存总体分析结果
    print("\n保存分析结果...")
    import json
    with open(os.path.join(OUTPUT_DIR, 'controlled_analysis_results.json'), 
              'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print("\n分析完成！结果保存在", OUTPUT_DIR)

if __name__ == '__main__':
    main() 