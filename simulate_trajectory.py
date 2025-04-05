#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轨迹速度模拟器

基于真实环境数据分析的模型参数，模拟生成轨迹速度。

输入：
- 轨迹路径点（经纬度）
- 环境数据（坡度、土地覆盖类型）

输出：
- 带有模拟速度的轨迹
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import rasterio
from rasterio.sample import sample_gen
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False
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

# 输出目录
OUTPUT_DIR = 'visualization_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定义土地覆盖类型映射
LANDCOVER_MAPPING = {
    20: '林地',
    40: '灌木地',
    60: '水体',
}

# 速度模型参数（基于分析结果优化）
SPEED_MODELS = {
    '林地': {
        'base_speed': 5.8,  # 基础速度
        'slope_effect': {
            'uphill': -0.15,  # 上坡影响系数
            'downhill': 0.08   # 下坡影响系数
        },
        'std_dev': 0.4,  # 减小随机扰动
        'max_speed': 8.0,  # 提高最大速度限制
        'min_speed': 4.0,   # 最小速度限制
        'transition_weight': 0.8  # 地类过渡权重
    },
    '灌木地': {
        'base_speed': 4.8,
        'slope_effect': {
            'uphill': -0.25,
            'downhill': 0.12
        },
        'std_dev': 0.6,
        'max_speed': 6.5,
        'min_speed': 2.0,
        'transition_weight': 0.6
    },
    '水体': {
        'base_speed': 6.1,
        'slope_effect': {
            'uphill': -0.10,
            'downhill': 0.15
        },
        'std_dev': 0.3,
        'max_speed': 7.5,
        'min_speed': 5.0,
        'transition_weight': 0.9
    }
}

def load_trajectory(traj_file):
    """加载轨迹数据"""
    df = pd.read_csv(traj_file)
    
    # 确保数据包含必要的列
    required_cols = ['timestamp_ms', 'longitude', 'latitude', 'altitude_m']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"轨迹数据缺少必要的列: {col}")
    
    # 重命名列以保持一致性
    df = df.rename(columns={
        'timestamp_ms': 'timestamp',
        'altitude_m': 'altitude'
    })
    
    # 将timestamp_ms转换为秒
    df['timestamp'] = df['timestamp'] / 1000.0
    
    return df

def get_env_data(df, raster_file):
    """从栅格文件获取环境数据"""
    with rasterio.open(raster_file) as src:
        # 获取坐标列表
        coords = [(x, y) for x, y in zip(df['longitude'], df['latitude'])]
        
        # 采样栅格值
        values = []
        for val in sample_gen(src, coords):
            values.append(float(val[0]) if val[0] is not None else np.nan)
        
    return values

def calculate_distances(df):
    """计算轨迹点之间的距离"""
    # 计算相邻点的距离
    df['dx'] = df['longitude'].diff() * 111320 * np.cos(np.radians(df['latitude']))
    df['dy'] = df['latitude'].diff() * 110540
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
    
    # 计算时间差（秒）
    df['dt'] = df['timestamp'].diff()
    
    # 第一个点
    df.loc[0, 'distance'] = 0
    df.loc[0, 'dt'] = 0
    
    return df

def calculate_angle_difference(angle1, angle2):
    """计算两个角度之间的夹角（0-180度）"""
    angle1 = angle1 % 360
    angle2 = angle2 % 360
    diff = abs(angle1 - angle2)
    return diff if diff <= 180 else 360 - diff

def calculate_effective_slope(slope, aspect, heading):
    """计算有效坡度（考虑移动方向）"""
    if pd.isna(slope) or pd.isna(aspect) or pd.isna(heading):
        return np.nan
        
    angle_diff = calculate_angle_difference(aspect, heading)
    direction = -1 if angle_diff > 90 else 1
    return direction * slope * abs(np.cos(np.radians(angle_diff)))

def calculate_speed(effective_slope, landcover, prev_speed=None, prev_landcover=None):
    """根据有效坡度和土地覆盖类型计算速度
    
    Args:
        effective_slope: 有效坡度（度）
        landcover: 土地覆盖类型代码
        prev_speed: 前一个点的速度（用于平滑）
        prev_landcover: 前一个点的土地覆盖类型（用于处理过渡）
    
    Returns:
        float: 计算得到的速度（米/秒）
    """
    # 获取当前土地类型
    landcover_type = LANDCOVER_MAPPING.get(landcover)
    if landcover_type is None:
        return np.nan
        
    # 获取模型参数
    model = SPEED_MODELS[landcover_type]
    
    # 计算坡度影响
    slope_effect = (model['slope_effect']['uphill'] if effective_slope > 0 
                   else model['slope_effect']['downhill'])
    
    # 计算基础速度（考虑坡度影响）
    speed = model['base_speed'] + slope_effect * abs(effective_slope)
    
    # 添加随机扰动（减小扰动幅度）
    speed += np.random.normal(0, model['std_dev'] * 0.2)
    
    # 处理地类过渡
    if prev_speed is not None and prev_landcover is not None:
        prev_type = LANDCOVER_MAPPING.get(prev_landcover)
        if prev_type and prev_type != landcover_type:
            # 获取前一个地类的模型
            prev_model = SPEED_MODELS[prev_type]
            # 计算过渡权重
            curr_weight = model['transition_weight']
            prev_weight = prev_model['transition_weight']
            # 使用加权平均进行平滑
            speed = (curr_weight * speed + prev_weight * prev_speed) / (curr_weight + prev_weight)
    
    # 速度平滑（如果有前一个速度）
    if prev_speed is not None:
        # 限制速度变化率
        max_change = 0.5  # 每步最大变化率
        speed_change = speed - prev_speed
        if abs(speed_change) > max_change:
            speed = prev_speed + np.sign(speed_change) * max_change
        
        # 应用平滑
        speed = 0.8 * speed + 0.2 * prev_speed
    
    # 限制速度范围
    speed = np.clip(speed, model['min_speed'], model['max_speed'])
    
    return speed

def simulate_trajectory(traj_file, window_size=None):
    """模拟轨迹
    
    Args:
        traj_file: 轨迹文件路径
        window_size: 时间窗口大小（秒），用于聚合数据
    
    Returns:
        tuple: (原始DataFrame, 模拟DataFrame)
    """
    print(f"\n处理轨迹文件: {os.path.basename(traj_file)}")
    
    try:
        # 加载轨迹数据
        df = pd.read_csv(traj_file)
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
        print(f"  原始数据加载成功，共 {len(df)} 行")
        
        # 过滤开头10分钟和结尾3分钟的数据
        start_time = df['timestamp'].min() + pd.Timedelta(minutes=10)
        end_time = df['timestamp'].max() - pd.Timedelta(minutes=3)
        df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
        print(f"  过滤后数据量: {len(df)} 行")
        
        # 获取环境数据
        df['slope'] = get_env_data(df, SLOPE_FILE)
        df['aspect'] = get_env_data(df, ASPECT_FILE)
        df['landcover'] = get_env_data(df, LANDCOVER_FILE)
        
        print(f"  环境数据获取成功:")
        print(f"    坡度数据: {df['slope'].notna().sum()} 个有效值")
        print(f"    坡向数据: {df['aspect'].notna().sum()} 个有效值")
        print(f"    土地覆盖: {df['landcover'].notna().sum()} 个有效值")
        
        # 计算移动方向
        dlon = df['longitude'].diff()
        dlat = df['latitude'].diff()
        df['heading'] = np.degrees(np.arctan2(dlon * np.cos(np.radians(df['latitude'])), dlat)) % 360
        df.loc[df.index[0], 'heading'] = df.loc[df.index[1], 'heading'] if len(df) > 1 else np.nan
        
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
        
        # 模拟速度
        simulated_speeds = []
        prev_speed = None
        prev_landcover = None
        for _, row in df.iterrows():
            speed = calculate_speed(
                row['effective_slope'], 
                row['landcover'], 
                prev_speed,
                prev_landcover
            )
            simulated_speeds.append(speed)
            prev_speed = speed
            prev_landcover = row['landcover']
        
        df['simulated_speed'] = simulated_speeds
        print(f"  速度模拟成功: {df['simulated_speed'].notna().sum()} 个有效值")
        
        # 如果指定了时间窗口，进行聚合
        if window_size:
            # 创建时间窗口
            df['time_window'] = df['timestamp'].dt.floor(f'{window_size}S')
            
            # 聚合数据
            agg_df = df.groupby('time_window').agg({
                'simulated_speed': 'mean',
                'velocity_2d_ms': 'mean',
                'effective_slope': 'mean',
                'slope': 'mean',
                'aspect': 'mean',
                'landcover': lambda x: x.mode().iloc[0] if not x.empty else np.nan,
                'heading': 'mean'
            }).reset_index()
            
            # 重命名列
            agg_df = agg_df.rename(columns={'time_window': 'timestamp'})
            
            return df, agg_df
        
        return df, df.copy()
        
    except Exception as e:
        print(f"处理文件失败: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None

def calculate_metrics(df):
    """计算模拟速度与原始速度的评估指标
    
    Args:
        df: 包含原始速度和模拟速度的DataFrame
        
    Returns:
        dict: 包含各项评估指标的字典
    """
    metrics = {}
    
    # 去除无效值
    valid_data = df.dropna(subset=['velocity_2d_ms', 'simulated_speed'])
    
    if len(valid_data) < 2:
        return None
        
    # 计算相关系数
    correlation = valid_data['velocity_2d_ms'].corr(valid_data['simulated_speed'])
    metrics['correlation'] = correlation
    
    # 计算均方根误差(RMSE)
    rmse = np.sqrt(((valid_data['velocity_2d_ms'] - valid_data['simulated_speed']) ** 2).mean())
    metrics['rmse'] = rmse
    
    # 计算平均绝对误差(MAE)
    mae = abs(valid_data['velocity_2d_ms'] - valid_data['simulated_speed']).mean()
    metrics['mae'] = mae
    
    # 计算平均相对误差(MAPE)
    mape = (abs(valid_data['velocity_2d_ms'] - valid_data['simulated_speed']) / valid_data['velocity_2d_ms']).mean() * 100
    metrics['mape'] = mape
    
    # 计算速度统计信息
    metrics['orig_mean'] = valid_data['velocity_2d_ms'].mean()
    metrics['sim_mean'] = valid_data['simulated_speed'].mean()
    metrics['orig_std'] = valid_data['velocity_2d_ms'].std()
    metrics['sim_std'] = valid_data['simulated_speed'].std()
    
    return metrics

def analyze_speed_anomalies(df, window_size, output_prefix):
    """分析速度异常点的环境信息
    
    Args:
        df: 包含速度和环境数据的DataFrame
        window_size: 时间窗口大小
        output_prefix: 输出文件前缀
    """
    # 计算速度变化率
    df['speed_change'] = df['velocity_2d_ms'].diff()
    
    # 定义速度突变阈值（标准差的2倍）
    threshold = df['speed_change'].std() * 2
    
    # 找出速度突变点
    anomalies = df[abs(df['speed_change']) > threshold].copy()
    
    if len(anomalies) == 0:
        print("未发现显著的速度突变点")
        return
        
    # 创建多子图
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    # 1. 速度和环境变量时序图
    ax1 = plt.subplot(gs[0])
    # 绘制速度
    ax1.plot(df.index, df['velocity_2d_ms'], 
            label='原始速度', linewidth=1, alpha=0.7)
    ax1.plot(df.index, df['simulated_speed'], 
            label='模拟速度', linewidth=1, alpha=0.7)
    
    # 标记突变点
    ax1.scatter(anomalies.index, anomalies['velocity_2d_ms'], 
                color='red', s=100, zorder=5, label='速度突变点')
    
    ax1.set_title(f'速度和环境变量时序图（{window_size}秒窗口）')
    ax1.set_xlabel('时间点')
    ax1.set_ylabel('速度（米/秒）')
    ax1.grid(True)
    ax1.legend()
    
    # 2. 坡度和有效坡度
    ax2 = plt.subplot(gs[1])
    ax2.plot(df.index, df['slope'], 
            label='原始坡度', linewidth=1, alpha=0.7)
    ax2.plot(df.index, df['effective_slope'], 
            label='有效坡度', linewidth=1, alpha=0.7)
    ax2.scatter(anomalies.index, anomalies['slope'], 
                color='red', s=100, zorder=5)
    
    ax2.set_xlabel('时间点')
    ax2.set_ylabel('坡度（度）')
    ax2.grid(True)
    ax2.legend()
    
    # 3. 土地覆盖类型
    ax3 = plt.subplot(gs[2])
    ax3.plot(df.index, df['landcover'], 
            label='土地覆盖类型', linewidth=1, alpha=0.7)
    ax3.scatter(anomalies.index, anomalies['landcover'], 
                color='red', s=100, zorder=5)
    
    # 设置y轴标签为土地覆盖类型名称
    yticks = sorted(list(LANDCOVER_MAPPING.keys()))
    ax3.set_yticks(yticks)
    ax3.set_yticklabels([LANDCOVER_MAPPING.get(code, str(code)) for code in yticks])
    
    ax3.set_xlabel('时间点')
    ax3.set_ylabel('土地覆盖类型')
    ax3.grid(True)
    
    # 保存图片
    output_file = os.path.join(OUTPUT_DIR, f'{output_prefix}_{window_size}秒_环境分析.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    # 输出突变点的详细信息
    print(f"\n速度突变点分析（{len(anomalies)}个点）:")
    for idx, row in anomalies.iterrows():
        print(f"\n时间点: {idx}")
        print(f"  速度变化: {row['speed_change']:.2f} m/s")
        print(f"  原始速度: {row['velocity_2d_ms']:.2f} m/s")
        print(f"  模拟速度: {row['simulated_speed']:.2f} m/s")
        print(f"  原始坡度: {row['slope']:.2f}°")
        print(f"  有效坡度: {row['effective_slope']:.2f}°")
        print(f"  土地类型: {LANDCOVER_MAPPING.get(row['landcover'], row['landcover'])}")

def compare_with_original(df, window_size, output_prefix):
    """生成速度对比图并计算评估指标"""
    try:
        # 计算评估指标
        metrics = calculate_metrics(df)
        if metrics is None:
            print("  警告: 数据不足，无法计算评估指标")
            return
            
        # 生成基本的对比图和指标
        plt.figure(figsize=(15, 10))
        
        # 创建子图
        gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
        
        # 速度对比图
        ax1 = plt.subplot(gs[0])
        ax1.plot(df.index, df['velocity_2d_ms'], 
                label='原始速度', linewidth=1, alpha=0.7)
        ax1.plot(df.index, df['simulated_speed'], 
                label='模拟速度', linewidth=1, alpha=0.7)
        
        ax1.set_title(f'速度对比（{window_size}秒窗口）')
        ax1.set_xlabel('时间点')
        ax1.set_ylabel('速度（米/秒）')
        ax1.grid(True)
        ax1.legend()
        
        # 评估指标文本
        ax2 = plt.subplot(gs[1])
        ax2.axis('off')
        metrics_text = (
            f"评估指标:\n"
            f"相关系数 (Correlation): {metrics['correlation']:.3f}\n"
            f"均方根误差 (RMSE): {metrics['rmse']:.3f} m/s\n"
            f"平均绝对误差 (MAE): {metrics['mae']:.3f} m/s\n"
            f"平均相对误差 (MAPE): {metrics['mape']:.1f}%\n\n"
            f"速度统计:\n"
            f"原始速度: 均值 = {metrics['orig_mean']:.2f} m/s, 标准差 = {metrics['orig_std']:.2f} m/s\n"
            f"模拟速度: 均值 = {metrics['sim_mean']:.2f} m/s, 标准差 = {metrics['sim_std']:.2f} m/s"
        )
        ax2.text(0.1, 0.5, metrics_text, fontsize=12, va='center')
        
        # 保存图片
        output_file = os.path.join(OUTPUT_DIR, f'{output_prefix}_{window_size}秒_速度比较.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  {window_size}秒窗口评估结果:")
        print(f"    相关系数: {metrics['correlation']:.3f}")
        print(f"    均方根误差: {metrics['rmse']:.3f} m/s")
        print(f"    平均绝对误差: {metrics['mae']:.3f} m/s")
        print(f"    平均相对误差: {metrics['mape']:.1f}%")
        
    except Exception as e:
        print(f"  生成对比图时出错: {str(e)}")

def main():
    """主函数"""
    # 获取轨迹文件列表
    traj_files = []
    for root, dirs, files in os.walk('core_trajectories'):
        for file in files:
            if file.endswith('_core.csv'):
                traj_files.append(os.path.join(root, file))
    
    if not traj_files:
        print("错误: 未找到轨迹文件")
        return
    
    print(f"找到 {len(traj_files)} 个轨迹文件")
    
    # 处理每个轨迹文件
    for traj_file in traj_files:
        traj_id = os.path.splitext(os.path.basename(traj_file))[0]
        
        # 生成1秒、5秒、10秒和15秒的对比结果
        for window_size in [1, 5, 10, 15]:
            # 模拟轨迹
            orig_df, agg_df = simulate_trajectory(traj_file, window_size)
            if orig_df is None or agg_df is None:
                continue
            
            # 生成对比图
            compare_with_original(agg_df, window_size, traj_id)
            
            print(f"  {window_size}秒窗口处理完成")
        
        print(f"\n轨迹 {traj_id} 处理完成")
    
    print("\n所有轨迹处理完成")

if __name__ == "__main__":
    main() 