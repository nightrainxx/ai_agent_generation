import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch
import rasterio
import numpy as np
from rasterio.mask import mask
import matplotlib.colors as colors

def get_center_point(dfs):
    """
    获取所有轨迹的中心点
    """
    all_x = []
    all_y = []
    for df in dfs:
        all_x.extend(df['longitude'].values)
        all_y.extend(df['latitude'].values)
    return np.mean(all_x), np.mean(all_y)

def normalize_coordinates(df, center_x, center_y):
    """
    将坐标归一化到中心点周围
    """
    df = df.copy()
    df['longitude'] = df['longitude'] - center_x
    df['latitude'] = df['latitude'] - center_y
    return df

def check_coordinates(df):
    """
    检查坐标范围，判断是否为经纬度或投影坐标
    """
    lon_range = (df['longitude'].min(), df['longitude'].max())
    lat_range = (df['latitude'].min(), df['latitude'].max())
    print(f"X坐标范围: {lon_range}")
    print(f"Y坐标范围: {lat_range}")
    return lon_range, lat_range

def plot_trajectory(csv_file, ax, color, label):
    """
    在给定的axes上绘制一条轨迹
    
    参数:
        csv_file: CSV文件路径
        ax: matplotlib axes对象
        color: 轨迹颜色
        label: 轨迹标签
    """
    df = pd.read_csv(csv_file)
    print(f"\n检查轨迹文件 {label} 的坐标范围:")
    print(f"X范围: ({df['longitude'].min()}, {df['longitude'].max()})")
    print(f"Y范围: ({df['latitude'].min()}, {df['latitude'].max()})")
    
    # 绘制轨迹线
    ax.plot(df['longitude'], df['latitude'], color=color, alpha=1.0, linewidth=2, label=label)
    
    # 标记起点和终点
    ax.scatter(df['longitude'].iloc[0], df['latitude'].iloc[0], color=color, marker='o', s=100, label=f'{label} 起点')
    ax.scatter(df['longitude'].iloc[-1], df['latitude'].iloc[-1], color=color, marker='s', s=100, label=f'{label} 终点')

def plot_dem_background(ax, dem_path):
    """
    在给定的axes上绘制DEM数据作为背景
    
    参数:
        ax: matplotlib axes对象
        dem_path: DEM文件路径
    """
    print(f"\n正在读取DEM文件: {dem_path}")
    with rasterio.open(dem_path) as src:
        # 读取DEM数据
        dem_data = src.read(1)
        
        # 获取坐标信息
        bounds = src.bounds
        print(f"DEM边界范围:")
        print(f"X范围: {bounds.left} 到 {bounds.right}")
        print(f"Y范围: {bounds.bottom} 到 {bounds.top}")
        
        # 打印DEM数据的基本信息
        print(f"\nDEM数据统计:")
        print(f"形状: {dem_data.shape}")
        print(f"数据类型: {dem_data.dtype}")
        
        # 处理无效值
        valid_data = dem_data[dem_data != -9999]
        if len(valid_data) > 0:
            print(f"有效数据范围: {np.min(valid_data)} 到 {np.max(valid_data)}")
            print(f"平均值: {np.mean(valid_data)}")
            
            # 使用2-98百分位数作为显示范围，避免极值影响
            vmin, vmax = np.percentile(valid_data, [2, 98])
            print(f"显示范围 (2-98%分位数): {vmin} 到 {vmax}")
        else:
            print("警告: 没有找到有效数据!")
            vmin, vmax = 0, 1
        
        # 设置显示范围
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        
        # 创建地形颜色映射
        terrain_colors = ['#014B15', '#236E2B', '#3D8C40', '#6FB663', '#8FCC80', '#ABC99C',
                         '#CCD4B8', '#E1D7C4', '#E8DBBA', '#EFE0AF', '#F2E0A3', '#F5E096']
        terrain_cmap = colors.LinearSegmentedColormap.from_list('terrain', terrain_colors, N=256)
        
        # 创建掩码数组
        dem_masked = np.ma.masked_where(dem_data == -9999, dem_data)
        
        # 绘制DEM数据
        im = ax.imshow(dem_masked, extent=extent, cmap=terrain_cmap, alpha=0.7,
                      aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('高程 (米)', fontsize=10)
        
        return extent

def main():
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # DEM文件路径
    dem_path = '../dem_aligned.tif'
    print(f"DEM文件路径: {dem_path}")
    
    # 检查DEM文件是否存在
    if not os.path.exists(dem_path):
        print(f"错误: DEM文件不存在: {dem_path}")
        return
    
    # 绘制DEM背景
    extent = plot_dem_background(ax, dem_path)
    
    # 颜色列表 - 使用醒目的颜色
    colors = ['#FF3333', '#FFD700', '#00FFFF', '#FF69B4']
    
    # 获取所有转换后的轨迹文件
    input_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = sorted([f for f in os.listdir(input_dir) if f.startswith('converted_sequence') and f.endswith('_core.csv')])
    
    print(f"\n找到{len(csv_files)}个轨迹文件")
    
    # 绘制每条轨迹
    for i, csv_file in enumerate(csv_files):
        file_path = os.path.join(input_dir, csv_file)
        sequence_num = csv_file.split('_')[2]  # 获取序列号
        print(f"\n正在处理轨迹文件: {file_path}")
        plot_trajectory(file_path, ax, colors[i], f'轨迹 {sequence_num}')
    
    # 设置图形属性
    ax.set_title('核心轨迹与地形叠加图 (EPSG:32630)', fontsize=16, pad=20)
    ax.set_xlabel('X坐标 (米)', fontsize=12)
    ax.set_ylabel('Y坐标 (米)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 添加图例
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    output_path = os.path.join(input_dir, 'trajectories_with_dem_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n可视化图像已保存至: {output_path}")
    
    # 关闭图形
    plt.close()

if __name__ == "__main__":
    main() 