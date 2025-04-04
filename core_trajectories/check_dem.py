import rasterio
import numpy as np
import pandas as pd
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
import os

def get_trajectory_bounds():
    """获取所有轨迹的边界范围"""
    bounds = {
        'minx': float('inf'),
        'miny': float('inf'),
        'maxx': float('-inf'),
        'maxy': float('-inf')
    }
    
    # 读取所有轨迹文件
    csv_files = [f for f in os.listdir('.') if f.startswith('converted_sequence') and f.endswith('_core.csv')]
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        bounds['minx'] = min(bounds['minx'], df['longitude'].min())
        bounds['miny'] = min(bounds['miny'], df['latitude'].min())
        bounds['maxx'] = max(bounds['maxx'], df['longitude'].max())
        bounds['maxy'] = max(bounds['maxy'], df['latitude'].max())
    
    # 添加缓冲区
    buffer = 1000  # 1000米的缓冲区
    bounds['minx'] -= buffer
    bounds['miny'] -= buffer
    bounds['maxx'] += buffer
    bounds['maxy'] += buffer
    
    return bounds

def check_dem(dem_path):
    """检查DEM数据的信息"""
    with rasterio.open(dem_path) as src:
        print("\nDEM数据信息:")
        print(f"坐标系统: {src.crs}")
        print(f"变换矩阵: {src.transform}")
        print(f"数据范围: {src.bounds}")
        print(f"像素大小: {src.res}")
        print(f"数据维度: {src.shape}")
        
        # 读取一些高程值样本
        data = src.read(1)
        print(f"\n高程值统计:")
        print(f"最小值: {np.min(data)}")
        print(f"最大值: {np.max(data)}")
        print(f"平均值: {np.mean(data)}")
        print(f"标准差: {np.std(data)}")

def main():
    # DEM文件路径
    dem_path = '../dem_transformed.tif'
    
    # 检查DEM数据
    print("正在检查DEM数据...")
    check_dem(dem_path)
    
    # 获取轨迹范围
    print("\n正在计算轨迹范围...")
    bounds = get_trajectory_bounds()
    print(f"轨迹范围（带1000米缓冲区）:")
    print(f"X范围: {bounds['minx']} 到 {bounds['maxx']}")
    print(f"Y范围: {bounds['miny']} 到 {bounds['maxy']}")

if __name__ == "__main__":
    main() 