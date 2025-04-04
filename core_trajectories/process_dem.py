import rasterio
import numpy as np
import pandas as pd
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
from rasterio.transform import from_origin
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

def process_dem(input_dem, output_dem, bounds):
    """处理DEM数据：重新投影并裁剪到指定范围"""
    with rasterio.open(input_dem) as src:
        # 计算新的变换矩阵
        width = int((bounds['maxx'] - bounds['minx']) / 30)  # 30米分辨率
        height = int((bounds['maxy'] - bounds['miny']) / 30)
        
        # 创建新的变换矩阵，使用实际的UTM坐标
        new_transform = from_origin(bounds['minx'], bounds['maxy'], 30, 30)
        
        # 创建新的DEM文件
        new_profile = src.profile.copy()
        new_profile.update({
            'height': height,
            'width': width,
            'transform': new_transform,
            'nodata': -9999
        })
        
        print(f"\n新DEM文件信息:")
        print(f"尺寸: {width} x {height}")
        print(f"分辨率: 30米")
        print(f"范围: {bounds}")
        
        with rasterio.open(output_dem, 'w', **new_profile) as dst:
            # 重投影和重采样数据
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs='EPSG:32630',
                resampling=Resampling.bilinear
            )

def main():
    # 获取轨迹范围
    print("正在计算轨迹范围...")
    bounds = get_trajectory_bounds()
    print(f"轨迹范围（带1000米缓冲区）:")
    print(f"X范围: {bounds['minx']} 到 {bounds['maxx']}")
    print(f"Y范围: {bounds['miny']} 到 {bounds['maxy']}")
    
    # 处理DEM数据
    input_dem = '../dem_transformed.tif'
    output_dem = '../dem_aligned.tif'
    
    print("\n正在处理DEM数据...")
    process_dem(input_dem, output_dem, bounds)
    print(f"\nDEM数据处理完成，已保存至: {output_dem}")

if __name__ == "__main__":
    main() 