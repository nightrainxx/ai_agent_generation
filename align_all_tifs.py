import rasterio
import numpy as np
import pandas as pd
import subprocess
import os
from osgeo import gdal, osr
import glob

def get_trajectory_bounds():
    """计算所有轨迹文件的坐标范围"""
    x_min = y_min = float('inf')
    x_max = y_max = float('-inf')
    
    # 遍历所有轨迹文件
    for file in glob.glob('core_trajectories/converted_sequence_*_core.csv'):
        df = pd.read_csv(file)
        x_min = min(x_min, df['longitude'].min())
        x_max = max(x_max, df['longitude'].max())
        y_min = min(y_min, df['latitude'].min())
        y_max = max(y_max, df['latitude'].max())
    
    # 添加5000米的缓冲区，确保有足够的上下文信息
    buffer = 5000
    return {
        'minx': x_min - buffer,
        'maxx': x_max + buffer,
        'miny': y_min - buffer,
        'maxy': y_max + buffer
    }

def align_raster(input_tif, output_tif, bounds):
    """将栅格数据对齐到指定的UTM坐标范围"""
    print(f"\n正在处理: {input_tif} -> {output_tif}")
    
    # 获取输入文件的数据类型和数据
    with rasterio.open(input_tif) as src:
        data = src.read(1)
        profile = src.profile
        dtype = src.dtypes[0]
        nodata = -9999 if dtype != 'uint8' else 255
        
        # 获取原始数据的范围
        width = src.width
        height = src.height
        transform = src.transform
        
        # 处理极端值
        valid_mask = (data > -1000) & (data < 10000)
        data[~valid_mask] = nodata
        
        # 计算中心点的UTM坐标（使用轨迹范围的中心点）
        center_x = (bounds['minx'] + bounds['maxx']) / 2
        center_y = (bounds['miny'] + bounds['maxy']) / 2
        
        # 计算原始数据的范围（以中心点为原点）
        half_width = width * abs(transform[0]) / 2
        half_height = height * abs(transform[4]) / 2
        
        print(f"原始数据大小: {width}x{height}")
        print(f"原始数据变换矩阵: {transform}")
        print(f"中心点UTM坐标: ({center_x}, {center_y})")
        print(f"数据范围: {half_width}x{half_height}米")
        print(f"目标范围: {bounds}")
        
        # 创建临时文件保存处理后的数据
        temp_tif = input_tif.replace('.tif', '_temp.tif')
        profile.update({
            'nodata': nodata,
            'compress': 'LZW'
        })
        with rasterio.open(temp_tif, 'w', **profile) as dst:
            dst.write(data, 1)
    
    # 使用gdal_translate直接设置正确的地理参考
    translate_cmd = [
        'gdal_translate',
        '-a_srs', 'EPSG:32630',
        '-a_ullr', 
        str(center_x - half_width), str(center_y + half_height),
        str(center_x + half_width), str(center_y - half_height),
        '-a_nodata', str(nodata),
        '-co', 'COMPRESS=LZW',
        temp_tif,
        output_tif
    ]
    
    print(f"执行translate命令: {' '.join(translate_cmd)}")
    subprocess.run(translate_cmd, check=True)
    
    # 删除临时文件
    os.remove(temp_tif)
    
    # 验证输出
    with rasterio.open(output_tif) as dst:
        data = dst.read(1)
        print(f"\n输出数据统计:")
        print(f"形状: {data.shape}")
        print(f"数据类型: {dst.dtypes[0]}")
        print(f"边界: {dst.bounds}")
        
        # 计算有效数据统计
        valid_data = data[data != nodata]
        print(f"有效数据点数: {len(valid_data)}")
        if len(valid_data) > 0:
            print(f"有效数据范围: {np.min(valid_data)} 到 {np.max(valid_data)}")
            print(f"平均值: {np.mean(valid_data)}")

def main():
    # 获取轨迹范围
    print("正在计算轨迹范围...")
    bounds = get_trajectory_bounds()
    print(f"轨迹范围（带5000米缓冲区）:")
    print(f"X范围: {bounds['minx']} 到 {bounds['maxx']}")
    print(f"Y范围: {bounds['miny']} 到 {bounds['maxy']}")
    
    # 基础目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 要处理的文件列表（使用绝对路径）
    files_to_process = [
        (os.path.join(base_dir, 'dem_transformed.tif'), os.path.join(base_dir, 'dem_aligned.tif')),
        (os.path.join(base_dir, 'aspect_transformed.tif'), os.path.join(base_dir, 'aspect_aligned.tif')),
        (os.path.join(base_dir, 'slope_transformed.tif'), os.path.join(base_dir, 'slope_aligned.tif')),
        (os.path.join(base_dir, 'landcover_transformed.tif'), os.path.join(base_dir, 'landcover_aligned.tif'))
    ]
    
    # 处理每个文件
    for input_file, output_file in files_to_process:
        if os.path.exists(input_file):
            align_raster(input_file, output_file, bounds)
            print(f"完成: {output_file}")
        else:
            print(f"跳过: {input_file} (文件不存在)")

if __name__ == "__main__":
    main() 