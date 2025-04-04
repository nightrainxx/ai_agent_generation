import rasterio
import os
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS

def check_tif(tif_path):
    """检查TIF文件的坐标系统和其他信息"""
    with rasterio.open(tif_path) as src:
        print(f"\n检查文件: {tif_path}")
        print(f"坐标系统: {src.crs}")
        print(f"变换矩阵: {src.transform}")
        print(f"数据范围: {src.bounds}")
        print(f"像素大小: {src.res}")
        print(f"数据维度: {src.shape}")
        
        # 读取数据统计信息
        data = src.read(1)
        valid_data = data[data != src.nodata] if src.nodata is not None else data
        if len(valid_data) > 0:
            print(f"\n有效数据统计:")
            print(f"最小值: {np.min(valid_data)}")
            print(f"最大值: {np.max(valid_data)}")
            print(f"平均值: {np.mean(valid_data)}")
            print(f"标准差: {np.std(valid_data)}")
        
        return src.crs

def convert_to_utm(input_tif, output_tif):
    """将TIF文件转换到UTM坐标系统(EPSG:32630)"""
    print(f"\n正在转换文件: {input_tif} -> {output_tif}")
    
    with rasterio.open(input_tif) as src:
        # 定义目标坐标系
        dst_crs = CRS.from_epsg(32630)
        
        # 计算新的变换矩阵和尺寸
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        
        # 更新输出配置
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # 创建输出文件
        with rasterio.open(output_tif, 'w', **kwargs) as dst:
            # 重投影数据
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear
                )
    
    print("转换完成!")

def main():
    # 获取所有TIF文件
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    tif_files = []
    for root, dirs, files in os.walk(workspace_dir):
        for file in files:
            if file.endswith('.tif'):
                tif_files.append(os.path.join(root, file))
    
    print(f"找到 {len(tif_files)} 个TIF文件")
    
    # 检查每个文件
    for tif_file in tif_files:
        current_crs = check_tif(tif_file)
        
        # 如果不是UTM坐标系，进行转换
        if current_crs is None or current_crs.to_epsg() != 32630:
            print(f"\n{tif_file} 不是UTM坐标系(EPSG:32630)，需要转换")
            output_tif = tif_file.replace('.tif', '_utm.tif')
            convert_to_utm(tif_file, output_tif)
            
            # 检查转换后的文件
            print("\n检查转换后的文件:")
            check_tif(output_tif)
        else:
            print(f"\n{tif_file} 已经是UTM坐标系(EPSG:32630)，无需转换")

if __name__ == "__main__":
    main() 