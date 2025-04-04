import rasterio
import numpy as np

def check_dem(dem_file):
    """检查DEM文件的数据内容"""
    print(f"检查文件: {dem_file}")
    
    with rasterio.open(dem_file) as src:
        # 读取数据
        data = src.read(1)
        
        # 基本信息
        print(f"\n基本信息:")
        print(f"形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"坐标系统: {src.crs}")
        print(f"变换矩阵: {src.transform}")
        
        # 所有数据的统计
        print(f"\n所有数据统计:")
        print(f"最小值: {np.min(data)}")
        print(f"最大值: {np.max(data)}")
        print(f"平均值: {np.mean(data)}")
        print(f"标准差: {np.std(data)}")
        
        # 有效数据的统计
        valid_data = data[data != -9999]
        print(f"\n有效数据统计:")
        print(f"有效数据点数: {len(valid_data)}")
        if len(valid_data) > 0:
            print(f"最小值: {np.min(valid_data)}")
            print(f"最大值: {np.max(valid_data)}")
            print(f"平均值: {np.mean(valid_data)}")
            print(f"标准差: {np.std(valid_data)}")
            
            # 打印一些非无效值的坐标
            valid_coords = np.where(data != -9999)
            print(f"\n一些有效数据点的位置和值:")
            for i, j in zip(valid_coords[0][:5], valid_coords[1][:5]):
                print(f"位置 ({i}, {j}): {data[i, j]}")

if __name__ == "__main__":
    # 检查原始DEM
    print("\n检查原始DEM文件:")
    check_dem("dem_transformed.tif")
    
    # 检查对齐后的DEM
    print("\n\n检查对齐后的DEM文件:")
    check_dem("dem_aligned.tif") 