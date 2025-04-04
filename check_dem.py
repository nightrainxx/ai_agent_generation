import rasterio
import numpy as np

def check_dem(filename):
    with rasterio.open(filename) as src:
        data = src.read(1)
        valid = data[data != -9999]
        print(f"文件: {filename}")
        print(f"数据形状: {data.shape}")
        print(f"有效数据点数: {len(valid)}")
        if len(valid) > 0:
            print(f"有效数据范围: {np.min(valid)} 到 {np.max(valid)}")
            print(f"平均值: {np.mean(valid)}")
        else:
            print("没有有效数据")

if __name__ == "__main__":
    # 检查原始DEM
    print("\n检查原始DEM:")
    check_dem("dem_transformed.tif")
    
    # 检查转换后的DEM
    print("\n检查转换后的DEM:")
    check_dem("dem_aligned.tif") 