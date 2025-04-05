"""环境地图模块

本模块提供了环境地图的加载和查询功能，包括：
1. 加载DEM、坡度、坡向和土地覆盖类型数据
2. 提供基于UTM坐标的环境特征查询
3. 确保所有栅格数据对齐

输入:
- dem_path: str - DEM数据路径(.tif)
- slope_path: str - 坡度数据路径(.tif)
- aspect_path: str - 坡向数据路径(.tif)
- landcover_path: str - 土地覆盖类型数据路径(.tif)

输出:
- 环境特征查询结果: Dict[str, float/int]
"""

import os
import numpy as np
import rasterio
from typing import Dict, Any, Tuple

class EnvironmentMaps:
    """环境地图类"""
    
    def __init__(self, env_data_dir: str):
        """初始化环境地图
        
        Args:
            env_data_dir: 环境数据目录路径
        """
        print("\n加载环境地图...")
        
        # 构建数据文件路径
        self.dem_path = os.path.join(env_data_dir, 'dem_aligned.tif')
        self.slope_path = os.path.join(env_data_dir, 'slope_aligned.tif')
        self.aspect_path = os.path.join(env_data_dir, 'aspect_aligned.tif')
        self.landcover_path = os.path.join(env_data_dir, 'landcover_aligned.tif')
        
        # 检查文件是否存在
        for path in [self.dem_path, self.slope_path, 
                    self.aspect_path, self.landcover_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"找不到环境数据文件: {path}")
        
        # 加载数据
        self.dem_ds = rasterio.open(self.dem_path)
        self.slope_ds = rasterio.open(self.slope_path)
        self.aspect_ds = rasterio.open(self.aspect_path)
        self.landcover_ds = rasterio.open(self.landcover_path)
        
        # 检查栅格对齐
        self._check_alignment()
        
        # 获取地图范围
        self.bounds = self.dem_ds.bounds
        print("环境地图加载完成")
        print(f"地图范围: {self.bounds}")
        
    def _check_alignment(self) -> None:
        """检查所有栅格数据是否对齐"""
        # 获取参考变换矩阵
        ref_transform = self.dem_ds.transform
        ref_shape = self.dem_ds.shape
        
        # 检查其他数据集
        for ds, name in [
            (self.slope_ds, '坡度'),
            (self.aspect_ds, '坡向'),
            (self.landcover_ds, '土地覆盖')
        ]:
            if ds.transform != ref_transform:
                raise ValueError(f"{name}数据的变换矩阵与DEM不一致")
            if ds.shape != ref_shape:
                raise ValueError(f"{name}数据的形状与DEM不一致")
    
    def _xy_to_rowcol(self, x: float, y: float) -> Tuple[int, int]:
        """将UTM坐标转换为栅格行列号
        
        Args:
            x: UTM东向坐标
            y: UTM北向坐标
            
        Returns:
            Tuple[int, int]: (行号, 列号)
        """
        row, col = rasterio.transform.rowcol(self.dem_ds.transform, x, y)
        return row, col
    
    def query_by_xy(self, x: float, y: float) -> Dict[str, Any]:
        """根据UTM坐标查询环境特征
        
        Args:
            x: UTM东向坐标
            y: UTM北向坐标
            
        Returns:
            Dict[str, Any]: 环境特征字典
        """
        # 1. 检查坐标是否在范围内
        if not (self.bounds.left <= x <= self.bounds.right and
                self.bounds.bottom <= y <= self.bounds.top):
            print(f"警告: 坐标({x}, {y})超出地图范围")
            return {
                'dem': 0.0,
                'slope_magnitude': 0.0,
                'slope_aspect': 0.0,
                'landcover': 90  # 其他类型
            }
        
        # 2. 转换为栅格索引
        row, col = self._xy_to_rowcol(x, y)
        
        # 3. 读取各层数据
        try:
            dem = self.dem_ds.read(1, window=((row, row+1), (col, col+1)))[0,0]
            slope = self.slope_ds.read(1, window=((row, row+1), (col, col+1)))[0,0]
            aspect = self.aspect_ds.read(1, window=((row, row+1), (col, col+1)))[0,0]
            landcover = self.landcover_ds.read(1, window=((row, row+1), (col, col+1)))[0,0]
            
            return {
                'dem': float(dem),
                'slope_magnitude': float(slope),
                'slope_aspect': float(aspect),
                'landcover': int(landcover)
            }
            
        except Exception as e:
            print(f"警告: 读取栅格数据时出错: {e}")
            return {
                'dem': 0.0,
                'slope_magnitude': 0.0,
                'slope_aspect': 0.0,
                'landcover': 90
            }
    
    def get_utm_bounds(self) -> Dict[str, float]:
        """获取地图的UTM边界范围
        
        Returns:
            Dict[str, float]: 边界范围字典
        """
        return {
            'min_x': self.bounds.left,
            'max_x': self.bounds.right,
            'min_y': self.bounds.bottom,
            'max_y': self.bounds.top
        }
        
    def __del__(self):
        """关闭数据集"""
        try:
            self.dem_ds.close()
            self.slope_ds.close()
            self.aspect_ds.close()
            self.landcover_ds.close()
        except:
            pass 