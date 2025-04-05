"""
轨迹生成器主程序

本程序用于:
1. 加载原始轨迹数据
2. 加载环境地图数据
3. 提取轨迹关键航点
4. 生成新轨迹
5. 验证生成效果
6. 保存结果

使用方法:
1. 将原始轨迹数据(.csv)放入data/trajectories目录
2. 将环境数据(.tif)放入data/environment目录
3. 运行本程序
4. 结果将保存在results目录
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any
from src.validation import TrajectoryValidator
from src.environment import EnvironmentMaps

def load_trajectory(file_path: str) -> pd.DataFrame:
    """加载轨迹数据
    
    Args:
        file_path: 轨迹文件路径
        
    Returns:
        pd.DataFrame: 轨迹数据
    """
    df = pd.read_csv(file_path)
    
    # 确保必要的列存在
    required_cols = [
        'timestamp_ms',
        'latitude',  # UTM-Y坐标
        'longitude', # UTM-X坐标
        'velocity_north_ms',
        'velocity_east_ms',
        'heading_deg'
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要的列: {col}")
    
    # 将latitude和longitude列重命名为x和y
    df = df.rename(columns={
        'longitude': 'x',  # UTM-X坐标
        'latitude': 'y'    # UTM-Y坐标
    })
            
    return df

def main():
    """主函数"""
    # 1. 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    env_dir = os.path.join(data_dir, "environment")
    traj_dir = os.path.join(data_dir, "trajectories")
    results_dir = os.path.join(base_dir, "results")
    
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    print("加载环境数据...")
    # 2. 加载环境地图
    env_maps = EnvironmentMaps(env_dir)
    
    print("创建验证器...")
    # 3. 创建验证器
    validator = TrajectoryValidator(env_maps)
    
    print("加载并处理轨迹...")
    # 4. 加载并处理原始轨迹
    trajectories = []
    for i in range(1, 5):  # 处理4条轨迹
        try:
            # 加载轨迹
            traj_path = os.path.join(traj_dir, f"trajectory_{i}.csv")
            print(f"处理轨迹 {i}: {traj_path}")
            
            df = load_trajectory(traj_path)
            trajectories.append((i, df))
            
        except Exception as e:
            print(f"处理轨迹{i}时出错: {e}")
            continue
    
    print("\n开始验证...")
    # 5. 验证每条轨迹
    for traj_id, original_df in trajectories:
        print(f"\n处理轨迹 {traj_id}")
        
        try:
            # 生成对比轨迹
            save_path = os.path.join(
                results_dir,
                f"trajectory_{traj_id}_comparison.png"
            )
            generated_df, metrics = validator.validate(
                original_df=original_df,
                goal_id=0,
                output_path=save_path,
                visualize=True
            )
            
            # 打印统计指标
            print("\n统计指标:")
            for key, value in metrics.items():
                print(f"{key}: {value:.3f}")
            
            # 保存生成的轨迹
            output_path = os.path.join(
                results_dir,
                f"trajectory_{traj_id}_generated.csv"
            )
            generated_df.to_csv(output_path, index=False)
            
            print(f"结果已保存到 {results_dir}")
            
        except Exception as e:
            print(f"验证轨迹{traj_id}时出错: {e}")
            continue

if __name__ == '__main__':
    main() 