"""轨迹生成器验证模块

本模块用于：
1. 从原始轨迹中提取路径骨架
2. 使用提取的骨架生成新轨迹
3. 对比生成轨迹和原始轨迹的统计特性
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from src.generator import TrajectoryGenerator

class TrajectoryValidator:
    """轨迹验证器类"""
    
    def __init__(self, env_maps: Any):
        """初始化验证器
        
        Args:
            env_maps: 环境地图对象
        """
        self.generator = TrajectoryGenerator()
        self.env_maps = env_maps
        
    def extract_waypoints(
        self,
        trajectory_df: pd.DataFrame,
        min_distance: float = 50.0,  # 最小航点间距(米)
        max_angle: float = 30.0,  # 最大转角(度)
        visualize: bool = True,  # 是否可视化
        save_path: str = None  # 可视化结果保存路径
    ) -> List[Tuple[float, float]]:
        """从轨迹中提取关键航点
        
        使用Douglas-Peucker算法的变体，考虑:
        1. 点间最小距离
        2. 转角阈值
        3. 速度变化
        
        Args:
            trajectory_df: 原始轨迹数据
            min_distance: 最小航点间距(米)
            max_angle: 最大转角阈值(度)
            visualize: 是否可视化过程
            save_path: 可视化结果保存路径
            
        Returns:
            List[Tuple[float, float]]: 提取的航点序列(UTM坐标)
        """
        # 1. 获取UTM坐标序列
        positions = np.column_stack([
            trajectory_df['x'].values,  # UTM-X坐标
            trajectory_df['y'].values   # UTM-Y坐标
        ])
        
        print(f"坐标范围检查:")
        print(f"utm_x范围: {trajectory_df['x'].min():.2f} - {trajectory_df['x'].max():.2f}")
        print(f"utm_y范围: {trajectory_df['y'].min():.2f} - {trajectory_df['y'].max():.2f}")
        
        # 2. 初始化航点列表
        waypoints = [(positions[0][0], positions[0][1])]  # 起点
        last_idx = 0
        
        print(f"\n第一个航点: ({positions[0][0]:.2f}, {positions[0][1]:.2f})")
        
        # 用于可视化的列表
        all_angles = []  # 记录所有点的转角
        all_distances = []  # 记录所有点的距离
        waypoint_indices = [0]  # 记录航点的索引
        
        # 3. 遍历寻找关键点
        for i in range(1, len(positions)-1):
            # 计算与上一个航点的距离
            current_pos = positions[i]
            last_pos = positions[last_idx]
            
            if i % 1000 == 0:  # 每1000个点打印一次
                print(f"\n处理第{i}个点:")
                print(f"当前点: ({current_pos[0]:.2f}, {current_pos[1]:.2f})")
                print(f"上一航点: ({last_pos[0]:.2f}, {last_pos[1]:.2f})")
            
            dist = np.linalg.norm(current_pos - last_pos)
            all_distances.append(dist)
            
            if dist >= min_distance:
                # 计算转角
                vec1 = current_pos - last_pos
                vec2 = positions[i+1] - current_pos
                
                if i % 1000 == 0:  # 每1000个点打印一次
                    print(f"距离: {dist:.2f}m")
                    print(f"向量1: ({vec1[0]:.2f}, {vec1[1]:.2f})")
                    print(f"向量2: ({vec2[0]:.2f}, {vec2[1]:.2f})")
                
                # 检查向量是否为零向量
                if np.all(vec1 == 0) or np.all(vec2 == 0):
                    if i % 1000 == 0:
                        print("警告: 检测到零向量，跳过此点")
                    angle = 0
                else:
                    angle = np.degrees(
                        np.arccos(
                            np.clip(
                                np.dot(vec1, vec2) / 
                                (np.linalg.norm(vec1) * np.linalg.norm(vec2)),
                                -1.0, 1.0
                            )
                        )
                    )
                
                all_angles.append(angle)
                
                if i % 1000 == 0:
                    print(f"转角: {angle:.2f}度")
                
                # 如果转角大于阈值，添加为航点
                if angle > max_angle:
                    waypoints.append((current_pos[0], current_pos[1]))
                    waypoint_indices.append(i)
                    last_idx = i
                    if i % 1000 == 0:
                        print("添加为新航点")
            else:
                all_angles.append(0)
                    
        # 4. 添加终点
        waypoints.append((
            positions[-1][0],
            positions[-1][1]
        ))
        waypoint_indices.append(len(positions)-1)
        
        print(f"\n提取的航点总数: {len(waypoints)}")
        
        # 5. 可视化过程
        if visualize:
            self._visualize_waypoint_extraction(
                positions,
                waypoints,
                waypoint_indices,
                all_angles,
                all_distances,
                min_distance,
                max_angle,
                save_path
            )
        
        return waypoints
        
    def _visualize_waypoint_extraction(
        self,
        positions: np.ndarray,
        waypoints: List[Tuple[float, float]],
        waypoint_indices: List[int],
        angles: List[float],
        distances: List[float],
        min_distance: float,
        max_angle: float,
        save_path: str = None
    ) -> None:
        """可视化航点提取过程
        
        Args:
            positions: 所有轨迹点的坐标
            waypoints: 提取的航点
            waypoint_indices: 航点在原始轨迹中的索引
            angles: 所有点的转角
            distances: 所有点的距离
            min_distance: 最小航点间距
            max_angle: 最大转角阈值
            save_path: 保存路径
        """
        plt.figure(figsize=(20, 15))
        
        # 1. 轨迹和航点图
        plt.subplot(2, 1, 1)
        
        # 绘制原始轨迹
        plt.plot(
            positions[:, 0],
            positions[:, 1],
            'b-',
            alpha=0.5,
            linewidth=1,
            label='原始轨迹'
        )
        
        # 绘制航点
        waypoints = np.array(waypoints)
        plt.scatter(
            waypoints[:, 0],
            waypoints[:, 1],
            c='r',
            s=100,
            label='提取的航点'
        )
        
        # 连接航点
        plt.plot(
            waypoints[:, 0],
            waypoints[:, 1],
            'g--',
            alpha=0.7,
            linewidth=2,
            label='航点连线'
        )
        
        # 标注航点编号
        for i, (x, y) in enumerate(waypoints):
            plt.annotate(
                f'P{i}',
                (x, y),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=12
            )
        
        plt.title(
            f'轨迹航点提取结果 (共{len(waypoints)}个航点)',
            fontsize=16,
            pad=20
        )
        plt.xlabel('UTM X (m)', fontsize=14)
        plt.ylabel('UTM Y (m)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        # 2. 转角和距离分析图
        plt.subplot(2, 1, 2)
        
        # 转角
        plt.plot(
            angles,
            'b-',
            alpha=0.5,
            label='转角',
            linewidth=1
        )
        plt.axhline(
            y=max_angle,
            color='r',
            linestyle='--',
            label=f'转角阈值 ({max_angle}°)'
        )
        
        # 标记航点位置
        plt.scatter(
            waypoint_indices[1:-1],  # 不包括起点和终点
            [angles[i-1] for i in waypoint_indices[1:-1]],
            c='r',
            s=100,
            label='航点位置'
        )
        
        plt.title('转角分析', fontsize=20, pad=20)
        plt.xlabel('轨迹点索引', fontsize=18)
        plt.ylabel('转角 (度)', fontsize=18)
        plt.legend(fontsize=16)
        plt.grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    def get_initial_state(
        self,
        trajectory_df: pd.DataFrame
    ) -> Dict:
        """从轨迹起点获取初始状态
        
        Args:
            trajectory_df: 原始轨迹数据
            
        Returns:
            Dict: 初始状态字典
        """
        return {
            'x0': trajectory_df['x'].iloc[0],  # UTM-X坐标
            'y0': trajectory_df['y'].iloc[0],  # UTM-Y坐标
            'vx0': trajectory_df['velocity_east_ms'].iloc[0],
            'vy0': trajectory_df['velocity_north_ms'].iloc[0],
            'heading0': trajectory_df['heading_deg'].iloc[0]
        }
    
    def generate_and_compare(
        self,
        original_df: pd.DataFrame,
        goal_id: int,
        generation_rules: Dict = None,
        control_params: Dict = None,
        sim_params: Dict = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """生成对比轨迹并计算统计指标
        
        Args:
            original_df: 原始轨迹数据
            goal_id: 目标点ID
            generation_rules: 生成规则参数(可选)
            control_params: 控制参数(可选)
            sim_params: 仿真参数(可选)
            
        Returns:
            Tuple[pd.DataFrame, Dict]: (生成的轨迹, 统计指标)
        """
        # 1. 提取航点
        waypoints = self.extract_waypoints(original_df)
        
        # 2. 获取初始状态
        initial_state = self.get_initial_state(original_df)
        
        # 3. 使用默认参数
        if generation_rules is None:
            generation_rules = {
                'base_speeds': self.generator.DEFAULT_BASE_SPEEDS,
                'slope_coefficients': self.generator.DEFAULT_SLOPE_COEFFICIENTS
            }
            
        if control_params is None:
            control_params = {
                'global_speed_multiplier': 1.0,
                'max_speed': 8.0,
                'max_acceleration': 2.0,
                'max_deceleration': 3.0,
                'max_turn_rate': 45.0,
                'speed_p_gain': 0.5,
                'turn_p_gain': 0.5,
                'waypoint_arrival_threshold': 10.0
            }
            
        if sim_params is None:
            sim_params = {'dt_sim': 0.25}  # 4Hz
            
        # 4. 生成新轨迹
        generated_df = self.generator.generate(
            waypoints=waypoints,
            initial_state=initial_state,
            goal_id=goal_id,
            env_maps=self.env_maps,
            generation_rules=generation_rules,
            control_params=control_params,
            sim_params=sim_params
        )
        
        # 5. 计算统计指标
        metrics = self._calculate_metrics(original_df, generated_df)
        
        return generated_df, metrics
    
    def _calculate_metrics(
        self,
        original_df: pd.DataFrame,
        generated_df: pd.DataFrame
    ) -> Dict:
        """计算对比指标
        
        Args:
            original_df: 原始轨迹数据
            generated_df: 生成的轨迹数据
            
        Returns:
            Dict: 统计指标
        """
        # 1. 速度统计
        orig_speed = np.sqrt(
            original_df['velocity_north_ms']**2 +
            original_df['velocity_east_ms']**2
        )
        gen_speed = np.sqrt(
            generated_df['velocity_north_ms']**2 +
            generated_df['velocity_east_ms']**2
        )
        
        # 重采样到相同长度
        target_len = min(len(original_df), len(generated_df))
        orig_indices = np.linspace(0, len(original_df)-1, target_len, dtype=int)
        gen_indices = np.linspace(0, len(generated_df)-1, target_len, dtype=int)
        
        orig_speed_resampled = orig_speed.iloc[orig_indices]
        gen_speed_resampled = gen_speed.iloc[gen_indices]
        
        speed_metrics = {
            'original_mean_speed': orig_speed.mean(),
            'generated_mean_speed': gen_speed.mean(),
            'original_std_speed': orig_speed.std(),
            'generated_std_speed': gen_speed.std(),
            'speed_correlation': np.corrcoef(orig_speed_resampled, gen_speed_resampled)[0, 1]
        }
        
        # 2. 轨迹形状相似度
        orig_points = np.column_stack([
            original_df['x'].values[orig_indices],
            original_df['y'].values[orig_indices]
        ])
        gen_points = np.column_stack([
            generated_df['x'].values[gen_indices],
            generated_df['y'].values[gen_indices]
        ])
        
        # 计算Hausdorff距离
        distances = cdist(orig_points, gen_points)
        hausdorff_dist = max(
            distances.min(axis=1).max(),
            distances.min(axis=0).max()
        )
        
        shape_metrics = {
            'hausdorff_distance': hausdorff_dist
        }
        
        # 3. 合并所有指标
        metrics = {**speed_metrics, **shape_metrics}
        
        return metrics
    
    def plot_comparison(
        self,
        original_df: pd.DataFrame,
        generated_df: pd.DataFrame,
        metrics: Dict,
        save_path: str = None
    ) -> None:
        """绘制对比图
        
        Args:
            original_df: 原始轨迹数据
            generated_df: 生成的轨迹数据
            metrics: 统计指标
            save_path: 保存路径
        """
        plt.figure(figsize=(15, 10))
        
        # 1. 轨迹对比
        plt.subplot(2, 1, 1)
        plt.plot(
            original_df['x'],
            original_df['y'],
            'b-',
            alpha=0.5,
            label='原始轨迹'
        )
        plt.plot(
            generated_df['x'],
            generated_df['y'],
            'r--',
            alpha=0.5,
            label='生成轨迹'
        )
        plt.title('轨迹对比')
        plt.xlabel('UTM-X (m)')
        plt.ylabel('UTM-Y (m)')
        plt.legend()
        plt.grid(True)
        
        # 2. 速度对比
        plt.subplot(2, 1, 2)
        orig_speed = np.sqrt(
            original_df['velocity_north_ms']**2 +
            original_df['velocity_east_ms']**2
        )
        gen_speed = np.sqrt(
            generated_df['velocity_north_ms']**2 +
            generated_df['velocity_east_ms']**2
        )
        
        plt.plot(
            original_df['timestamp_ms'],
            orig_speed,
            'b-',
            alpha=0.5,
            label='原始速度'
        )
        plt.plot(
            generated_df['timestamp_ms'],
            gen_speed,
            'r--',
            alpha=0.5,
            label='生成速度'
        )
        plt.title('速度对比')
        plt.xlabel('时间 (ms)')
        plt.ylabel('速度 (m/s)')
        plt.legend()
        plt.grid(True)
        
        # 添加统计指标文本
        text = (
            f"统计指标:\n"
            f"原始平均速度: {metrics['original_mean_speed']:.2f} m/s\n"
            f"生成平均速度: {metrics['generated_mean_speed']:.2f} m/s\n"
            f"速度相关系数: {metrics['speed_correlation']:.2f}\n"
            f"Hausdorff距离: {metrics['hausdorff_distance']:.2f} m"
        )
        plt.figtext(0.02, 0.02, text, fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
def main():
    """主函数"""
    # 这里需要实现：
    # 1. 加载原始轨迹数据
    # 2. 加载或创建环境地图对象
    # 3. 创建验证器
    # 4. 进行验证
    pass

if __name__ == '__main__':
    main() 