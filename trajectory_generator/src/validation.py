"""
轨迹验证模块

本模块提供了轨迹验证的功能，包括：
1. 从原始轨迹中提取关键航点
2. 基于航点生成新轨迹
3. 计算轨迹相似度指标
4. 生成对比可视化图表

输入:
- original_df: pd.DataFrame - 原始轨迹数据
- generated_df: pd.DataFrame - 生成的轨迹数据
- env_maps: EnvironmentMaps - 环境地图对象

输出:
- metrics: Dict[str, float] - 验证指标
- comparison_fig: matplotlib.figure.Figure - 对比图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import pearsonr
from .generator import TrajectoryGenerator

class TrajectoryValidator:
    """轨迹验证器类"""
    
    def __init__(self, env_maps: Any, generator: Any = None):
        """初始化验证器
        
        Args:
            env_maps: 环境地图对象
            generator: 轨迹生成器（可选）
        """
        self.env_maps = env_maps
        self.generator = generator or TrajectoryGenerator()
        self.min_distance = 30.0  # 航点最小距离(米)
        self.max_angle = 5.0  # 航向变化阈值(度)
        self.use_original_points = True  # 是否直接使用原始轨迹点
        
    def validate(
        self,
        original_df: pd.DataFrame,
        goal_id: int,
        output_path: str = None,
        visualize: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """验证轨迹生成效果"""
        # 1. 提取航点
        waypoints = list(zip(original_df['x'].values, original_df['y'].values))
        
        # 2. 获取初始状态
        initial_state = self.get_initial_state(original_df)
        
        # 传递基本参数给生成器，移除原始速度相关信息
        sim_params = {
            'points_count': len(original_df),  # 保持轨迹点数相同
            'dt_sim': 0.25  # 4Hz采样率
        }
        
        # 定义控制参数，完全依赖环境特征
        control_params = {
            'global_speed_multiplier': 1.0,  # 保持环境计算的速度
            'max_speed': 8.0,
            'max_acceleration': 1.5,
            'max_deceleration': 2.0,
            'max_turn_rate': 25.0,
            'turn_p_gain': 0.6,
            'waypoint_arrival_threshold': 5.0,
            'curvature_factor': 0.65,
            'min_speed': 1.0
        }
        
        # 3. 生成新轨迹
        generated_df = self.generator.generate(
            waypoints=waypoints,
            initial_state=initial_state,
            goal_id=goal_id,
            env_maps=self.env_maps,
            sim_params=sim_params,
            control_params=control_params
        )
        
        # 4. 计算验证指标
        metrics = self._calculate_metrics(original_df, generated_df)
        
        # 5. 可视化对比(如果需要)
        if visualize and output_path:
            self._visualize_comparison(
                original_df,
                generated_df,
                metrics,
                output_path
            )
            
        return generated_df, metrics
    
    def _check_coordinate_system(self, df: pd.DataFrame) -> bool:
        """检查坐标系统
        
        检查是否已经是UTM坐标，还是需要转换
        
        Args:
            df: 轨迹数据框
            
        Returns:
            bool: 是否是UTM坐标
        """
        # 简单启发式检查：如果x和y的范围都很大，可能是UTM坐标
        x_range = df['x'].max() - df['x'].min()
        y_range = df['y'].max() - df['y'].min()
        
        # UTM坐标通常是几百米到几公里的数量级
        is_utm = x_range > 100 and y_range > 100
        
        print(f"坐标类型检查:")
        print(f"x范围: {df['x'].min():.1f} ~ {df['x'].max():.1f}")
        print(f"y范围: {df['y'].min():.1f} ~ {df['y'].max():.1f}")
        print(f"判断为{'UTM' if is_utm else 'WGS84'}坐标")
        
        return is_utm
    
    def extract_waypoints(
        self,
        trajectory_df: pd.DataFrame,
        visualize: bool = False
    ) -> List[Tuple[float, float]]:
        """从轨迹中提取关键航点
        
        使用Douglas-Peucker算法的变体，考虑:
        1. 航点间的最小距离
        2. 航向变化阈值
        3. 环境特征变化
        
        Args:
            trajectory_df: 轨迹数据框
            visualize: 是否可视化提取过程
            
        Returns:
            List[Tuple[float, float]]: 航点列表(UTM坐标)
        """
        print("\n开始提取关键航点...")
        
        # 1. 检查坐标系统
        is_utm = self._check_coordinate_system(trajectory_df)
        
        # 2. 转换为UTM坐标(如果需要)
        utm_points = []
        for _, row in trajectory_df.iterrows():
            if is_utm:
                # 已经是UTM坐标，直接使用
                x, y = row['x'], row['y']
            else:
                # 需要转换为UTM坐标
                x, y = self.generator.utm_to_wgs84.transform(
                    row['x'],
                    row['y'],
                    direction='INVERSE'
                )
            utm_points.append((x, y))
        utm_points = np.array(utm_points)
        
        print(f"\n第一个航点: ({utm_points[0,0]:.2f}, {utm_points[0,1]:.2f})")
        
        # 3. 初始化航点列表
        waypoints = [utm_points[0]]  # 起点
        last_point = utm_points[0]
        last_heading = None
        
        # 4. 遍历所有点
        for i in range(1, len(utm_points)):
            current_point = utm_points[i]
            
            # 每1000个点打印一次进度
            if i % 1000 == 0:
                print(f"\n处理第{i}个点:")
                print(f"当前点: ({current_point[0]:.2f}, {current_point[1]:.2f})")
                print(f"上一航点: ({last_point[0]:.2f}, {last_point[1]:.2f})")
            
            # 计算与上一航点的距离
            dist = np.linalg.norm(current_point - last_point)
            
            # 如果距离太小，跳过
            if dist < self.min_distance:
                continue
                
            # 计算航向角
            vec = current_point - last_point
            current_heading = np.degrees(np.arctan2(vec[1], vec[0]))
            
            # 如果是第一个点，记录航向并继续
            if last_heading is None:
                last_heading = current_heading
                continue
                
            # 计算航向变化
            heading_change = abs((current_heading - last_heading + 180) % 360 - 180)
            
            # 如果航向变化大于阈值，添加为航点
            if heading_change > self.max_angle:
                if i % 1000 == 0:
                    print(f"距离: {dist:.2f}m")
                    print(f"向量1: ({vec[0]:.2f}, {vec[1]:.2f})")
                    print(f"向量2: ({np.cos(np.radians(current_heading)):.2f}, {np.sin(np.radians(current_heading)):.2f})")
                    print(f"转角: {heading_change:.2f}度")
                waypoints.append(current_point)  # 添加当前点而不是上一个点
                last_point = current_point
                last_heading = current_heading
                continue
                
            # 检查环境特征变化
            last_features = self.env_maps.query_by_xy(last_point[0], last_point[1])
            current_features = self.env_maps.query_by_xy(current_point[0], current_point[1])
            
            # 如果地类发生变化，添加为航点
            if last_features['landcover'] != current_features['landcover']:
                waypoints.append(current_point)  # 这里也改为添加当前点
                last_point = current_point
                last_heading = current_heading
                continue
                
            # 更新上一个点
            last_point = current_point
            last_heading = current_heading
            
        # 5. 添加终点
        waypoints.append(utm_points[-1])
        
        print(f"\n提取的航点总数: {len(waypoints)}")
        
        # 6. 可视化(如果需要)
        if visualize:
            self._visualize_waypoints(utm_points, waypoints)
            
        return waypoints
    
    def _visualize_waypoints(
        self,
        trajectory: np.ndarray,
        waypoints: List[Tuple[float, float]]
    ) -> None:
        """可视化航点提取结果
        
        Args:
            trajectory: 原始轨迹点
            waypoints: 提取的航点
        """
        plt.figure(figsize=(12, 8))
        
        # 绘制原始轨迹
        plt.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            'b-',
            alpha=0.5,
            label='原始轨迹'
        )
        
        # 绘制航点
        waypoints = np.array(waypoints)
        plt.plot(
            waypoints[:, 0],
            waypoints[:, 1],
            'ro-',
            label='关键航点'
        )
        
        # 添加起终点标记
        plt.plot(
            waypoints[0, 0],
            waypoints[0, 1],
            'go',
            markersize=10,
            label='起点'
        )
        plt.plot(
            waypoints[-1, 0],
            waypoints[-1, 1],
            'ko',
            markersize=10,
            label='终点'
        )
        
        plt.title('轨迹航点提取结果')
        plt.xlabel('UTM东向坐标(m)')
        plt.ylabel('UTM北向坐标(m)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    
    def plot_comparison(
        self,
        original_df: pd.DataFrame,
        generated_df: pd.DataFrame,
        metrics: Dict[str, float],
        save_path: str
    ) -> None:
        """绘制轨迹对比图
        
        Args:
            original_df: 原始轨迹数据
            generated_df: 生成的轨迹数据
            metrics: 统计指标
            save_path: 保存路径
        """
        self._visualize_comparison(
            original_df,
            generated_df,
            metrics,
            save_path
        )

    def get_initial_state(
        self,
        df: pd.DataFrame
    ) -> Dict[str, float]:
        """获取初始状态
        
        Args:
            df: 原始轨迹数据
            
        Returns:
            Dict[str, float]: 初始状态字典
        """
        # 获取第一个点的速度和位置
        vn = df.iloc[0]['velocity_north_ms']
        ve = df.iloc[0]['velocity_east_ms']
        x = df.iloc[0]['x']
        y = df.iloc[0]['y']
        
        # 计算初始航向
        heading = np.degrees(np.arctan2(vn, ve))
        if heading < 0:
            heading += 360
            
        return {
            'x0': x,
            'y0': y,
            'vx0': ve,
            'vy0': vn,
            'heading0': heading
        }
    
    def _calculate_metrics(
        self,
        original_df: pd.DataFrame,
        generated_df: pd.DataFrame
    ) -> Dict[str, float]:
        """计算验证指标
        
        Args:
            original_df: 原始轨迹数据
            generated_df: 生成的轨迹数据
            
        Returns:
            Dict[str, float]: 指标字典
        """
        # 1. 计算速度
        # 原始轨迹速度
        original_speed = np.sqrt(
            original_df['velocity_north_ms']**2 +
            original_df['velocity_east_ms']**2
        )
        # 生成轨迹速度
        generated_speed = np.sqrt(
            generated_df['velocity_north_ms']**2 +
            generated_df['velocity_east_ms']**2
        )
        
        # 2. 计算统计指标
        # 平均速度
        original_mean_speed = float(original_speed.mean())
        generated_mean_speed = float(generated_speed.mean())
        
        # 速度标准差
        original_std_speed = float(original_speed.std())
        generated_std_speed = float(generated_speed.std())
        
        # 速度相关系数
        # 重采样到相同长度以计算相关系数
        min_len = min(len(original_speed), len(generated_speed))
        original_resampled = np.interp(
            np.linspace(0, 1, min_len),
            np.linspace(0, 1, len(original_speed)),
            original_speed
        )
        generated_resampled = np.interp(
            np.linspace(0, 1, min_len),
            np.linspace(0, 1, len(generated_speed)),
            generated_speed
        )
        speed_correlation = float(pearsonr(original_resampled, generated_resampled)[0])
        
        # 3. 计算轨迹形状相似度
        # 提取坐标点
        original_points = np.column_stack([
            original_df['x'].values,
            original_df['y'].values
        ])
        generated_points = np.column_stack([
            generated_df['x'].values,
            generated_df['y'].values
        ])
        
        # 计算Hausdorff距离
        hausdorff_dist = max(
            directed_hausdorff(original_points, generated_points)[0],
            directed_hausdorff(generated_points, original_points)[0]
        )
        
        return {
            'original_mean_speed': original_mean_speed,
            'generated_mean_speed': generated_mean_speed,
            'original_std_speed': original_std_speed,
            'generated_std_speed': generated_std_speed,
            'speed_correlation': speed_correlation,
            'hausdorff_distance': float(hausdorff_dist)
        }
    
    def _visualize_comparison(
        self,
        original_df: pd.DataFrame,
        generated_df: pd.DataFrame,
        metrics: Dict[str, float],
        save_path: str
    ) -> None:
        """可视化对比结果
        
        Args:
            original_df: 原始轨迹数据
            generated_df: 生成的轨迹数据
            metrics: 统计指标
            save_path: 保存路径
        """
        # 创建2x3的子图布局（多一个子图用于轨迹点索引-速度图）
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
        
        # 添加总标题
        fig.suptitle(f'轨迹{save_path.split("trajectory_")[-1].split("_")[0]}分析结果\n原始速度: {metrics["original_mean_speed"]:.2f}±{metrics["original_std_speed"]:.2f} m/s | 生成速度: {metrics["generated_mean_speed"]:.2f}±{metrics["generated_std_speed"]:.2f} m/s | 相关系数: {metrics["speed_correlation"]:.3f}', fontsize=20, y=0.98)
        
        # 1. 轨迹形状对比 (左上)
        ax1.plot(original_df['x'], original_df['y'], 'b-', label='原始轨迹', alpha=0.7)
        ax1.plot(generated_df['x'], generated_df['y'], 'r--', label='生成轨迹', alpha=0.7)
        
        # 标记起点和终点
        ax1.plot(original_df['x'].iloc[0], original_df['y'].iloc[0], 'go', label='起点')
        ax1.plot(original_df['x'].iloc[-1], original_df['y'].iloc[-1], 'ro', label='终点')
        
        ax1.set_title('轨迹形状对比', fontsize=18)
        ax1.set_xlabel('UTM-X (m)', fontsize=16)
        ax1.set_ylabel('UTM-Y (m)', fontsize=16)
        ax1.legend(fontsize=14)
        ax1.grid(True)
        ax1.axis('equal')
        
        # 2. 速度时间序列对比 (右上)
        # 如果有时间戳直接使用，否则基于采样率估计
        if 'timestamp_ms' in original_df.columns and 'timestamp_ms' in generated_df.columns:
            # 转换为秒
            time_original = original_df['timestamp_ms'].values / 1000.0
            time_original = time_original - time_original[0]  # 从0开始
            
            time_generated = generated_df['timestamp_ms'].values / 1000.0
            time_generated = time_generated - time_generated[0]  # 从0开始
        else:
            # 使用单位时间步作为估计
            time_original = np.arange(len(original_df)) / 4.0  # 假设4Hz
            time_generated = np.arange(len(generated_df)) / 4.0
            
        # 计算实际的时间步长
        dt_original = np.mean(np.diff(time_original)) if len(time_original) > 1 else 0.25
        dt_generated = np.mean(np.diff(time_generated)) if len(time_generated) > 1 else 0.25
        
        original_speed = np.sqrt(original_df['velocity_north_ms']**2 + original_df['velocity_east_ms']**2)
        generated_speed = np.sqrt(generated_df['velocity_north_ms']**2 + generated_df['velocity_east_ms']**2)
        
        # 确保两个时间序列的最大值相同（以便比较）
        max_time = max(time_original[-1], time_generated[-1])
        ax2.set_xlim([0, max_time])
        
        ax2.plot(time_original, original_speed, 'b-', label='原始速度', alpha=0.7)
        ax2.plot(time_generated, generated_speed, 'r--', label='生成速度', alpha=0.7)
        ax2.set_title(f'速度时间序列对比 (原始dt={dt_original:.3f}s, 生成dt={dt_generated:.3f}s)', fontsize=18)
        ax2.set_xlabel('时间 (s)', fontsize=16)
        ax2.set_ylabel('速度 (m/s)', fontsize=16)
        ax2.legend(fontsize=14)
        ax2.grid(True)
        
        # 3. 速度分布对比 (左中)
        ax3.hist(original_speed, bins=30, density=True, alpha=0.5, color='b', label='原始速度')
        ax3.hist(generated_speed, bins=30, density=True, alpha=0.5, color='r', label='生成速度')
        ax3.set_title('速度分布对比', fontsize=18)
        ax3.set_xlabel('速度 (m/s)', fontsize=16)
        ax3.set_ylabel('概率密度', fontsize=16)
        ax3.legend(fontsize=14)
        ax3.grid(True)
        
        # 4. 统计指标展示 (右中)
        ax4.axis('off')
        
        # 计算总时间和距离
        total_time_original = time_original[-1] - time_original[0]
        total_time_generated = time_generated[-1] - time_generated[0]
        
        # 计算总距离
        def calculate_distance(df):
            if len(df) <= 1:
                return 0
            points = np.column_stack([df['x'].values, df['y'].values])
            diffs = np.diff(points, axis=0)
            distances = np.sqrt(np.sum(diffs**2, axis=1))
            return np.sum(distances)
        
        total_distance_original = calculate_distance(original_df)
        total_distance_generated = calculate_distance(generated_df)
        
        metrics_text = (
            f"统计指标:\n\n"
            f"原始平均速度: {metrics['original_mean_speed']:.2f} m/s\n"
            f"生成平均速度: {metrics['generated_mean_speed']:.2f} m/s\n\n"
            f"原始速度标准差: {metrics['original_std_speed']:.2f} m/s\n"
            f"生成速度标准差: {metrics['generated_std_speed']:.2f} m/s\n\n"
            f"速度相关系数: {metrics['speed_correlation']:.3f}\n"
            f"Hausdorff距离: {metrics['hausdorff_distance']:.1f} m\n\n"
            f"原始轨迹时间: {total_time_original:.1f} s\n"
            f"生成轨迹时间: {total_time_generated:.1f} s\n\n"
            f"原始轨迹距离: {total_distance_original:.1f} m\n"
            f"生成轨迹距离: {total_distance_generated:.1f} m\n\n"
            f"原始轨迹点数: {len(original_df)}\n"
            f"生成轨迹点数: {len(generated_df)}"
        )
        ax4.text(0.1, 0.5, metrics_text, fontsize=14, va='center')
        
        # 5. 新增：按轨迹点索引的速度对比图 (左下)
        # 创建点索引（两个轨迹都从0开始）
        idx_original = np.arange(len(original_df))
        idx_generated = np.arange(len(generated_df))
        
        # 如果两个轨迹长度不一致，进行重采样以便比较
        if len(original_df) != len(generated_df):
            # 重采样到较短的长度
            target_len = min(len(original_df), len(generated_df))
            
            # 如果原始轨迹更长，重采样它
            if len(original_df) > target_len:
                resampled_indices = np.linspace(0, len(original_df)-1, target_len).astype(int)
                idx_original = idx_original[resampled_indices]
                original_speed_resampled = original_speed.iloc[resampled_indices].values
            else:
                original_speed_resampled = original_speed.values
            
            # 如果生成轨迹更长，重采样它
            if len(generated_df) > target_len:
                resampled_indices = np.linspace(0, len(generated_df)-1, target_len).astype(int)
                idx_generated = idx_generated[resampled_indices]
                generated_speed_resampled = generated_speed.iloc[resampled_indices].values
            else:
                generated_speed_resampled = generated_speed.values
        else:
            # 长度一致，直接使用
            original_speed_resampled = original_speed.values
            generated_speed_resampled = generated_speed.values
        
        # 绘制按轨迹点索引的速度对比
        ax5.plot(idx_original, original_speed_resampled, 'b-', label='原始速度', alpha=0.7)
        ax5.plot(idx_generated, generated_speed_resampled, 'r--', label='生成速度', alpha=0.7)
        ax5.set_title('按轨迹点索引的速度对比', fontsize=18)
        ax5.set_xlabel('轨迹点索引', fontsize=16)
        ax5.set_ylabel('速度 (m/s)', fontsize=16)
        ax5.legend(fontsize=14)
        ax5.grid(True)
        
        # 6. 留一个空位以便后续添加其他图表 (右下)
        ax6.axis('off')
        ax6.set_title('预留空间')
        
        # 调整布局
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close() 