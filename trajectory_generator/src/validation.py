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
    
    def __init__(self, env_maps: Any):
        """初始化验证器
        
        Args:
            env_maps: 环境地图对象
        """
        self.env_maps = env_maps
        self.generator = TrajectoryGenerator()
        
        # 航点提取参数
        self.min_distance = 30.0  # 最小航点间距(米)
        self.max_angle = 5.0      # 最大航向变化(度)，降低到5度以更精确捕捉转向
        
        # 轨迹重采样参数
        self.resample_freq = '250ms'  # 重采样频率(4Hz)
        
    def validate(
        self,
        original_df: pd.DataFrame,
        goal_id: int,
        output_path: str = None,
        visualize: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """验证轨迹生成效果
        
        Args:
            original_df: 原始轨迹数据
            goal_id: 目标点ID
            output_path: 输出路径(可选)
            visualize: 是否可视化
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, float]]:
                生成的轨迹数据
                验证指标
        """
        print("\n开始轨迹验证...")
        
        # 1. 提取关键航点
        waypoints = self.extract_waypoints(original_df, visualize=visualize)
        
        # 2. 获取初始状态
        initial_state = self._get_initial_state(original_df, waypoints[0])
        
        # 3. 生成新轨迹
        generated_df = self.generator.generate(
            waypoints=waypoints,
            initial_state=initial_state,
            goal_id=goal_id,
            env_maps=self.env_maps
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
        """检查数据框的坐标系统
        
        通过检查x和y列的数值范围来判断是否已经是UTM坐标系
        
        Args:
            df: 轨迹数据框
            
        Returns:
            bool: 如果是UTM坐标系返回True，否则返回False
        """
        # 检查坐标范围
        x_range = (df['x'].min(), df['x'].max())
        y_range = (df['y'].min(), df['y'].max())
        
        print("坐标范围检查:")
        print(f"utm_x范围: {x_range[0]:.2f} - {x_range[1]:.2f}")
        print(f"utm_y范围: {y_range[0]:.2f} - {y_range[1]:.2f}")
        
        # UTM坐标通常是米制的大数值
        # 经纬度通常在-180到180之间
        is_utm = (
            abs(x_range[0]) > 180 or
            abs(x_range[1]) > 180 or
            abs(y_range[0]) > 90 or
            abs(y_range[1]) > 90
        )
        
        return is_utm
    
    def extract_waypoints(
        self,
        df: pd.DataFrame,
        visualize: bool = False
    ) -> List[Tuple[float, float]]:
        """从轨迹中提取关键航点
        
        使用Douglas-Peucker算法的变体，考虑:
        1. 航点间的最小距离
        2. 航向变化阈值
        3. 环境特征变化
        
        Args:
            df: 轨迹数据框
            visualize: 是否可视化提取过程
            
        Returns:
            List[Tuple[float, float]]: 航点列表(UTM坐标)
        """
        print("\n开始提取关键航点...")
        
        # 1. 检查坐标系统
        is_utm = self._check_coordinate_system(df)
        
        # 2. 转换为UTM坐标(如果需要)
        utm_points = []
        for _, row in df.iterrows():
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

    def _get_initial_state(
        self,
        df: pd.DataFrame,
        start_point: Tuple[float, float]
    ) -> Dict[str, float]:
        """获取初始状态
        
        Args:
            df: 原始轨迹数据
            start_point: 起点UTM坐标
            
        Returns:
            Dict[str, float]: 初始状态字典
        """
        # 获取第一个点的速度
        vn = df.iloc[0]['velocity_north_ms']
        ve = df.iloc[0]['velocity_east_ms']
        
        # 计算初始航向
        heading = np.degrees(np.arctan2(vn, ve))
        if heading < 0:
            heading += 360
            
        return {
            'x0': start_point[0],
            'y0': start_point[1],
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
        # 创建2x2的子图布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 轨迹形状对比 (左上)
        ax1.plot(original_df['x'], original_df['y'], 'b-', label='原始轨迹', alpha=0.7)
        ax1.plot(generated_df['x'], generated_df['y'], 'r--', label='生成轨迹', alpha=0.7)
        
        # 标记起点和终点
        ax1.plot(original_df['x'].iloc[0], original_df['y'].iloc[0], 'go', label='起点')
        ax1.plot(original_df['x'].iloc[-1], original_df['y'].iloc[-1], 'ro', label='终点')
        
        ax1.set_title('轨迹形状对比')
        ax1.set_xlabel('UTM-X (m)')
        ax1.set_ylabel('UTM-Y (m)')
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')  # 保持比例一致
        
        # 2. 速度时间序列对比 (右上)
        time_original = np.arange(len(original_df)) / 4.0  # 4Hz采样率
        time_generated = np.arange(len(generated_df)) / 4.0
        
        original_speed = np.sqrt(original_df['velocity_north_ms']**2 + original_df['velocity_east_ms']**2)
        generated_speed = np.sqrt(generated_df['velocity_north_ms']**2 + generated_df['velocity_east_ms']**2)
        
        ax2.plot(time_original, original_speed, 'b-', label='原始速度', alpha=0.7)
        ax2.plot(time_generated, generated_speed, 'r--', label='生成速度', alpha=0.7)
        ax2.set_title('速度时间序列对比')
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('速度 (m/s)')
        ax2.legend()
        ax2.grid(True)
        
        # 3. 速度分布对比 (左下)
        ax3.hist(original_speed, bins=30, density=True, alpha=0.5, color='b', label='原始速度')
        ax3.hist(generated_speed, bins=30, density=True, alpha=0.5, color='r', label='生成速度')
        ax3.set_title('速度分布对比')
        ax3.set_xlabel('速度 (m/s)')
        ax3.set_ylabel('概率密度')
        ax3.legend()
        ax3.grid(True)
        
        # 4. 统计指标展示 (右下)
        ax4.axis('off')
        metrics_text = (
            f"统计指标:\n\n"
            f"原始平均速度: {metrics['original_mean_speed']:.2f} m/s\n"
            f"生成平均速度: {metrics['generated_mean_speed']:.2f} m/s\n\n"
            f"原始速度标准差: {metrics['original_std_speed']:.2f} m/s\n"
            f"生成速度标准差: {metrics['generated_std_speed']:.2f} m/s\n\n"
            f"速度相关系数: {metrics['speed_correlation']:.3f}\n"
            f"Hausdorff距离: {metrics['hausdorff_distance']:.1f} m"
        )
        ax4.text(0.1, 0.5, metrics_text, fontsize=12, va='center')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig) 