"""
轨迹生成器模块

本模块实现了一个基于规则和统计特性的轨迹生成器，可以：
1. 接收路径规划的离散航点序列
2. 基于真实数据分析的规则生成连续轨迹
3. 支持参数化调整以生成不同特性的轨迹

输入:
- waypoints: List[Tuple[float, float]] - 路径规划航点序列 (UTM坐标)
- initial_state: dict - 初始状态 (位置、速度、航向)
- goal_id: int - 目标点ID (0-3)
- env_maps: EnvironmentMaps - 环境地图对象
- generation_rules: dict - 生成规则参数
- control_params: dict - 控制参数
- sim_params: dict - 仿真参数

输出:
- trajectory_df: pd.DataFrame - 生成的轨迹数据(4Hz采样)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import pyproj
import os
import glob

@dataclass
class EnvironmentState:
    """环境状态数据类"""
    landcover: int  # 土地覆盖类型
    slope_magnitude: float  # 坡度大小
    slope_aspect: float  # 坡向(度)
    
@dataclass
class VehicleState:
    """车辆状态数据类"""
    position: np.ndarray  # [x, y] UTM坐标
    velocity: np.ndarray  # [vx, vy] 速度分量
    heading: float  # 航向角(度)
    acceleration: np.ndarray  # [ax, ay] 加速度分量
    timestamp: float  # 时间戳(秒)

class TrajectoryGenerator:
    """轨迹生成器类"""
    
    # 默认的地类基准速度(m/s)，基于分析结果估计的平均值
    DEFAULT_BASE_SPEEDS = {
        20: 6.2,  # 林地 - 根据分析结果调整为更高值
        40: 5.5,  # 灌木地 - 根据分析结果调整为更高值
        60: 4.8,  # 水体 - 根据分析结果调整为更高值
        90: 3.5   # 其他 - 提高默认值以匹配观察到的速度
    }
    
    # 默认的坡度影响系数，基于分析结果估计的平均值
    DEFAULT_SLOPE_COEFFICIENTS = {
        20: -0.032,  # 林地 - 坡度影响更显著
        40: -0.025,  # 灌木地 - 中等坡度影响
        60: -0.020,  # 水体 - 较弱坡度影响
        90: -0.028   # 其他 - 增加默认值
    }
    
    def __init__(self):
        """初始化转换器"""
        self.utm_to_wgs84 = pyproj.Transformer.from_crs(
            "EPSG:32630", "EPSG:4326", always_xy=True
        )
        
        # 从分析结果加载更精确的模型
        # 为每条轨迹建立单独的模型
        self.slope_speed_models = {}
        self.trajectory_specific_models = {}  # 存储特定轨迹的模型
        self.residual_distributions = {}  # 存储残差分布
        self.load_analysis_results()
        
    def load_analysis_results(self):
        """加载分析结果，构建更精确的速度模型"""
        # 基础路径，可以根据实际情况调整
        analysis_dir = '/home/yzc/data/Sucess_or_Die/ai_agent_generation/analysisi_result'
        
        # 定义地类映射
        landcover_names = {
            20: "林地",
            40: "灌木地",
            60: "水体"
        }
        
        # 1. 为每种地类建立通用模型
        self._load_general_models(analysis_dir, landcover_names)
        
        # 2. 为每条轨迹建立特定模型
        self._load_trajectory_specific_models(analysis_dir, landcover_names)
        
        # 3. 加载残差分布
        self._load_residual_distributions(analysis_dir, landcover_names)
            
        # 如果没有加载到任何模型，使用默认值
        if not self.slope_speed_models and not self.trajectory_specific_models:
            print("警告: 未能加载任何分析结果，将使用默认的速度和坡度系数")
    
    def _load_general_models(self, analysis_dir, landcover_names):
        """加载通用速度-坡度模型（按地类）"""
        # 加载所有轨迹的坡度分箱统计
        for landcover_id, landcover_name in landcover_names.items():
            slope_bins = []
            slope_values = []
            speed_means = []
            speed_stds = []
            
            # 查找所有相关的CSV文件（使用1秒窗口的分析结果）
            pattern = os.path.join(analysis_dir, f"轨迹*_坡度分箱统计_1秒_{landcover_name}.csv")
            csv_files = glob.glob(pattern)
            
            for csv_file in csv_files:
                try:
                    print(f"正在处理: {csv_file}")
                    df = pd.read_csv(csv_file)
                    
                    # 合并数据
                    if not slope_bins:
                        slope_bins = df['slope_bin'].tolist()
                        # 从斜率范围的字符串中提取中间值
                        for bin_str in slope_bins:
                            # 处理字符串格式 "(-35, -30]"
                            try:
                                # 去除引号和括号
                                clean_str = bin_str.replace('"', '').replace('(', '').replace(']', '')
                                # 分割并获取两个值
                                parts = clean_str.split(',')
                                # 取中间值
                                if len(parts) == 2:
                                    min_val = float(parts[0].strip())
                                    max_val = float(parts[1].strip())
                                    slope_values.append((min_val + max_val) / 2)
                                else:
                                    slope_values.append(0)  # 默认值
                            except:
                                print(f"解析斜率范围时出错: {bin_str}")
                                slope_values.append(0)  # 默认值
                    
                    # 累加均值和标准差（后面会取平均）
                    if 'mean_speed' in df.columns:
                        if not speed_means:
                            speed_means = df['mean_speed'].fillna(0).tolist()
                            speed_stds = df['std_speed'].fillna(0).tolist()
                        else:
                            for i, speed in enumerate(df['mean_speed'].fillna(0)):
                                if i < len(speed_means):
                                    speed_means[i] += speed
                                    speed_stds[i] += df['std_speed'].fillna(0).iloc[i]
                except Exception as e:
                    print(f"警告: 无法加载 {csv_file}: {e}")
            
            # 计算平均值
            if csv_files and speed_means:
                n_files = len(csv_files)
                speed_means = [speed / n_files for speed in speed_means]
                speed_stds = [std / n_files for std in speed_stds]
                
                # 创建完整的模型（斜率、截距和方差）
                # 使用简单线性回归：speed = slope * slope_magnitude + intercept
                if len(slope_values) >= 2 and len(speed_means) >= 2:
                    # 确保数据长度一致
                    min_len = min(len(slope_values), len(speed_means))
                    x = np.array(slope_values[:min_len])
                    y = np.array(speed_means[:min_len])
                    
                    # 过滤掉NaN值
                    valid_indices = ~np.isnan(x) & ~np.isnan(y)
                    x_filtered = x[valid_indices]
                    y_filtered = y[valid_indices]
                    
                    if len(x_filtered) >= 2:  # 需要至少2个点才能进行回归
                        # 简单线性回归
                        slope, intercept = np.polyfit(x_filtered, y_filtered, 1)
                        
                        # 寻找斜率为0的基准速度
                        flat_speed_idx = np.argmin(np.abs(x_filtered))
                        base_speed = y_filtered[flat_speed_idx] if flat_speed_idx < len(y_filtered) else y_filtered.mean()
                        
                        # 存储模型
                        self.slope_speed_models[landcover_id] = {
                            'slope': slope,
                            'intercept': intercept,
                            'base_speed': base_speed,
                            'std': np.nanmean(speed_stds[:min_len])  # 使用平均标准差
                        }
                        
                        print(f"已加载{landcover_name}的速度-坡度模型: speed = {slope:.4f} * slope + {intercept:.2f}")
                    else:
                        print(f"警告: {landcover_name}的有效数据不足，使用默认模型")
                else:
                    print(f"警告: {landcover_name}的数据不足，使用默认模型")
    
    def _load_trajectory_specific_models(self, analysis_dir, landcover_names):
        """为每条轨迹加载特定的速度-坡度模型"""
        # 轨迹ID模式：轨迹1, 轨迹2, 轨迹3, 轨迹4
        trajectory_ids = [1, 2, 3, 4]
        
        for traj_id in trajectory_ids:
            # 为每个地类创建模型
            traj_models = {}
            
            for landcover_id, landcover_name in landcover_names.items():
                # 找到特定轨迹的分析结果
                csv_path = os.path.join(analysis_dir, f"轨迹{traj_id}_坡度分箱统计_1秒_{landcover_name}.csv")
                
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        
                        # 从斜率范围中提取中间值
                        slope_values = []
                        for bin_str in df['slope_bin'].tolist():
                            try:
                                # 去除引号和括号
                                clean_str = bin_str.replace('"', '').replace('(', '').replace(']', '')
                                # 分割并获取两个值
                                parts = clean_str.split(',')
                                # 取中间值
                                if len(parts) == 2:
                                    min_val = float(parts[0].strip())
                                    max_val = float(parts[1].strip())
                                    slope_values.append((min_val + max_val) / 2)
                                else:
                                    slope_values.append(0)  # 默认值
                            except:
                                print(f"解析斜率范围时出错: {bin_str}")
                                slope_values.append(0)  # 默认值
                        
                        # 检查数据完整性
                        if 'mean_speed' in df.columns and len(slope_values) >= 2:
                            x = np.array(slope_values)
                            y = df['mean_speed'].fillna(0).values
                            
                            # 过滤掉NaN值
                            valid_indices = ~np.isnan(x) & ~np.isnan(y)
                            x_filtered = x[valid_indices]
                            y_filtered = y[valid_indices]
                            
                            if len(x_filtered) >= 2:
                                # 线性回归
                                slope, intercept = np.polyfit(x_filtered, y_filtered, 1)
                                
                                # 标准差计算
                                std_values = df['std_speed'].fillna(0).values
                                std_filtered = std_values[valid_indices]
                                
                                # 存储特定轨迹的模型
                                traj_models[landcover_id] = {
                                    'slope': slope,
                                    'intercept': intercept,
                                    'std': np.mean(std_filtered),
                                    'x_values': x_filtered.tolist(),  # 保存原始数据点，用于残差分析
                                    'y_values': y_filtered.tolist()
                                }
                                
                                print(f"已加载轨迹{traj_id}-{landcover_name}的模型: speed = {slope:.4f} * slope + {intercept:.2f}")
                    except Exception as e:
                        print(f"加载轨迹{traj_id}-{landcover_name}模型失败: {e}")
            
            # 如果有模型，保存到字典中
            if traj_models:
                self.trajectory_specific_models[traj_id] = traj_models
                print(f"轨迹{traj_id}已加载{len(traj_models)}个地类模型")
    
    def _load_residual_distributions(self, analysis_dir, landcover_names):
        """加载并存储残差分布"""
        try:
            # 对于每个轨迹特定模型，计算残差分布
            for traj_id, landcover_models in self.trajectory_specific_models.items():
                traj_residuals = {}
                
                for landcover_id, model in landcover_models.items():
                    # 从模型中提取原始数据点
                    if 'x_values' in model and 'y_values' in model:
                        x_values = np.array(model['x_values'])
                        y_values = np.array(model['y_values'])
                        
                        # 使用模型预测
                        y_pred = model['slope'] * x_values + model['intercept']
                        
                        # 计算残差
                        residuals = y_values - y_pred
                        
                        # 存储残差分布
                        traj_residuals[landcover_id] = {
                            'values': residuals.tolist(),
                            'mean': float(np.mean(residuals)),
                            'std': float(np.std(residuals)),
                            'min': float(np.min(residuals)),
                            'max': float(np.max(residuals))
                        }
                        
                        print(f"轨迹{traj_id}-{landcover_names.get(landcover_id, '未知')}的残差统计: "
                              f"均值={np.mean(residuals):.3f}, 标准差={np.std(residuals):.3f}")
                
                if traj_residuals:
                    self.residual_distributions[traj_id] = traj_residuals
        except Exception as e:
            print(f"加载残差分布时出错: {e}")
    
    def _sample_from_residual_distribution(self, traj_id, landcover_id):
        """从残差分布中采样
        
        如果有特定轨迹的残差分布，使用实际残差分布采样
        否则退化为高斯随机采样
        
        Args:
            traj_id: 轨迹ID
            landcover_id: 地类ID
            
        Returns:
            float: 采样的残差值
        """
        # 检查是否有特定轨迹的残差分布
        if traj_id in self.residual_distributions and landcover_id in self.residual_distributions[traj_id]:
            # 有残差分布，直接从实际值中随机选择
            residuals = self.residual_distributions[traj_id][landcover_id]['values']
            if residuals:
                return np.random.choice(residuals)
            else:
                # 没有残差值，使用统计量生成
                dist = self.residual_distributions[traj_id][landcover_id]
                return np.random.normal(dist['mean'], dist['std'])
        
        # 检查是否有特定轨迹的模型
        if traj_id in self.trajectory_specific_models and landcover_id in self.trajectory_specific_models[traj_id]:
            model = self.trajectory_specific_models[traj_id][landcover_id]
            if 'std' in model:
                return np.random.normal(0, model['std'])
        
        # 检查是否有通用模型
        if landcover_id in self.slope_speed_models and 'std' in self.slope_speed_models[landcover_id]:
            return np.random.normal(0, self.slope_speed_models[landcover_id]['std'] * 0.5)
        
        # 默认返回小随机值
        return np.random.normal(0, 0.2)
            
    def generate(
        self,
        waypoints: List[Tuple[float, float]],
        initial_state: Dict,
        goal_id: int,
        env_maps: Any,
        generation_rules: Dict = None,
        control_params: Dict = None,
        sim_params: Dict = None
    ) -> pd.DataFrame:
        """生成轨迹
        
        Args:
            waypoints: 航点列表 [(x1,y1), (x2,y2), ...]
            initial_state: 初始状态
            goal_id: 目标点ID
            env_maps: 环境地图对象
            generation_rules: 生成规则参数(可选)
            control_params: 控制参数(可选)
            sim_params: 仿真参数(可选)
            
        Returns:
            pd.DataFrame: 生成的轨迹数据
        """
        print("\n开始生成轨迹...")
        
        # 检查轨迹点数量
        total_points = len(waypoints)
        print(f"输入航点数量: {total_points}")
        
        # 如果输入点过多，进行抽样减少计算量
        if total_points > 10000:
            print(f"输入点数过多，进行抽样...")
            sample_rate = max(1, total_points // 5000)  # 抽样比例，保证不超过5000个点
            waypoints = waypoints[::sample_rate]
            print(f"抽样后航点数量: {len(waypoints)}")
        
        # 初始化参数
        print(f"初始状态: {initial_state}")
        
        # 1. 使用默认参数
        if generation_rules is None:
            generation_rules = {
                'base_speeds': self.DEFAULT_BASE_SPEEDS,
                'slope_coefficients': self.DEFAULT_SLOPE_COEFFICIENTS,
                'trajectory_id': goal_id + 1  # 使用goal_id+1作为轨迹ID
            }
        else:
            # 确保有轨迹ID
            if 'trajectory_id' not in generation_rules:
                generation_rules['trajectory_id'] = goal_id + 1
            
        if control_params is None:
            # 对于使用所有原始点的情况，调整控制参数以获得更平滑的曲线
            control_params = {
                'global_speed_multiplier': 0.8,  # 降低全局速度，使用原始速度作为主导
                'max_speed': 8.0,
                'max_acceleration': 1.5,  # 降低加速度限制
                'max_deceleration': 2.0,  # 降低减速度限制
                'max_turn_rate': 25.0,     # 降低最大转向速率
                'turn_p_gain': 0.6,        # 降低转向响应以更平滑
                'waypoint_arrival_threshold': 5.0,  # 减小到达阈值
                'curvature_factor': 0.65,    # 增强曲率影响
                'use_original_speeds': True  # 使用原始速度作为参考
            }
            
        if sim_params is None:
            sim_params = {'dt_sim': 0.25}  # 4Hz
            
        # 检查是否需要控制轨迹点数量
        match_original_count = False
        target_points_count = None
        if sim_params and 'points_count' in sim_params:
            match_original_count = True
            target_points_count = sim_params.get('points_count', 0)
            print(f"目标轨迹点数量: {target_points_count}")
            
        # 检查是否有原始时间戳
        use_original_timestamps = False
        original_timestamps = None
        if sim_params and 'original_timestamps' in sim_params and sim_params['original_timestamps'] is not None:
            use_original_timestamps = True
            original_timestamps = sim_params['original_timestamps']
            print(f"使用原始时间戳: {len(original_timestamps)}个点")
            
            # 强制匹配原始点数
            if len(original_timestamps) > 0:
                match_original_count = True
                target_points_count = len(original_timestamps)
                print(f"将匹配原始轨迹点数: {target_points_count}点")
        
        # 提取原始速度信息，用于更好地匹配原始轨迹的速度模式
        original_speeds = None
        if 'original_speeds' in sim_params and sim_params['original_speeds'] is not None:
            original_speeds = sim_params['original_speeds']
            print(f"使用原始速度作为参考: {len(original_speeds)}个点")
        elif control_params.get('use_original_speeds', False) and sim_params and 'original_df' in sim_params:
            # 如果有原始数据帧，提取速度
            orig_df = sim_params.get('original_df')
            if orig_df is not None and 'velocity_north_ms' in orig_df.columns and 'velocity_east_ms' in orig_df.columns:
                original_speeds = np.sqrt(orig_df['velocity_north_ms']**2 + orig_df['velocity_east_ms']**2).values
                print(f"从原始数据中提取了速度: {len(original_speeds)}个点")
            
        # 2. 初始化状态
        current_state = self._init_vehicle_state(initial_state)
        trajectory_points = []  # 存储轨迹点
        dt = sim_params.get('dt_sim', 0.25)
        
        # 当使用所有原始点时，目标索引的步进会更小
        target_step = 1
        if len(waypoints) > 1000:
            target_step = max(1, len(waypoints) // 500)  # 动态调整步进，确保生成过程高效
        
        target_waypoint_idx = target_step  # 当前目标航点索引
        
        # 简化的曲率计算，只在关键点处计算
        path_curvatures = None
        if len(waypoints) < 5000:
            path_curvatures = self._calculate_path_curvatures(waypoints)
        else:
            print("轨迹点过多，跳过预计算曲率")
            path_curvatures = [0.0] * len(waypoints)
        
        # 记录初始状态
        # 如果有原始时间戳，使用第一个时间戳
        if use_original_timestamps and original_timestamps is not None and len(original_timestamps) > 0:
            current_state.timestamp = original_timestamps[0]
            
        self._record_state(trajectory_points, current_state, goal_id)
        
        # 4. 主循环
        # 使用更高效的算法跟踪轨迹，适合点数多的情况
        while target_waypoint_idx < len(waypoints):
            target_waypoint = np.array(waypoints[target_waypoint_idx])
            
            if len(trajectory_points) % 1000 == 0:  # 每1000步打印一次
                print(f"\n当前位置: ({current_state.position[0]:.2f}, {current_state.position[1]:.2f})")
                print(f"目标航点索引: {target_waypoint_idx}/{len(waypoints)}")
                print(f"目标航点: ({target_waypoint[0]:.2f}, {target_waypoint[1]:.2f})")
                if match_original_count:
                    progress = len(trajectory_points) / target_points_count * 100 if target_points_count > 0 else 0
                    print(f"轨迹点进度: {len(trajectory_points)}/{target_points_count} ({progress:.1f}%)")
                    
                    # 如果超出目标点数，提前结束
                    if len(trajectory_points) >= target_points_count:
                        print("已达到目标点数量，停止生成")
                        break
            
            # 计算到目标航点的向量和距离
            vec_to_target = target_waypoint - current_state.position
            dist_to_target = np.linalg.norm(vec_to_target)
            
            # 检查是否到达当前目标航点
            if dist_to_target < control_params.get('waypoint_arrival_threshold', 5.0):
                # 增加目标航点索引，使用步进以提高效率
                target_waypoint_idx += target_step
                
                # 如果接近终点，使用更小的步进以确保精确跟踪
                if target_waypoint_idx > len(waypoints) - 50:
                    target_step = 1
                
                if target_waypoint_idx < len(waypoints):
                    if len(trajectory_points) % 1000 == 0:
                        print(f"到达航点，前往下一个索引: {target_waypoint_idx}")
                continue
            
            # 查询当前环境状态
            env_state = self._get_environment_state(env_maps, current_state.position)
            
            # 获取当前参考速度（如果有）
            reference_speed = None
            if original_speeds is not None and len(trajectory_points) < len(original_speeds):
                reference_speed = original_speeds[len(trajectory_points)]
            
            # 计算目标速度 (考虑地形和坡度)
            base_target_speed = self._calculate_target_speed(
                env_state, current_state, generation_rules, control_params, reference_speed
            )
            
            # 应用曲率调整 (考虑转弯)
            curvature = self._get_current_curvature(target_waypoint_idx, path_curvatures)
            adjusted_target_speed = self._adjust_speed_for_curvature(
                base_target_speed, curvature, control_params
            )
            
            # 如果需要匹配原始轨迹长度，调整速度
            if match_original_count and target_points_count > 0:
                # 计算剩余路程和剩余点数
                remaining_points = target_points_count - len(trajectory_points)
                
                if remaining_points > 0:
                    # 估算剩余距离，使用更高效的方法
                    remaining_distance = dist_to_target
                    if len(waypoints) - target_waypoint_idx > 10:
                        # 采样计算距离，不必计算每个点
                        sample_step = max(1, (len(waypoints) - target_waypoint_idx) // 10)
                        for i in range(target_waypoint_idx + sample_step, len(waypoints), sample_step):
                            remaining_distance += np.linalg.norm(
                                np.array(waypoints[i]) - np.array(waypoints[i-sample_step])
                            )
                        
                        # 调整采样偏差
                        remaining_distance *= (len(waypoints) - target_waypoint_idx) / (len(waypoints) - target_waypoint_idx) * 10 / ((len(waypoints) - target_waypoint_idx) // sample_step)
                    
                    # 计算平均每点的距离
                    avg_distance_per_point = remaining_distance / remaining_points
                    
                    # 计算目标dt
                    target_dt = dt
                    if use_original_timestamps and original_timestamps is not None:
                        if len(trajectory_points) < len(original_timestamps):
                            next_timestamp = original_timestamps[len(trajectory_points)]
                            target_dt = next_timestamp - current_state.timestamp
                            target_dt = max(0.001, target_dt)  # 确保dt为正值
                    
                    # 计算调整后的速度
                    target_speed = avg_distance_per_point / target_dt
                    
                    # 使用平滑的调整倍数
                    max_adjustment = 2.0  # 最大调整倍数
                    min_adjustment = 0.5  # 最小调整倍数
                    
                    # 如果有参考速度，优先使用
                    if reference_speed is not None:
                        # 混合使用参考速度和计算速度
                        blend_factor = 0.7  # 参考速度的权重
                        target_speed = reference_speed * blend_factor + target_speed * (1 - blend_factor)
                    
                    speed_ratio = target_speed / adjusted_target_speed if adjusted_target_speed > 0 else 1.0
                    speed_ratio = max(min_adjustment, min(speed_ratio, max_adjustment))
                    
                    # 应用调整，使用平滑函数避免突变
                    adjusted_target_speed *= speed_ratio
            
            # 计算目标航向
            target_heading = np.degrees(np.arctan2(vec_to_target[1], vec_to_target[0]))
            turn_rate = self._calculate_turn_rate(
                current_state.heading, target_heading,
                control_params
            )
            
            # 设置时间步长
            current_dt = dt
            if use_original_timestamps and original_timestamps is not None:
                if len(trajectory_points) < len(original_timestamps):
                    # 使用原始轨迹的时间步长
                    next_timestamp = original_timestamps[len(trajectory_points)]
                    current_dt = next_timestamp - current_state.timestamp
                    # 确保dt为正值
                    current_dt = max(0.001, current_dt)
            
            # 更新状态
            new_state = self._update_state(current_state, adjusted_target_speed, turn_rate, current_dt)
            current_state = new_state
            
            # 记录状态
            self._record_state(trajectory_points, current_state, goal_id)
            
            # 检查是否需要提前结束（基于点数）
            if match_original_count and len(trajectory_points) >= target_points_count:
                print(f"已生成足够的轨迹点 ({len(trajectory_points)}/{target_points_count})，提前结束")
                break
        
        print(f"\n轨迹生成完成，总步数: {len(trajectory_points)}")
        
        # 5. 生成输出数据框
        generated_df = self._create_output_dataframe(trajectory_points)
        
        # 6. 如果需要，使用原始轨迹的时间戳
        if use_original_timestamps and original_timestamps is not None:
            if len(generated_df) != len(original_timestamps):
                print(f"警告: 生成的轨迹点数 ({len(generated_df)}) 与原始时间戳数量 ({len(original_timestamps)}) 不匹配")
                print("将通过重采样匹配时间戳...")
                
                # 重采样轨迹点以匹配原始时间戳数量
                generated_df = self._resample_trajectory(generated_df, len(original_timestamps))
            
            # 替换时间戳
            print("应用原始时间戳...")
            generated_df['timestamp_ms'] = original_timestamps * 1000  # 转换为毫秒
            
        return generated_df
    
    def _init_vehicle_state(self, initial_state: Dict[str, Any]) -> VehicleState:
        """初始化车辆状态"""
        return VehicleState(
            position=np.array([initial_state['x0'], initial_state['y0']]),
            velocity=np.array([
                initial_state.get('vx0', 0.0),
                initial_state.get('vy0', 0.0)
            ]),
            heading=initial_state.get('heading0', 0.0),
            acceleration=np.array([0.0, 0.0]),
            timestamp=0.0
        )
    
    def _get_environment_state(self, env_maps: Any, position: np.ndarray) -> EnvironmentState:
        """获取给定位置的环境状态"""
        features = env_maps.query_by_xy(position[0], position[1])
        return EnvironmentState(
            landcover=features.get('landcover', 90),  # 默认为其他类型
            slope_magnitude=features.get('slope_magnitude', 0.0),
            slope_aspect=features.get('slope_aspect', 0.0)
        )
    
    def _calculate_target_speed(
        self,
        env_state: EnvironmentState,
        state: VehicleState,
        generation_rules: Dict[str, Any],
        control_params: Dict[str, Any],
        reference_speed: float = None
    ) -> float:
        """计算目标速度"""
        landcover = env_state.landcover
        slope_magnitude = env_state.slope_magnitude
        
        # 如果有原始参考速度，优先使用
        if reference_speed is not None and control_params.get('use_original_speeds', False):
            # 使用参考速度，但仍考虑环境因素进行小幅调整
            base_speed = reference_speed
            
            # 应用坡度影响的小幅调整
            traj_id = generation_rules.get('trajectory_id', 1)
            slope_factor = 0.3  # 增加坡度影响因子，更好地反映分析结果
            
            if traj_id in self.trajectory_specific_models and landcover in self.trajectory_specific_models[traj_id]:
                model = self.trajectory_specific_models[traj_id][landcover]
                slope_effect = model['slope'] * slope_magnitude * slope_factor
                base_speed += slope_effect
            elif landcover in self.slope_speed_models:
                model = self.slope_speed_models[landcover]
                slope_effect = model['slope'] * slope_magnitude * slope_factor
                base_speed += slope_effect
                
            # 全局速度乘数
            speed_multiplier = control_params.get('global_speed_multiplier', 0.9)  # 提高乘数，使速度更接近原始数据
            target_speed = base_speed * speed_multiplier
            
        else:
            # 获取当前轨迹ID
            traj_id = generation_rules.get('trajectory_id', 1)
            
            # 1. 尝试使用特定轨迹的模型
            if traj_id in self.trajectory_specific_models and landcover in self.trajectory_specific_models[traj_id]:
                model = self.trajectory_specific_models[traj_id][landcover]
                target_speed = model['slope'] * slope_magnitude + model['intercept']
                
                # 从残差分布中添加变异
                residual = self._sample_from_residual_distribution(traj_id, landcover)
                target_speed += residual
                
            # 2. 退化到通用模型
            elif landcover in self.slope_speed_models:
                model = self.slope_speed_models[landcover]
                target_speed = model['slope'] * slope_magnitude + model['intercept']
                
                # 添加随机变异（基于学习的标准差）
                if 'std' in model:
                    variation = np.random.normal(0, model['std'] * 0.6)  # 增加随机变异比例以更好地匹配分析结果
                    target_speed += variation
            else:
                # 3. 回退到基于规则的计算
                # 获取地类基准速度
                base_speeds = generation_rules.get('base_speeds', self.DEFAULT_BASE_SPEEDS)
                base_speed = base_speeds.get(landcover, base_speeds.get(90, 3.5))
                
                # 应用坡度影响
                slope_coeffs = generation_rules.get('slope_coefficients', self.DEFAULT_SLOPE_COEFFICIENTS)
                slope_coeff = slope_coeffs.get(landcover, -0.028)
                
                # 使用非线性坡度影响函数，更好地模拟实际数据中观察到的模式
                # 当坡度较大时，影响更加显著
                slope_factor = 1.0
                if abs(slope_magnitude) > 15:
                    slope_factor = 1.2  # 陡坡时加大影响
                elif abs(slope_magnitude) > 7:
                    slope_factor = 1.1  # 中等坡度时轻微增加影响
                
                target_speed = base_speed + slope_coeff * slope_magnitude * slope_factor
                
            # 全局速度乘数
            speed_multiplier = control_params.get('global_speed_multiplier', 0.9)  # 提高乘数，更接近分析结果
            target_speed *= speed_multiplier
        
        # 速度限幅
        max_speed = control_params.get('max_speed', 10.0)  # 提高最大速度限制
        min_speed = control_params.get('min_speed', 1.0)   # 设置最小速度限制，防止速度过低
        target_speed = max(min_speed, min(target_speed, max_speed))
        
        return target_speed
    
    def _calculate_turn_rate(
        self,
        current_heading: float,
        target_heading: float,
        control_params: Dict[str, Any]
    ) -> float:
        """计算转向速率(度/秒)"""
        # 计算航向差，选择最短的转向路径
        heading_diff = target_heading - current_heading
        if heading_diff > 180:
            heading_diff -= 360
        elif heading_diff < -180:
            heading_diff += 360
        
        # 简单的比例控制
        k_p = control_params.get('turn_p_gain', 1.2)
        raw_turn_rate = heading_diff * k_p
        
        # 限制转向速率
        max_turn_rate = control_params.get('max_turn_rate', 45.0)  # 度/秒
        return max(-max_turn_rate, min(raw_turn_rate, max_turn_rate))
    
    def _update_state(
        self,
        state: VehicleState,
        target_speed: float,
        turn_rate: float,
        dt: float
    ) -> VehicleState:
        """更新车辆状态"""
        # 1. 更新航向(确保在0-360度范围内)
        new_heading = state.heading + turn_rate * dt
        while new_heading >= 360:
            new_heading -= 360
        while new_heading < 0:
            new_heading += 360
        
        # 2. 直接使用目标速度，不再计算加速度
        new_velocity = np.array([
            target_speed * np.cos(np.radians(new_heading)),
            target_speed * np.sin(np.radians(new_heading))
        ])
        
        # 3. 更新位置
        new_position = state.position + new_velocity * dt
        
        # 4. 计算加速度(仅用于记录)
        acceleration = (new_velocity - state.velocity) / dt
        
        return VehicleState(
            position=new_position,
            velocity=new_velocity,
            heading=new_heading,
            acceleration=acceleration,
            timestamp=state.timestamp + dt
        )
    
    def _record_state(
        self,
        points: List[Tuple],
        state: VehicleState,
        goal_id: int
    ) -> None:
        """记录当前状态
        
        Args:
            points: 轨迹点列表
            state: 当前状态
            goal_id: 目标点ID
        """
        points.append((
            int(state.timestamp * 1000),  # timestamp_ms
            state.position[0],  # x
            state.position[1],  # y
            state.velocity[1],  # velocity_north_ms (y方向)
            state.velocity[0],  # velocity_east_ms (x方向)
            state.heading,  # heading_deg
            state.acceleration[1],  # acceleration_north_ms2 (y方向)
            state.acceleration[0],  # acceleration_east_ms2 (x方向)
            goal_id  # goal_id
        ))
    
    def _create_output_dataframe(self, points: List[Tuple]) -> pd.DataFrame:
        """创建输出数据框
        
        Args:
            points: 轨迹点列表
            
        Returns:
            pd.DataFrame: 轨迹数据框
        """
        df = pd.DataFrame(points, columns=[
            'timestamp_ms',
            'x',
            'y',
            'velocity_north_ms',
            'velocity_east_ms',
            'heading_deg',
            'acceleration_north_ms2',
            'acceleration_east_ms2',
            'goal_id'
        ])
        
        # 添加UTM坐标列
        df['utm_x'] = df['x']
        df['utm_y'] = df['y']
        
        return df
    
    def _calculate_path_curvatures(self, waypoints: List[Tuple[float, float]]) -> List[float]:
        """计算路径的曲率序列
        
        Args:
            waypoints: 航点列表
            
        Returns:
            List[float]: 曲率列表
        """
        if len(waypoints) < 3:
            return [0.0] * len(waypoints)
            
        curvatures = []
        
        # 第一个点的曲率为0
        curvatures.append(0.0)
        
        # 计算中间点的曲率
        for i in range(1, len(waypoints) - 1):
            # 使用三点法估计曲率
            p1 = np.array(waypoints[i-1])
            p2 = np.array(waypoints[i])
            p3 = np.array(waypoints[i+1])
            
            # 计算三个点的距离
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p3 - p1)
            
            # 防止除零错误
            if a * b * c == 0:
                curvatures.append(0.0)
                continue
                
            # 使用海伦公式计算三角形面积
            s = (a + b + c) / 2
            try:
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            except:
                # 如果计算失败，可能是点共线
                curvatures.append(0.0)
                continue
                
            # 曲率 = 4 * 面积 / (a * b * c)
            # 曲率半径 = 1 / 曲率
            if area == 0:
                curvatures.append(0.0)
            else:
                # 增强曲率感知，使用更敏感的指数函数
                raw_curvature = 4 * area / (a * b * c)
                
                # 计算夹角，获得更直接的转弯度量
                v1 = p1 - p2
                v2 = p3 - p2
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                
                # 结合曲率和角度，创建更敏感的曲率度量
                # 当角度接近180度（直线）时，曲率接近0
                # 当角度接近0度（急转弯）时，曲率接近1
                enhanced_curvature = raw_curvature * (1 - np.cos(angle))
                
                curvatures.append(enhanced_curvature)
        
        # 最后一个点的曲率与倒数第二个点相同
        curvatures.append(curvatures[-1])
        
        return curvatures
    
    def _get_current_curvature(self, waypoint_idx: int, curvatures: List[float]) -> float:
        """获取当前位置的曲率
        
        Args:
            waypoint_idx: 当前目标航点索引
            curvatures: 预计算的曲率列表
            
        Returns:
            float: 当前位置的曲率
        """
        if 0 <= waypoint_idx < len(curvatures):
            return curvatures[waypoint_idx]
        return 0.0
    
    def _adjust_speed_for_curvature(
        self, 
        base_speed: float, 
        curvature: float,
        control_params: Dict[str, Any]
    ) -> float:
        """根据路径曲率调整速度
        
        曲率越大，速度越低；直线段曲率为0，保持原速
        
        Args:
            base_speed: 基础目标速度
            curvature: 当前路径曲率
            control_params: 控制参数
            
        Returns:
            float: 调整后的速度
        """
        # 曲率影响因子 (0-1)，值越大影响越明显
        curvature_factor = control_params.get('curvature_factor', 0.65)  # 增强曲率影响
        
        # 标准化曲率范围，使其更合理地影响速度
        # 根据分析结果，更敏感地响应曲率变化
        normalized_curvature = min(curvature * 4.0, 1.0)  # 增加曲率敏感度
        
        # 使用指数函数增强对高曲率的响应，更符合分析中观察到的转弯特性
        # 使用二次函数，使得低曲率影响较小，高曲率影响显著增加
        curve_effect = normalized_curvature ** 1.8  # 使用稍低的指数，使中等曲率也有一定影响
        
        # 计算速度调整因子 (曲率越大，速度越低)
        # 根据分析结果，转弯时速度可能降低到直线时的40-70%
        min_turn_speed_ratio = 0.4  # 最小转弯速度比例
        speed_factor = 1.0 - (curve_effect * curvature_factor)
        
        # 应用调整，但确保转弯速度不会低于某个阈值
        adjusted_speed = base_speed * max(min_turn_speed_ratio, speed_factor)
        
        # 确保速度不会过低
        min_speed = max(0.4 * base_speed, control_params.get('min_speed', 1.0))  # 更低的最小速度，更真实地模拟急转弯
        
        # 转弯速度限幅
        return max(min_speed, adjusted_speed)
    
    def _resample_trajectory(self, df: pd.DataFrame, target_length: int) -> pd.DataFrame:
        """重采样轨迹以匹配目标长度
        
        Args:
            df: 原始轨迹数据框
            target_length: 目标长度
            
        Returns:
            pd.DataFrame: 重采样后的轨迹
        """
        if len(df) == target_length:
            return df
            
        print(f"重采样轨迹: {len(df)} → {target_length}点")
        
        # 创建归一化索引
        orig_indices = np.linspace(0, 1, len(df))
        target_indices = np.linspace(0, 1, target_length)
        
        # 为每列创建插值器
        new_data = {}
        for column in df.columns:
            if column == 'goal_id':
                # 对于goal_id使用最近邻插值
                new_data[column] = np.interp(
                    target_indices,
                    orig_indices,
                    df[column].values,
                    left=df[column].iloc[0],
                    right=df[column].iloc[-1]
                ).astype(int)
            else:
                # 对于其他列使用线性插值
                new_data[column] = np.interp(
                    target_indices,
                    orig_indices,
                    df[column].values
                )
        
        return pd.DataFrame(new_data) 