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
    
    # 默认的地类基准速度(m/s)，来自数据分析
    DEFAULT_BASE_SPEEDS = {
        20: 5.8,  # 林地
        40: 4.8,  # 灌木地
        60: 4.0,  # 水体
        90: 0.5   # 其他
    }
    
    # 默认的坡度影响系数，来自数据分析
    DEFAULT_SLOPE_COEFFICIENTS = {
        20: -0.020,  # 林地
        40: -0.010,  # 灌木地
        60: -0.015   # 水体
    }
    
    def __init__(self):
        """初始化转换器"""
        self.utm_to_wgs84 = pyproj.Transformer.from_crs(
            "EPSG:32630", "EPSG:4326", always_xy=True
        )
        
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
        print(f"航点数量: {len(waypoints)}")
        print(f"初始状态: {initial_state}")
        
        # 1. 使用默认参数
        if generation_rules is None:
            generation_rules = {
                'base_speeds': self.DEFAULT_BASE_SPEEDS,
                'slope_coefficients': self.DEFAULT_SLOPE_COEFFICIENTS
            }
            
        if control_params is None:
            control_params = {
                'global_speed_multiplier': 1.0,
                'max_speed': 8.0,
                'max_acceleration': 2.0,
                'max_deceleration': 3.0,
                'max_turn_rate': 45.0,  # 增加最大转向速率
                'speed_p_gain': 1.2,    # 增加速度控制响应
                'turn_p_gain': 1.2,     # 增加转向控制响应
                'waypoint_arrival_threshold': 10.0  # 增加到达阈值
            }
            
        if sim_params is None:
            sim_params = {'dt_sim': 0.25}  # 4Hz
            
        # 2. 初始化状态
        current_state = self._init_vehicle_state(initial_state)
        trajectory_points = []  # 存储轨迹点
        dt = sim_params.get('dt_sim', 0.25)  # 仿真时间步长(默认4Hz)
        target_waypoint_idx = 1  # 当前目标航点索引
        
        # 记录初始状态
        self._record_state(trajectory_points, current_state, goal_id)
        
        # 3. 主循环
        while target_waypoint_idx < len(waypoints):
            target_waypoint = np.array(waypoints[target_waypoint_idx])
            
            if len(trajectory_points) % 1000 == 0:  # 每1000步打印一次
                print(f"\n当前位置: ({current_state.position[0]:.2f}, {current_state.position[1]:.2f})")
                print(f"目标航点: ({target_waypoint[0]:.2f}, {target_waypoint[1]:.2f})")
            
            # 计算到目标航点的向量和距离
            vec_to_target = target_waypoint - current_state.position
            dist_to_target = np.linalg.norm(vec_to_target)
            
            if len(trajectory_points) % 1000 == 0:
                print(f"到目标点距离: {dist_to_target:.2f}m")
            
            # 检查是否到达当前目标航点
            if dist_to_target < control_params.get('waypoint_arrival_threshold', 10.0):
                target_waypoint_idx += 1
                if target_waypoint_idx < len(waypoints):
                    print(f"\n到达航点{target_waypoint_idx-1}，前往下一个航点")
                continue
            
            # 查询当前环境状态
            env_state = self._get_environment_state(env_maps, current_state.position)
            
            # 计算目标速度
            target_speed = self._calculate_target_speed(
                env_state, current_state, generation_rules, control_params
            )
            
            # 计算目标航向：使用arctan2(y, x)而不是arctan2(x, y)
            target_heading = np.degrees(np.arctan2(vec_to_target[1], vec_to_target[0]))
            turn_rate = self._calculate_turn_rate(
                current_state.heading, target_heading,
                control_params
            )
            
            if len(trajectory_points) % 1000 == 0:
                print(f"当前航向: {current_state.heading:.2f}度")
                print(f"目标航向: {target_heading:.2f}度")
                print(f"转向速率: {turn_rate:.2f}度/秒")
            
            # 更新状态
            new_state = self._update_state(current_state, target_speed, turn_rate, dt)
            current_state = new_state
            
            # 记录状态
            self._record_state(trajectory_points, current_state, goal_id)
            
        print(f"\n轨迹生成完成，总步数: {len(trajectory_points)}")
        
        # 4. 生成输出数据框
        return self._create_output_dataframe(trajectory_points)
    
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
        vehicle_state: VehicleState,
        generation_rules: Dict[str, Any],
        control_params: Dict[str, Any]
    ) -> float:
        """计算目标速度"""
        # 1. 获取基准速度
        base_speeds = generation_rules.get('base_speeds', self.DEFAULT_BASE_SPEEDS)
        base_speed = base_speeds.get(env_state.landcover, base_speeds[90])
        
        # 2. 计算坡度影响
        slope_coeffs = generation_rules.get('slope_coefficients', 
                                          self.DEFAULT_SLOPE_COEFFICIENTS)
        slope_coeff = slope_coeffs.get(env_state.landcover, 0.0)
        
        # 计算当前航向与坡向的夹角
        heading_rad = np.radians(vehicle_state.heading)
        aspect_rad = np.radians(env_state.slope_aspect)
        delta_angle = heading_rad - aspect_rad
        
        # 计算有效坡度(考虑行进方向)
        effective_slope = env_state.slope_magnitude * np.cos(delta_angle)
        
        # 3. 应用坡度影响
        target_speed = base_speed * (1 + slope_coeff * effective_slope)
        
        # 4. 应用全局速度系数
        target_speed *= control_params.get('global_speed_multiplier', 1.0)
        
        # 5. 应用速度限制
        max_speed = control_params.get('max_speed', 8.0)
        target_speed = max(0.0, min(target_speed, max_speed))
        
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