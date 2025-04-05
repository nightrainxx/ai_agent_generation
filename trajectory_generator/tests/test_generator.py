"""轨迹生成器测试模块"""

import unittest
import numpy as np
from src.generator import TrajectoryGenerator, EnvironmentState, VehicleState

class MockEnvironmentMaps:
    """模拟环境地图类"""
    def query_by_xy(self, x: float, y: float) -> dict:
        """模拟环境查询"""
        return {
            'landcover': 40,  # 灌木地
            'slope_magnitude': 5.0,  # 5度坡度
            'slope_aspect': 45.0  # 东北方向
        }

class TestTrajectoryGenerator(unittest.TestCase):
    """轨迹生成器测试类"""
    
    def setUp(self):
        """测试准备"""
        self.generator = TrajectoryGenerator()
        self.env_maps = MockEnvironmentMaps()
        
        # 测试用航点序列
        self.waypoints = [
            (0.0, 0.0),  # 起点
            (100.0, 100.0),  # 终点
        ]
        
        # 初始状态
        self.initial_state = {
            'x0': 0.0,
            'y0': 0.0,
            'vx0': 0.0,
            'vy0': 0.0,
            'heading0': 45.0  # 朝向东北
        }
        
        # 生成规则
        self.generation_rules = {
            'base_speeds': {
                20: 5.8,  # 林地
                40: 4.8,  # 灌木地
                60: 4.0,  # 水体
                90: 0.5   # 其他
            },
            'slope_coefficients': {
                20: -0.020,
                40: -0.010,
                60: -0.015
            }
        }
        
        # 控制参数
        self.control_params = {
            'global_speed_multiplier': 1.0,
            'max_speed': 8.0,
            'max_acceleration': 2.0,
            'max_deceleration': 3.0,
            'max_turn_rate': 45.0,
            'speed_p_gain': 0.5,
            'turn_p_gain': 0.5,
            'waypoint_arrival_threshold': 10.0
        }
        
        # 仿真参数
        self.sim_params = {
            'dt_sim': 0.25  # 4Hz
        }
        
    def test_generate_basic_trajectory(self):
        """测试基本轨迹生成"""
        # 生成轨迹
        trajectory_df = self.generator.generate(
            self.waypoints,
            self.initial_state,
            goal_id=0,
            env_maps=self.env_maps,
            generation_rules=self.generation_rules,
            control_params=self.control_params,
            sim_params=self.sim_params
        )
        
        # 基本检查
        self.assertIsNotNone(trajectory_df)
        self.assertGreater(len(trajectory_df), 0)
        
        # 检查必要的列
        required_columns = [
            'timestamp_ms',
            'latitude',
            'longitude',
            'velocity_north_ms',
            'velocity_east_ms',
            'heading_deg',
            'acceleration_x_ms2',
            'acceleration_y_ms2',
            'goal_id'
        ]
        for col in required_columns:
            self.assertIn(col, trajectory_df.columns)
            
        # 检查时间戳递增
        self.assertTrue(
            (trajectory_df['timestamp_ms'].diff()[1:] > 0).all()
        )
        
        # 检查速度限制
        speeds = np.sqrt(
            trajectory_df['velocity_north_ms']**2 +
            trajectory_df['velocity_east_ms']**2
        )
        self.assertTrue(
            (speeds <= self.control_params['max_speed']).all()
        )
        
    def test_speed_adjustment(self):
        """测试速度调整"""
        # 使用不同的速度倍率
        fast_params = self.control_params.copy()
        fast_params['global_speed_multiplier'] = 2.0
        
        slow_params = self.control_params.copy()
        slow_params['global_speed_multiplier'] = 0.5
        
        # 生成不同速度的轨迹
        normal_traj = self.generator.generate(
            self.waypoints,
            self.initial_state,
            goal_id=0,
            env_maps=self.env_maps,
            generation_rules=self.generation_rules,
            control_params=self.control_params,
            sim_params=self.sim_params
        )
        
        fast_traj = self.generator.generate(
            self.waypoints,
            self.initial_state,
            goal_id=0,
            env_maps=self.env_maps,
            generation_rules=self.generation_rules,
            control_params=fast_params,
            sim_params=self.sim_params
        )
        
        slow_traj = self.generator.generate(
            self.waypoints,
            self.initial_state,
            goal_id=0,
            env_maps=self.env_maps,
            generation_rules=self.generation_rules,
            control_params=slow_params,
            sim_params=self.sim_params
        )
        
        # 计算平均速度
        def calc_mean_speed(df):
            return np.mean(np.sqrt(
                df['velocity_north_ms']**2 +
                df['velocity_east_ms']**2
            ))
        
        normal_speed = calc_mean_speed(normal_traj)
        fast_speed = calc_mean_speed(fast_traj)
        slow_speed = calc_mean_speed(slow_traj)
        
        # 验证速度关系
        self.assertGreater(fast_speed, normal_speed)
        self.assertLess(slow_speed, normal_speed)
        
if __name__ == '__main__':
    unittest.main() 