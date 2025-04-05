#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
R²分布分析脚本
- 分析不同时间窗口的R²分布
- 确定最佳聚合窗口大小
- 输出分析结果和可视化图表
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 设置中文字体
try:
    # 尝试使用系统中文字体
    chinese_font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
except:
    # 如果找不到，回退到默认字体
    chinese_font = FontProperties()

# 定义分析目录和输出目录
analysis_dir = "analysisi_result"
output_dir = "r2_analysis"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 用于存储不同时间窗口和地类的R²值
r2_values = {
    "1秒": {"林地": [], "灌木地": [], "水体": []},
    "2秒": {"林地": [], "灌木地": [], "水体": []},
    "5秒": {"林地": [], "灌木地": [], "水体": []},
    "10秒": {"林地": [], "灌木地": [], "水体": []},
    "30秒": {"林地": [], "灌木地": [], "水体": []},
    "60秒": {"林地": [], "灌木地": [], "水体": []}
}

# 用于存储不同轨迹的R²值
trajectory_r2 = {
    "轨迹1": {"林地": {}, "灌木地": {}, "水体": {}},
    "轨迹2": {"林地": {}, "灌木地": {}, "水体": {}},
    "轨迹3": {"林地": {}, "灌木地": {}, "水体": {}},
    "轨迹4": {"林地": {}, "灌木地": {}, "水体": {}}
}

def extract_r2_from_csv(file_path):
    """从CSV文件中提取R²值"""
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 查找R²值 - 可能位于不同列名中
        r2_column = None
        for col in ['binned_r2', 'r2', 'R2', 'r_squared']:
            if col in df.columns:
                r2_column = col
                break
        
        if r2_column is None:
            # 如果没有明确的R²列，检查第一行是否包含值
            if 'regression_coef' in df.columns and 'regression_intercept' in df.columns:
                # 提取回归系数和截距
                slope = df['regression_coef'].iloc[0]
                intercept = df['regression_intercept'].iloc[0]
                
                # 提取斜率值和速度值
                slope_values = []
                for bin_str in df['slope_bin'].values:
                    try:
                        bin_str = bin_str.strip('"')
                        bin_range = bin_str.strip('()[]')
                        low, high = map(float, bin_range.split(','))
                        slope_values.append((low + high) / 2)
                    except:
                        if 'mean_slope' in df.columns:
                            slope_values = df['mean_slope'].values
                            break
                        else:
                            return None
                
                x = np.array(slope_values)
                y = df['mean_speed'].values
                
                # 过滤掉NaN值
                valid_idx = ~(np.isnan(x) | np.isnan(y))
                x = x[valid_idx]
                y = y[valid_idx]
                
                if len(x) < 2:
                    return None
                
                # 计算预测值
                y_pred = slope * x + intercept
                
                # 计算R²
                ss_total = np.sum((y - np.mean(y))**2)
                ss_residual = np.sum((y - y_pred)**2)
                r2 = 1 - (ss_residual / ss_total)
                
                return r2
            else:
                print(f"文件 {file_path} 中未找到R²值")
                return None
        else:
            # 获取第一个非NaN的R²值
            r2_values = df[r2_column].dropna().values
            if len(r2_values) > 0:
                return r2_values[0]
            else:
                print(f"文件 {file_path} 中R²值全为NaN")
                return None
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_r2_boxplot():
    """绘制不同时间窗口的R²箱线图"""
    plt.figure(figsize=(12, 8))
    
    # 准备数据
    data = []
    for window in r2_values.keys():
        for land_type in r2_values[window].keys():
            for r2 in r2_values[window][land_type]:
                if r2 is not None:
                    data.append({
                        'time_window': window,
                        'land_type': land_type,
                        'r2': r2
                    })
    
    if not data:
        print("没有有效的R²数据进行绘图")
        return
    
    df = pd.DataFrame(data)
    
    # 绘制箱线图
    sns.boxplot(x='time_window', y='r2', hue='land_type', data=df)
    plt.title('不同时间窗口和地类的R²分布', fontproperties=chinese_font)
    plt.xlabel('时间窗口', fontproperties=chinese_font)
    plt.ylabel('R²值', fontproperties=chinese_font)
    plt.legend(title='地类', prop=chinese_font)
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "R2_箱线图.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_r2_heatmap():
    """绘制时间窗口和地类的R²热力图"""
    plt.figure(figsize=(12, 8))
    
    # 准备数据
    data = np.zeros((len(r2_values), len(r2_values["1秒"])))
    windows = list(r2_values.keys())
    land_types = list(r2_values["1秒"].keys())
    
    for i, window in enumerate(windows):
        for j, land_type in enumerate(land_types):
            values = [v for v in r2_values[window][land_type] if v is not None]
            if values:
                data[i, j] = np.mean(values)
            else:
                data[i, j] = np.nan
    
    # 绘制热力图
    ax = sns.heatmap(data, annot=True, fmt=".3f", cmap="YlGnBu",
                   xticklabels=land_types, yticklabels=windows)
    plt.title('各时间窗口和地类的平均R²值', fontproperties=chinese_font)
    plt.xlabel('地类', fontproperties=chinese_font)
    plt.ylabel('时间窗口', fontproperties=chinese_font)
    
    # 设置中文标签
    ax.set_xticklabels(ax.get_xticklabels(), font=chinese_font.get_name())
    ax.set_yticklabels(ax.get_yticklabels(), font=chinese_font.get_name())
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "R2_热力图.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_trajectory_r2():
    """绘制各轨迹在不同时间窗口的R²变化"""
    for trajectory in trajectory_r2.keys():
        plt.figure(figsize=(12, 8))
        
        for land_type in trajectory_r2[trajectory].keys():
            windows = []
            r2s = []
            
            for window in sorted(trajectory_r2[trajectory][land_type].keys(), 
                               key=lambda x: int(x.replace('秒', ''))):
                windows.append(window)
                r2s.append(trajectory_r2[trajectory][land_type][window])
            
            if windows and any(r2 is not None for r2 in r2s):
                plt.plot(windows, r2s, 'o-', label=land_type)
        
        plt.title(f'{trajectory}在不同时间窗口的R²变化', fontproperties=chinese_font)
        plt.xlabel('时间窗口', fontproperties=chinese_font)
        plt.ylabel('R²值', fontproperties=chinese_font)
        plt.legend(prop=chinese_font)
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{trajectory}_R2变化.png"), dpi=300, bbox_inches='tight')
        plt.close()

def create_r2_report():
    """创建R²分析报告"""
    report = ["# R²分布分析报告", ""]
    
    # 添加总体分析
    report.append("## 各时间窗口R²统计")
    report.append("")
    report.append("| 时间窗口 | 地类 | 样本数 | 平均R² | 标准差 | 最小值 | 最大值 |")
    report.append("| -------- | ---- | ------ | ------ | ------ | ------ | ------ |")
    
    # 最佳窗口的评分
    window_scores = {}
    
    for window in r2_values.keys():
        window_scores[window] = 0
        for land_type in r2_values[window].keys():
            values = [v for v in r2_values[window][land_type] if v is not None]
            if values:
                n = len(values)
                mean = np.mean(values)
                std = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                # 添加到报告
                report.append(f"| {window} | {land_type} | {n} | {mean:.4f} | {std:.4f} | {min_val:.4f} | {max_val:.4f} |")
                
                # 评分: 基于平均R²和标准差
                window_scores[window] += mean * 1.0 - std * 0.5
            else:
                report.append(f"| {window} | {land_type} | 0 | - | - | - | - |")
    
    # 判断最佳窗口
    best_window = max(window_scores.items(), key=lambda x: x[1])
    
    report.append("")
    report.append("## 最佳聚合窗口分析")
    report.append("")
    report.append("基于R²值和标准差的综合评分，我们得出最佳聚合窗口是：")
    report.append("")
    report.append(f"**{best_window[0]}** (评分: {best_window[1]:.4f})")
    report.append("")
    report.append("评分标准：平均R²值越高越好，标准差越低越好。具体计算公式为：")
    report.append("")
    report.append("```")
    report.append("窗口评分 = 平均R² * 1.0 - 标准差 * 0.5")
    report.append("```")
    report.append("")
    
    # 添加各地类的最佳窗口
    report.append("## 各地类的最佳窗口")
    report.append("")
    
    for land_type in ["林地", "灌木地", "水体"]:
        land_scores = {}
        for window in r2_values.keys():
            values = [v for v in r2_values[window][land_type] if v is not None]
            if values:
                mean = np.mean(values)
                std = np.std(values)
                land_scores[window] = mean * 1.0 - std * 0.5
        
        if land_scores:
            best_land_window = max(land_scores.items(), key=lambda x: x[1])
            report.append(f"### {land_type}")
            report.append("")
            report.append(f"最佳窗口: **{best_land_window[0]}** (评分: {best_land_window[1]:.4f})")
            report.append("")
    
    # 添加轨迹特定分析
    report.append("## 轨迹特定分析")
    report.append("")
    
    for trajectory in trajectory_r2.keys():
        report.append(f"### {trajectory}")
        report.append("")
        report.append("| 地类 | 最佳窗口 | R²值 |")
        report.append("| ---- | -------- | ---- |")
        
        for land_type in trajectory_r2[trajectory].keys():
            if trajectory_r2[trajectory][land_type]:
                best_window_r2 = max(trajectory_r2[trajectory][land_type].items(), 
                                   key=lambda x: x[1] if x[1] is not None else -float('inf'))
                report.append(f"| {land_type} | {best_window_r2[0]} | {best_window_r2[1]:.4f} |")
        
        report.append("")
    
    # 结论和建议
    report.append("## 结论和建议")
    report.append("")
    report.append(f"根据R²分布分析，我们建议使用 **{best_window[0]}** 作为最佳聚合窗口，因为它在多个地类上都表现良好，尤其是在林地和灌木地上。")
    report.append("")
    report.append("对于不同地形类型，可以考虑使用它们各自的最佳窗口：")
    report.append("")
    
    land_windows = {}
    for land_type in ["林地", "灌木地", "水体"]:
        land_scores = {}
        for window in r2_values.keys():
            values = [v for v in r2_values[window][land_type] if v is not None]
            if values:
                mean = np.mean(values)
                std = np.std(values)
                land_scores[window] = mean * 1.0 - std * 0.5
        
        if land_scores:
            best_land_window = max(land_scores.items(), key=lambda x: x[1])
            land_windows[land_type] = best_land_window[0]
            report.append(f"- {land_type}: {best_land_window[0]}")
    
    report.append("")
    report.append("这些窗口大小可以作为各自地形类型速度模型的输入聚合参数。")
    
    # 写入文件
    with open(os.path.join(output_dir, "R2分析报告.md"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    return best_window[0], land_windows

def main():
    """主函数"""
    print("开始分析R²分布...")
    
    # 查找所有分箱统计CSV文件
    csv_files = glob.glob(os.path.join(analysis_dir, '*坡度分箱统计*.csv'))
    
    # 处理每个文件
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"处理文件: {file_name}")
        
        # 提取信息
        try:
            parts = file_name.split('_')
            
            # 确保文件名格式正确
            if len(parts) >= 4:
                trajectory = parts[0]
                time_window = parts[2]
                land_type = parts[3].split('.')[0]
            else:
                # 尝试从文件名中提取信息
                trajectory = '未知轨迹'
                time_window = '1秒'
                land_type = '未知地类'
                
                for part in parts:
                    if part.startswith('轨迹'):
                        trajectory = part
                    elif part.endswith('秒'):
                        time_window = part
                    elif part in ['林地', '灌木地', '水体']:
                        land_type = part
            
            print(f"  解析结果: 轨迹={trajectory}, 时间窗口={time_window}, 地类={land_type}")
            
            # 提取R²值
            r2 = extract_r2_from_csv(file_path)
            if r2 is not None:
                print(f"  提取的R²值: {r2:.4f}")
                
                # 添加到时间窗口字典
                if time_window in r2_values and land_type in r2_values[time_window]:
                    r2_values[time_window][land_type].append(r2)
                
                # 添加到轨迹字典
                if trajectory in trajectory_r2 and land_type in trajectory_r2[trajectory]:
                    trajectory_r2[trajectory][land_type][time_window] = r2
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")
            continue
    
    # 绘制R²分布图
    print("绘制R²分布图...")
    plot_r2_boxplot()
    plot_r2_heatmap()
    plot_trajectory_r2()
    
    # 创建R²分析报告
    print("创建R²分析报告...")
    best_window, land_best_windows = create_r2_report()
    
    print(f"R²分析完成！最佳聚合窗口是: {best_window}")
    print(f"各地类最佳窗口: {land_best_windows}")
    print(f"结果保存在 {output_dir} 目录中")

if __name__ == "__main__":
    main() 