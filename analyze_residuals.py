#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
残差分析和建模脚本
- 分析线性模型的残差分布
- 可视化残差
- 提出残差建模方法
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
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

# 定义分析目录
analysis_dir = "analysisi_result"
output_dir = "residual_analysis"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 用于存储所有残差的字典
all_residuals = {
    "林地": [],
    "灌木地": [],
    "水体": []
}

# 用于存储不同时间窗口的残差
time_window_residuals = {
    "1秒": {"林地": [], "灌木地": [], "水体": []},
    "2秒": {"林地": [], "灌木地": [], "水体": []},
    "5秒": {"林地": [], "灌木地": [], "水体": []},
    "10秒": {"林地": [], "灌木地": [], "水体": []},
    "30秒": {"林地": [], "灌木地": [], "水体": []},
    "60秒": {"林地": [], "灌木地": [], "水体": []}
}

# 用于存储不同轨迹的残差
trajectory_residuals = {
    "轨迹1": {"林地": [], "灌木地": [], "水体": []},
    "轨迹2": {"林地": [], "灌木地": [], "水体": []},
    "轨迹3": {"林地": [], "灌木地": [], "水体": []},
    "轨迹4": {"林地": [], "灌木地": [], "水体": []}
}

def extract_residuals_from_csv(file_path):
    """从CSV文件中提取残差数据"""
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 检查必要的列是否存在
        if 'slope_bin' not in df.columns:
            print(f"文件 {file_path} 缺少slope_bin列")
            return None
            
        # 检查速度列 - 使用mean_speed而不是avg_speed
        speed_column = 'mean_speed'
        if speed_column not in df.columns:
            print(f"文件 {file_path} 缺少{speed_column}列")
            return None
        
        # 提取斜率和截距
        # 需要处理slope_bin这种"(-35, -30]"格式的字符串
        # 将其转换为数值，取区间中点
        slope_values = []
        for bin_str in df['slope_bin'].values:
            try:
                # 处理"(-35, -30]"这样的字符串
                bin_str = bin_str.strip('"')  # 移除引号
                bin_range = bin_str.strip('()[]')  # 移除括号
                low, high = map(float, bin_range.split(','))
                slope_values.append((low + high) / 2)  # 取区间中点
            except:
                # 如果解析失败，尝试使用mean_slope列
                if 'mean_slope' in df.columns:
                    slope_values = df['mean_slope'].values
                    break
                else:
                    raise ValueError(f"无法解析slope_bin: {bin_str}")
        
        x = np.array(slope_values)
        y = df[speed_column].values
        
        # 过滤掉NaN值
        valid_idx = ~(np.isnan(x) | np.isnan(y))
        x = x[valid_idx]
        y = y[valid_idx]
        
        if len(x) < 2:  # 至少需要两个点才能拟合直线
            print(f"文件 {file_path} 有效数据点不足")
            return None
        
        # 计算线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # 计算预测值
        y_pred = slope * x + intercept
        
        # 计算残差
        residuals = y - y_pred
        
        return {
            'file': os.path.basename(file_path),
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            'residuals': residuals,
            'x': x,
            'y': y,
            'y_pred': y_pred
        }
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_residual_distribution(residuals):
    """分析残差分布特征"""
    if len(residuals) == 0:
        return None
    
    # 计算基本统计量
    mean = np.mean(residuals)
    std = np.std(residuals)
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)
    
    # 正态性检验
    shapiro_test = stats.shapiro(residuals)
    
    # 尝试拟合不同的分布
    distributions = ['norm', 'laplace', 'logistic', 't']
    best_dist = None
    best_params = None
    best_aic = float('inf')
    
    for dist_name in distributions:
        try:
            # 拟合分布
            dist = getattr(stats, dist_name)
            params = dist.fit(residuals)
            
            # 计算对数似然
            log_likelihood = np.sum(dist.logpdf(residuals, *params))
            
            # 计算AIC
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            
            if aic < best_aic:
                best_dist = dist_name
                best_params = params
                best_aic = aic
        except:
            continue
    
    # 尝试高斯混合模型
    try:
        gmm = GaussianMixture(n_components=2, random_state=0).fit(residuals.reshape(-1, 1))
        gmm_aic = 2 * gmm.n_components * 2 - 2 * gmm.score(residuals.reshape(-1, 1)) * len(residuals)
        
        if gmm_aic < best_aic:
            best_dist = 'gmm'
            best_params = {
                'means': gmm.means_.flatten(),
                'weights': gmm.weights_,
                'covariances': gmm.covariances_.flatten()
            }
            best_aic = gmm_aic
    except:
        pass
    
    result = {
        'mean': mean,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'shapiro_test': shapiro_test,
        'best_distribution': best_dist,
        'best_params': best_params,
        'best_aic': best_aic
    }
    
    return result

def plot_residual_analysis(data, title, save_path):
    """绘制残差分析图"""
    residuals = data['residuals']
    
    plt.figure(figsize=(15, 12))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False
    
    # 子图1：原始数据和拟合线
    plt.subplot(2, 2, 1)
    plt.scatter(data['x'], data['y'], alpha=0.6)
    plt.plot(data['x'], data['y_pred'], 'r-')
    equation = f"y = {data['slope']:.4f}x + {data['intercept']:.4f}"
    plt.title(f"原始数据与拟合线 (R² = {data['r_squared']:.4f})\n{equation}", fontproperties=chinese_font)
    plt.xlabel("坡度", fontproperties=chinese_font)
    plt.ylabel("速度 (m/s)", fontproperties=chinese_font)
    plt.grid(True, alpha=0.3)
    
    # 子图2：残差散点图
    plt.subplot(2, 2, 2)
    plt.scatter(data['x'], residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title("残差散点图", fontproperties=chinese_font)
    plt.xlabel("坡度", fontproperties=chinese_font)
    plt.ylabel("残差", fontproperties=chinese_font)
    plt.grid(True, alpha=0.3)
    
    # 子图3：残差直方图和密度曲线
    plt.subplot(2, 2, 3)
    sns.histplot(residuals, kde=True, stat="density", bins=20)
    
    # 添加拟合的分布
    analysis = analyze_residual_distribution(residuals)
    x_range = np.linspace(min(residuals), max(residuals), 1000)
    
    if analysis and analysis['best_distribution'] == 'gmm':
        # 绘制GMM分布
        gmm_pdf = np.zeros_like(x_range)
        for i, (mean, weight, cov) in enumerate(zip(
            analysis['best_params']['means'],
            analysis['best_params']['weights'],
            analysis['best_params']['covariances']
        )):
            gmm_pdf += weight * stats.norm.pdf(x_range, mean, np.sqrt(cov))
        plt.plot(x_range, gmm_pdf, 'r-', label='GMM拟合')
    elif analysis and analysis['best_distribution']:
        # 绘制最佳拟合分布
        dist = getattr(stats, analysis['best_distribution'])
        pdf = dist.pdf(x_range, *analysis['best_params'])
        plt.plot(x_range, pdf, 'r-', label=f"{analysis['best_distribution']}拟合")
    
    plt.title(f"残差分布 (均值={np.mean(residuals):.4f}, 标准差={np.std(residuals):.4f})", 
              fontproperties=chinese_font)
    plt.xlabel("残差", fontproperties=chinese_font)
    plt.ylabel("密度", fontproperties=chinese_font)
    plt.grid(True, alpha=0.3)
    plt.legend(prop=chinese_font)
    
    # 子图4：Q-Q图
    plt.subplot(2, 2, 4)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("正态Q-Q图", fontproperties=chinese_font)
    plt.grid(True, alpha=0.3)
    
    # 添加总标题
    plt.suptitle(title, fontsize=16, fontproperties=chinese_font)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_plots():
    """创建汇总分析图"""
    # 绘制所有地类的残差分布比较
    plt.figure(figsize=(15, 10))
    
    for i, land_type in enumerate(all_residuals.keys()):
        if len(all_residuals[land_type]) > 0:
            plt.subplot(2, 2, i+1)
            sns.histplot(np.concatenate(all_residuals[land_type]), kde=True, stat="density", bins=20)
            plt.title(f"{land_type}残差分布 (n={len(np.concatenate(all_residuals[land_type]))})", 
                      fontproperties=chinese_font)
            plt.xlabel("残差", fontproperties=chinese_font)
            plt.ylabel("密度", fontproperties=chinese_font)
            plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    for land_type in all_residuals.keys():
        if len(all_residuals[land_type]) > 0:
            sns.kdeplot(np.concatenate(all_residuals[land_type]), label=land_type)
    plt.title("不同地类残差分布比较", fontproperties=chinese_font)
    plt.xlabel("残差", fontproperties=chinese_font)
    plt.ylabel("密度", fontproperties=chinese_font)
    plt.legend(prop=chinese_font)
    plt.grid(True, alpha=0.3)
    
    plt.suptitle("不同地类残差分布分析", fontsize=16, fontproperties=chinese_font)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "地类残差分布比较.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 分析不同时间窗口的残差分布
    for time_window in time_window_residuals.keys():
        plt.figure(figsize=(15, 10))
        
        for i, land_type in enumerate(time_window_residuals[time_window].keys()):
            if len(time_window_residuals[time_window][land_type]) > 0:
                plt.subplot(2, 2, i+1)
                sns.histplot(np.concatenate(time_window_residuals[time_window][land_type]), 
                            kde=True, stat="density", bins=20)
                plt.title(f"{time_window} {land_type}残差分布", fontproperties=chinese_font)
                plt.xlabel("残差", fontproperties=chinese_font)
                plt.ylabel("密度", fontproperties=chinese_font)
                plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        for land_type in time_window_residuals[time_window].keys():
            if len(time_window_residuals[time_window][land_type]) > 0:
                sns.kdeplot(np.concatenate(time_window_residuals[time_window][land_type]), label=land_type)
        plt.title(f"{time_window}残差分布比较", fontproperties=chinese_font)
        plt.xlabel("残差", fontproperties=chinese_font)
        plt.ylabel("密度", fontproperties=chinese_font)
        plt.legend(prop=chinese_font)
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f"{time_window}时间窗口残差分布分析", fontsize=16, fontproperties=chinese_font)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f"{time_window}残差分布比较.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 分析不同轨迹的残差分布
    for trajectory in trajectory_residuals.keys():
        plt.figure(figsize=(15, 10))
        
        for i, land_type in enumerate(trajectory_residuals[trajectory].keys()):
            if len(trajectory_residuals[trajectory][land_type]) > 0:
                plt.subplot(2, 2, i+1)
                sns.histplot(np.concatenate(trajectory_residuals[trajectory][land_type]), 
                            kde=True, stat="density", bins=20)
                plt.title(f"{trajectory} {land_type}残差分布", fontproperties=chinese_font)
                plt.xlabel("残差", fontproperties=chinese_font)
                plt.ylabel("密度", fontproperties=chinese_font)
                plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        for land_type in trajectory_residuals[trajectory].keys():
            if len(trajectory_residuals[trajectory][land_type]) > 0:
                sns.kdeplot(np.concatenate(trajectory_residuals[trajectory][land_type]), label=land_type)
        plt.title(f"{trajectory}残差分布比较", fontproperties=chinese_font)
        plt.xlabel("残差", fontproperties=chinese_font)
        plt.ylabel("密度", fontproperties=chinese_font)
        plt.legend(prop=chinese_font)
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f"{trajectory}残差分布分析", fontsize=16, fontproperties=chinese_font)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f"{trajectory}残差分布比较.png"), dpi=300, bbox_inches='tight')
        plt.close()

def create_residual_model_report():
    """创建残差模型报告"""
    report = ["# 残差分析与建模报告", ""]
    
    # 添加总体分析
    report.append("## 总体残差分析")
    report.append("")
    
    for land_type in all_residuals.keys():
        if len(all_residuals[land_type]) > 0:
            all_res = np.concatenate(all_residuals[land_type])
            analysis = analyze_residual_distribution(all_res)
            
            if analysis:
                report.append(f"### {land_type}残差分析")
                report.append("")
                report.append(f"- 样本数量: {len(all_res)}")
                report.append(f"- 均值: {analysis['mean']:.4f}")
                report.append(f"- 标准差: {analysis['std']:.4f}")
                report.append(f"- 偏度: {analysis['skewness']:.4f}")
                report.append(f"- 峰度: {analysis['kurtosis']:.4f}")
                report.append(f"- Shapiro-Wilk正态性检验: 统计量={analysis['shapiro_test'][0]:.4f}, p值={analysis['shapiro_test'][1]:.4f}")
                report.append(f"- 最佳拟合分布: {analysis['best_distribution']}")
                
                if analysis['best_distribution'] == 'gmm':
                    means = analysis['best_params']['means']
                    weights = analysis['best_params']['weights']
                    covs = analysis['best_params']['covariances']
                    report.append("- 高斯混合模型参数:")
                    for i, (mean, weight, cov) in enumerate(zip(means, weights, covs)):
                        report.append(f"  - 成分{i+1}: 均值={mean:.4f}, 权重={weight:.4f}, 方差={cov:.4f}")
                elif analysis['best_distribution']:
                    report.append(f"- 分布参数: {analysis['best_params']}")
                
                report.append("")
    
    # 添加残差建模建议
    report.append("## 残差建模建议")
    report.append("")
    report.append("基于以上分析，我们提出以下残差建模方法：")
    report.append("")
    
    # 1. 正态分布模型
    report.append("### 1. 正态分布模型")
    report.append("")
    report.append("对于近似正态分布的残差，可以使用正态分布模型：")
    report.append("")
    report.append("```python")
    report.append("def generate_residual(land_type):")
    report.append("    mean = residual_params[land_type]['mean']")
    report.append("    std = residual_params[land_type]['std']")
    report.append("    return np.random.normal(mean, std)")
    report.append("```")
    report.append("")
    
    # 2. 高斯混合模型
    report.append("### 2. 高斯混合模型")
    report.append("")
    report.append("对于分布更复杂的残差，可以使用高斯混合模型：")
    report.append("")
    report.append("```python")
    report.append("def generate_residual_gmm(land_type):")
    report.append("    # 选择一个高斯成分")
    report.append("    component = np.random.choice(len(gmm_params[land_type]['weights']), p=gmm_params[land_type]['weights'])")
    report.append("    # 从选定的成分生成")
    report.append("    mean = gmm_params[land_type]['means'][component]")
    report.append("    std = np.sqrt(gmm_params[land_type]['covariances'][component])")
    report.append("    return np.random.normal(mean, std)")
    report.append("```")
    report.append("")
    
    # 3. 残差抽样法
    report.append("### 3. 基于历史数据的残差抽样法")
    report.append("")
    report.append("对于难以用参数分布拟合的残差，可以直接从历史残差中随机抽样：")
    report.append("")
    report.append("```python")
    report.append("def generate_residual_sampling(land_type):")
    report.append("    # 从历史残差中随机抽样")
    report.append("    historical_residuals = all_residuals[land_type]")
    report.append("    return np.random.choice(historical_residuals)")
    report.append("```")
    report.append("")
    
    # 4. 非参数核密度估计
    report.append("### 4. 非参数核密度估计")
    report.append("")
    report.append("使用核密度估计来模拟复杂的残差分布：")
    report.append("")
    report.append("```python")
    report.append("def generate_residual_kde(land_type):")
    report.append("    # 使用KDE模型生成随机样本")
    report.append("    kde = kde_models[land_type]")
    report.append("    sample = kde.sample(1)[0][0]")
    report.append("    return sample")
    report.append("```")
    report.append("")
    
    # 5. 整合模型
    report.append("### 5. 整合到速度生成模型")
    report.append("")
    report.append("将残差模型整合到速度生成中：")
    report.append("")
    report.append("```python")
    report.append("def calculate_speed(slope, land_type):")
    report.append("    # 计算基础速度（线性模型）")
    report.append("    base_speed = models[land_type]['slope'] * slope + models[land_type]['intercept']")
    report.append("    ")
    report.append("    # 生成随机残差")
    report.append("    residual = generate_residual(land_type)")
    report.append("    ")
    report.append("    # 添加残差")
    report.append("    speed = base_speed + residual")
    report.append("    ")
    report.append("    # 确保速度在合理范围内")
    report.append("    speed = max(min_speed, min(speed, max_speed))")
    report.append("    ")
    report.append("    return speed")
    report.append("```")
    
    # 写入文件
    with open(os.path.join(output_dir, "残差分析与建模报告.md"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

def main():
    """主函数"""
    print("开始分析残差数据...")
    
    # 查找所有分箱统计CSV文件
    csv_files = glob.glob(os.path.join(analysis_dir, '*坡度分箱统计*.csv'))
    
    # 处理每个文件
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"处理文件: {file_name}")
        
        # 提取信息，更健壮的解析方式
        try:
            parts = file_name.split('_')
            # 确保文件名格式正确
            if len(parts) >= 4:
                # 轨迹1_坡度分箱统计_1秒_林地.csv 格式
                trajectory = parts[0]
                time_window = parts[2]
                land_type = parts[3].split('.')[0]
            elif '林地' in file_name:
                land_type = '林地'
                time_window = [p for p in parts if '秒' in p][0] if any('秒' in p for p in parts) else '1秒'
                trajectory = parts[0] if parts[0].startswith('轨迹') else '未知轨迹'
            elif '灌木地' in file_name:
                land_type = '灌木地'
                time_window = [p for p in parts if '秒' in p][0] if any('秒' in p for p in parts) else '1秒'
                trajectory = parts[0] if parts[0].startswith('轨迹') else '未知轨迹'
            elif '水体' in file_name:
                land_type = '水体'
                time_window = [p for p in parts if '秒' in p][0] if any('秒' in p for p in parts) else '1秒'
                trajectory = parts[0] if parts[0].startswith('轨迹') else '未知轨迹'
            else:
                print(f"无法解析文件名: {file_name}，跳过")
                continue
                
            print(f"  解析结果: 轨迹={trajectory}, 时间窗口={time_window}, 地类={land_type}")
            
            # 提取残差
            result = extract_residuals_from_csv(file_path)
            if result:
                # 添加到总体残差字典
                all_residuals[land_type].append(result['residuals'])
                
                # 添加到时间窗口残差字典
                if time_window in time_window_residuals:
                    time_window_residuals[time_window][land_type].append(result['residuals'])
                
                # 添加到轨迹残差字典
                if trajectory in trajectory_residuals:
                    trajectory_residuals[trajectory][land_type].append(result['residuals'])
                
                # 绘制残差分析图
                plot_title = f"{trajectory} {time_window} {land_type}残差分析"
                save_path = os.path.join(output_dir, f"{trajectory}_{time_window}_{land_type}_残差分析.png")
                plot_residual_analysis(result, plot_title, save_path)
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")
            continue
    
    # 创建汇总分析图
    print("创建汇总分析图...")
    create_summary_plots()
    
    # 创建残差模型报告
    print("创建残差模型报告...")
    create_residual_model_report()
    
    print("残差分析完成！结果保存在 residual_analysis 目录中")

if __name__ == "__main__":
    main() 