import numpy as np
import pandas as pd
import rasterio
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, f_oneway
import seaborn as sns

# 设置matplotlib支持中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 设置全局字体大小
plt.rcParams['font.size'] = 12

# 土地覆盖类型映射
LANDCOVER_NAMES = {
    10: '耕地',
    20: '林地',
    30: '草地',
    40: '灌木地',
    50: '湿地',
    60: '水体',
    70: '苔原',
    80: '人工表面',
    90: '裸地',
    100: '永久积雪'
}

def load_gis_data():
    """加载所有GIS数据"""
    gis_data = {}
    files = {
        'dem': 'dem_aligned.tif',
        'slope': 'slope_aligned.tif',
        'aspect': 'aspect_aligned.tif',
        'landcover': 'landcover_aligned.tif'
    }
    
    for name, file in files.items():
        with rasterio.open(file) as src:
            gis_data[name] = {
                'data': src.read(1),
                'transform': src.transform,
                'nodata': src.nodata
            }
    return gis_data

def calculate_angle_difference(angle1, angle2):
    """计算两个角度之间的夹角（0-180度）"""
    diff = abs(angle1 - angle2) % 360
    return min(diff, 360 - diff)

def calculate_effective_slope(slope, aspect, heading):
    """计算有效坡度（考虑移动方向）
    
    Args:
        slope: 坡度（度）
        aspect: 坡向（度，0-360，0表示正北，顺时针增加）
        heading: 移动方向（度，0-360，0表示正北，顺时针增加）
    
    Returns:
        effective_slope: 有效坡度（度，正值表示上坡，负值表示下坡）
    """
    # 确保输入角度在0-360度范围内
    aspect = aspect % 360
    heading = heading % 360
    
    # 计算坡向与移动方向的夹角
    angle_diff = calculate_angle_difference(aspect, heading)
    
    # 判断是上坡还是下坡
    # 当移动方向与坡向夹角小于90度时为上坡
    # 当夹角大于90度时为下坡
    if angle_diff > 90:
        direction = -1  # 下坡
    else:
        direction = 1   # 上坡
        
    # 计算有效坡度：坡度 * cos(夹角) * 方向
    effective_slope = direction * slope * abs(np.cos(np.radians(angle_diff)))
    
    return effective_slope

def get_environment_features(gis_data, x, y):
    """获取给定坐标点的环境特征"""
    features = {}
    for name, data in gis_data.items():
        row, col = ~data['transform'] * (x, y)
        row, col = int(row), int(col)
        if 0 <= row < data['data'].shape[0] and 0 <= col < data['data'].shape[1]:
            value = data['data'][row, col]
            if value != data['nodata']:
                features[name] = value
            else:
                features[name] = np.nan
        else:
            features[name] = np.nan
    return features

def aggregate_trajectory_data(df, window_size):
    """对轨迹数据进行时间窗口聚合
    
    Args:
        df: 轨迹数据DataFrame
        window_size: 时间窗口大小（秒）
    """
    # 将毫秒时间戳转换为datetime
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    
    # 创建时间窗口
    df['time_window'] = df['timestamp'].dt.floor(f'{window_size}S')
    
    # 聚合数据
    agg_df = df.groupby('time_window').agg({
        'velocity_2d_ms': 'mean',
        'longitude': 'mean',
        'latitude': 'mean',
        'heading_deg': lambda x: np.mean(x) % 360,  # 处理角度的平均值
        'effective_slope': 'mean',
        'landcover': lambda x: x.mode().iloc[0] if not x.empty else np.nan,
        'trajectory_id': 'first'  # 保留轨迹ID
    }).reset_index()
    
    return agg_df

def analyze_trajectory(trajectory_file, gis_data, trajectory_id):
    """分析单个轨迹文件的速度与环境特征关系"""
    # 读取轨迹数据
    df = pd.read_csv(trajectory_file)
    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df['trajectory_id'] = trajectory_id  # 添加轨迹ID
    
    # 提取环境特征
    features = []
    for _, row in df.iterrows():
        env = get_environment_features(gis_data, row['longitude'], row['latitude'])
        
        # 计算有效坡度
        if not np.isnan(env.get('slope', np.nan)) and not np.isnan(env.get('aspect', np.nan)):
            # 打印调试信息
            if len(features) == 0:  # 只打印第一个点的信息
                print(f"\n调试信息（轨迹{trajectory_id}第一个点）:")
                print(f"坡度: {env['slope']:.2f}度")
                print(f"坡向: {env['aspect']:.2f}度")
                print(f"移动方向: {row['heading_deg']:.2f}度")
            
            env['effective_slope'] = calculate_effective_slope(
                env['slope'],
                env['aspect'],
                row['heading_deg']
            )
            
            # 打印第一个点的有效坡度
            if len(features) == 0:
                print(f"计算的有效坡度: {env['effective_slope']:.2f}度")
        else:
            env['effective_slope'] = np.nan
            
        features.append(env)
    
    # 将特征转换为DataFrame并添加时间戳和其他信息
    feature_df = pd.DataFrame(features)
    feature_df['timestamp'] = df['timestamp']
    feature_df['velocity_2d_ms'] = df['velocity_2d_ms']
    feature_df['longitude'] = df['longitude']
    feature_df['latitude'] = df['latitude']
    feature_df['heading_deg'] = df['heading_deg']
    feature_df['trajectory_id'] = df['trajectory_id']
    
    return feature_df

def analyze_speed_vs_slope(feature_df, window_size, output_prefix, landcover_type=None):
    """分析速度与有效坡度的关系（指定时间窗口和土地覆盖类型）"""
    agg_df = aggregate_trajectory_data(feature_df, window_size)
    
    # 如果指定了土地覆盖类型，只分析该类型的数据
    if landcover_type is not None:
        agg_df = agg_df[agg_df['landcover'] == landcover_type]
        type_name = LANDCOVER_NAMES.get(landcover_type, str(landcover_type))
    
    # 打印有效坡度的基本统计信息
    print(f"\n有效坡度统计信息（{window_size}秒窗口）:")
    print(f"最小值: {agg_df['effective_slope'].min():.2f}")
    print(f"最大值: {agg_df['effective_slope'].max():.2f}")
    print(f"平均值: {agg_df['effective_slope'].mean():.2f}")
    print(f"标准差: {agg_df['effective_slope'].std():.2f}")
    print(f"上坡样本数: {(agg_df['effective_slope'] > 0).sum()}")
    print(f"下坡样本数: {(agg_df['effective_slope'] < 0).sum()}")
    
    # 按轨迹ID分组分析
    results = []
    for traj_id in agg_df['trajectory_id'].unique():
        traj_df = agg_df[agg_df['trajectory_id'] == traj_id]
        
        valid_mask = ~(np.isnan(traj_df['effective_slope']) | np.isnan(traj_df['velocity_2d_ms']))
        if valid_mask.sum() < 2:  # 至少需要2个点才能计算相关性
            continue
            
        X = traj_df['effective_slope'][valid_mask].values.reshape(-1, 1)
        y = traj_df['velocity_2d_ms'][valid_mask].values
        
        # 分别计算上坡和下坡的相关性
        up_mask = X.ravel() >= 0
        down_mask = X.ravel() < 0
        
        # 只在样本数足够时计算相关系数
        up_corr = pearsonr(X[up_mask].ravel(), y[up_mask])[0] if up_mask.sum() >= 2 else np.nan
        down_corr = pearsonr(X[down_mask].ravel(), y[down_mask])[0] if down_mask.sum() >= 2 else np.nan
        
        # 分别计算上坡和下坡的R²
        if up_mask.sum() >= 2:
            up_model = LinearRegression()
            up_model.fit(X[up_mask], y[up_mask])
            up_r2 = up_model.score(X[up_mask], y[up_mask])
        else:
            up_r2 = np.nan
            
        if down_mask.sum() >= 2:
            down_model = LinearRegression()
            down_model.fit(X[down_mask], y[down_mask])
            down_r2 = down_model.score(X[down_mask], y[down_mask])
        else:
            down_r2 = np.nan
        
        # 总体相关性和回归
        correlation, p_value = pearsonr(X.ravel(), y)
        model = LinearRegression()
        model.fit(X, y)
        r2_score = model.score(X, y)
        
        results.append({
            'trajectory_id': traj_id,
            'correlation': correlation,
            'up_correlation': up_corr,
            'down_correlation': down_corr,
            'up_r2': up_r2,
            'down_r2': down_r2,
            'p_value': p_value,
            'r2_score': r2_score,
            'coefficient': model.coef_[0],
            'intercept': model.intercept_,
            'sample_count': len(y),
            'up_sample_count': up_mask.sum(),
            'down_sample_count': down_mask.sum()
        })
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # 构建基础标题
    base_title = f'时间窗口：{window_size}秒'
    if landcover_type is not None:
        base_title += f' | 地类：{type_name}'
    
    # 所有轨迹的散点图
    plt.subplot(221)
    for traj_id in agg_df['trajectory_id'].unique():
        traj_df = agg_df[agg_df['trajectory_id'] == traj_id]
        valid_mask = ~(np.isnan(traj_df['effective_slope']) | np.isnan(traj_df['velocity_2d_ms']))
        
        # 分别绘制上坡和下坡的点
        up_mask = traj_df['effective_slope'] >= 0
        down_mask = traj_df['effective_slope'] < 0
        
        plt.scatter(traj_df['effective_slope'][valid_mask & up_mask], 
                   traj_df['velocity_2d_ms'][valid_mask & up_mask], 
                   alpha=0.3, label=f'轨迹{traj_id}上坡')
        plt.scatter(traj_df['effective_slope'][valid_mask & down_mask], 
                   traj_df['velocity_2d_ms'][valid_mask & down_mask], 
                   alpha=0.3, marker='s', label=f'轨迹{traj_id}下坡')
    
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('有效坡度（度，负值表示下坡）')
    plt.ylabel('速度（米/秒）')
    plt.title(f'速度-坡度散点图\n{base_title}')
    plt.legend()
    
    # 核密度估计
    plt.subplot(222)
    valid_mask = ~(np.isnan(agg_df['effective_slope']) | np.isnan(agg_df['velocity_2d_ms']))
    sns.kdeplot(data=agg_df[valid_mask], x='effective_slope', y='velocity_2d_ms', 
                cmap='viridis', fill=True)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('有效坡度（度，负值表示下坡）')
    plt.ylabel('速度（米/秒）')
    plt.title(f'速度-坡度联合密度分布\n{base_title}')
    
    # 每条轨迹的回归线
    plt.subplot(223)
    x_range = np.linspace(agg_df['effective_slope'].min(), 
                         agg_df['effective_slope'].max(), 100)
    for result in results:
        traj_id = result['trajectory_id']
        y_pred = result['coefficient'] * x_range + result['intercept']
        label = (f'轨迹{traj_id}\n'
                f'总体R$^2$={result["r2_score"]:.3f}\n'
                f'上坡：R$^2$={result["up_r2"]:.3f}, r={result["up_correlation"]:.3f} (n={result["up_sample_count"]})\n'
                f'下坡：R$^2$={result["down_r2"]:.3f}, r={result["down_correlation"]:.3f} (n={result["down_sample_count"]})')
        plt.plot(x_range, y_pred, label=label)
    
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('有效坡度（度，负值表示下坡）')
    plt.ylabel('速度（米/秒）')
    plt.title(f'速度-坡度回归分析\n{base_title}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 箱线图
    plt.subplot(224)
    x_min, x_max = agg_df['effective_slope'].min(), agg_df['effective_slope'].max()
    bins = np.linspace(x_min, x_max, 11)
    agg_df.loc[valid_mask, 'slope_bin'] = pd.cut(
        agg_df.loc[valid_mask, 'effective_slope'], 
        bins=bins, labels=False)
    sns.boxplot(data=agg_df[valid_mask], x='slope_bin', y='velocity_2d_ms')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_labels = [f'{val:.1f}' for val in bin_centers]
    plt.xticks(range(10), bin_labels, rotation=45)
    plt.xlabel('有效坡度区间中点（度，负值表示下坡）')
    plt.ylabel('速度（米/秒）')
    plt.title(f'速度分布随坡度变化\n{base_title}')
    
    plt.tight_layout()
    
    # 构建中文输出文件名
    output_filename = f'速度坡度分析_{window_size}秒'
    if landcover_type is not None:
        output_filename += f'_{type_name}'
    plt.savefig(f'{output_filename}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return results

def analyze_landcover_effect(feature_df, window_size, output_prefix):
    """分析土地覆盖类型对速度的影响（指定时间窗口）"""
    agg_df = aggregate_trajectory_data(feature_df, window_size)
    
    # 添加中文名称
    agg_df['landcover_name'] = agg_df['landcover'].map(LANDCOVER_NAMES)
    
    # 按轨迹ID和土地覆盖类型分组计算统计量
    grouped_stats = agg_df.groupby(['trajectory_id', 'landcover'])['velocity_2d_ms'].agg([
        'count', 'mean', 'std', 'median'
    ]).round(3)
    
    # 进行单因素方差分析
    categories = sorted(agg_df['landcover'].unique())
    category_speeds = [agg_df[agg_df['landcover'] == cat]['velocity_2d_ms'] 
                      for cat in categories]
    f_stat, p_value = f_oneway(*category_speeds)
    
    # 计算η²值
    ss_between = sum(len(group) * (group.mean() - agg_df['velocity_2d_ms'].mean())**2 
                    for group in category_speeds)
    ss_total = sum((x - agg_df['velocity_2d_ms'].mean())**2 
                   for x in agg_df['velocity_2d_ms'])
    eta_squared = ss_between / ss_total if ss_total != 0 else 0
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    base_title = f'时间窗口：{window_size}秒'
    
    # 总体箱线图
    plt.subplot(221)
    sns.boxplot(data=agg_df, x='landcover_name', y='velocity_2d_ms')
    plt.title(f'不同地类的速度分布\n{base_title} | 效应量η$^2$={eta_squared:.3f}')
    plt.xlabel('土地覆盖类型')
    plt.ylabel('速度（米/秒）')
    plt.xticks(rotation=45)
    
    # 按轨迹分组的箱线图
    plt.subplot(222)
    sns.boxplot(data=agg_df, x='landcover_name', y='velocity_2d_ms', 
                hue='trajectory_id')
    plt.title(f'各轨迹在不同地类的速度分布\n{base_title}')
    plt.xlabel('土地覆盖类型')
    plt.ylabel('速度（米/秒）')
    plt.xticks(rotation=45)
    
    # 速度随时间的变化
    plt.subplot(212)
    for traj_id in agg_df['trajectory_id'].unique():
        traj_df = agg_df[agg_df['trajectory_id'] == traj_id]
        plt.plot(traj_df['time_window'], traj_df['velocity_2d_ms'], 
                label=f'轨迹{traj_id}', alpha=0.7)
    plt.title(f'速度时间序列\n{base_title}')
    plt.xlabel('时间')
    plt.ylabel('速度（米/秒）')
    plt.legend()
    
    plt.tight_layout()
    
    # 构建中文输出文件名
    output_filename = f'地类速度分析_{window_size}秒'
    plt.savefig(f'{output_filename}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return {
        'window_size': window_size,
        'group_stats': grouped_stats,
        'eta_squared': eta_squared,
        'f_statistic': f_stat,
        'p_value': p_value
    }

def main():
    # 加载GIS数据
    print("加载GIS数据...")
    gis_data = load_gis_data()
    
    # 分析所有轨迹
    print("分析轨迹数据...")
    all_features = []
    for i in range(1, 5):
        trajectory_file = f'core_trajectories/converted_sequence_{i}_core.csv'
        features = analyze_trajectory(trajectory_file, gis_data, i)
        all_features.append(features)
    
    # 合并所有特征
    combined_features = pd.concat(all_features, ignore_index=True)
    
    # 获取主要的土地覆盖类型（样本数大于100的类型）
    landcover_counts = combined_features['landcover'].value_counts()
    major_landcover_types = landcover_counts[landcover_counts > 100].index
    
    # 不同时间窗口的分析
    window_sizes = [1, 2, 5, 10, 30, 60]  # 秒
    
    print("进行不同时间窗口的分析...")
    
    # 创建报告
    report = "二维速度与环境特征关系分析报告\n" + "="*50 + "\n\n"
    
    # 对每个主要土地覆盖类型进行分析
    for landcover_type in major_landcover_types:
        report += f"\n土地覆盖类型：{LANDCOVER_NAMES.get(landcover_type, str(landcover_type))}\n"
        report += "-" * 40 + "\n"
        report += "窗口大小(s) | 轨迹ID | 样本数 | 相关系数 | R² | 斜率 | 截距 | P值\n"
        report += "-" * 100 + "\n"
        
        for window_size in window_sizes:
            print(f"分析 {window_size}s 时间窗口，土地覆盖类型 {landcover_type}...")
            results = analyze_speed_vs_slope(
                combined_features, window_size, 'speed_vs_slope',
                landcover_type=landcover_type)
            
            for result in results:
                report += f"{window_size:11d} | {result['trajectory_id']:8d} | "
                report += f"{result['sample_count']:7d} | {result['correlation']:9.3f} | "
                report += f"{result['r2_score']:4.3f} | {result['coefficient']:6.3f} | "
                report += f"{result['intercept']:6.3f} | {result['p_value']:.2e}\n"
    
    # 添加土地覆盖类型的总体分析
    report += "\n土地覆盖类型对速度的影响（不同时间窗口）:\n" + "-"*40 + "\n"
    report += "窗口大小(s) | η² | F统计量 | P值\n"
    report += "-" * 50 + "\n"
    
    for window_size in window_sizes:
        print(f"分析土地覆盖类型影响 {window_size}s 时间窗口...")
        result = analyze_landcover_effect(
            combined_features, window_size, 'speed_vs_landcover')
        
        report += f"{result['window_size']:11d} | {result['eta_squared']:4.3f} | "
        report += f"{result['f_statistic']:9.3f} | {result['p_value']:.2e}\n"
        report += "\n分组统计（按轨迹ID和土地覆盖类型）:\n"
        report += str(result['group_stats']) + "\n\n"
    
    # 保存报告
    with open('speed_analysis_summary.txt', 'w') as f:
        f.write(report)
    
    print("分析完成，结果已保存。")

if __name__ == "__main__":
    main() 