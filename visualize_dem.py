import rasterio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_raster_with_trajectory(ax, raster_file, trajectory_file, title, cmap='terrain'):
    """在一个子图中绘制栅格数据和轨迹"""
    # 读取栅格数据
    with rasterio.open(raster_file) as src:
        data = src.read(1)
        bounds = src.bounds
        
        # 处理无效值
        nodata = src.nodata if src.nodata is not None else -9999
        if data.dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
            # 对于整数类型数据，使用掩码数组
            mask = data == nodata
            masked_data = np.ma.array(data, mask=mask)
        else:
            # 对于浮点数据，使用nan
            data = data.astype(float)
            data[data == nodata] = np.nan
            masked_data = data
        
        # 计算有效值的范围
        if isinstance(masked_data, np.ma.MaskedArray):
            valid_data = masked_data.compressed()  # 获取非掩码数据
        else:
            valid_data = masked_data[~np.isnan(masked_data)]
            
        if len(valid_data) > 0:
            if 'landcover' in raster_file:
                # 对于土地覆盖数据使用离散颜色
                unique_values = np.unique(valid_data)
                vmin, vmax = unique_values.min(), unique_values.max()
                im = ax.imshow(masked_data, cmap='tab20', vmin=vmin, vmax=vmax,
                             extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                             aspect='equal')
            else:
                # 对于其他数据使用百分位数
                vmin, vmax = np.percentile(valid_data, [2, 98])
                im = ax.imshow(masked_data, cmap=cmap, vmin=vmin, vmax=vmax,
                             extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                             aspect='equal')
            plt.colorbar(im, ax=ax)
            
            # 添加数据统计信息
            stats_text = f"数据统计:\n"
            stats_text += f"形状: {data.shape}\n"
            if 'landcover' in raster_file:
                stats_text += f"类别数: {len(unique_values)}\n"
                stats_text += f"类别值: {sorted(unique_values)}\n"
            else:
                stats_text += f"范围: {vmin:.1f} 到 {vmax:.1f}\n"
                stats_text += f"平均值: {np.mean(valid_data):.1f}\n"
            ax.text(1.02, 0.5, stats_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='center')
        else:
            ax.text(0.5, 0.5, '没有有效数据', ha='center', va='center', transform=ax.transAxes)
        
        # 读取并绘制轨迹
        df = pd.read_csv(trajectory_file)
        ax.plot(df['longitude'], df['latitude'], 'r-', linewidth=1.0, alpha=1.0, label='轨迹')
        
        ax.set_title(title)
        ax.set_xlabel('X坐标 (m)')
        ax.set_ylabel('Y坐标 (m)')
        ax.grid(True)
        ax.legend()

def main():
    # 创建2x2的子图布局
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. DEM数据
    ax1 = fig.add_subplot(gs[0, 0])
    plot_raster_with_trajectory(
        ax1,
        'dem_aligned.tif',
        'core_trajectories/converted_sequence_1_core.csv',
        'DEM数据 (高程)',
        cmap='terrain'
    )
    
    # 2. 坡度数据
    ax2 = fig.add_subplot(gs[0, 1])
    plot_raster_with_trajectory(
        ax2,
        'slope_aligned.tif',
        'core_trajectories/converted_sequence_1_core.csv',
        '坡度数据 (度)',
        cmap='YlOrRd'  # 黄-橙-红色渐变，适合表示坡度
    )
    
    # 3. 坡向数据
    ax3 = fig.add_subplot(gs[1, 0])
    plot_raster_with_trajectory(
        ax3,
        'aspect_aligned.tif',
        'core_trajectories/converted_sequence_1_core.csv',
        '坡向数据 (度)',
        cmap='hsv'  # 循环色图，适合表示角度
    )
    
    # 4. 土地覆盖数据
    ax4 = fig.add_subplot(gs[1, 1])
    plot_raster_with_trajectory(
        ax4,
        'landcover_aligned.tif',
        'core_trajectories/converted_sequence_1_core.csv',
        '土地覆盖数据 (类别)',
        cmap='tab20'
    )
    
    plt.savefig('gis_data_comparison.png', dpi=300, bbox_inches='tight')
    print("可视化结果已保存到 gis_data_comparison.png")

if __name__ == "__main__":
    main() 