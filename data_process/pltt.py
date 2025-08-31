import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_flight_trajectory(df, flight_id):
    """
    可视化指定 flight_id 的飞行轨迹。

    参数：
    df (DataFrame): 包含飞行数据的 DataFrame。
    flight_id (int or str): 要可视化的飞行的 ID。
    """
    # 筛选出指定 flight_id 的数据
    flight_data = df[df['flight_id'] == flight_id]

    if flight_data.empty:
        print(f"No data found for flight_id {flight_id}")
        return

    # 创建3D图
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制飞行轨迹
    ax.plot(flight_data['longitude'], flight_data['latitude'], flight_data['altitude'], label=f'Flight {flight_id}')

    # 设置标签和标题
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude (m)')
    ax.set_title(f'3D Trajectory of Flight ID: {flight_id}')

    # 显示图例
    ax.legend()

    # 显示图像
    plt.show()


# 示例用法
# 假设 df 是包含飞行数据的 DataFrame
# 你可以通过读取数据文件的方式获取这个 DataFrame
df = pd.read_parquet('squawk7700_trajectories.parquet.gz', engine='pyarrow')

# 可视化特定 flight_id 的轨迹
visualize_flight_trajectory(df, flight_id='AAL900_20180425')
