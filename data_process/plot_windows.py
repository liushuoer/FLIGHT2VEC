import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import numpy as np
from scipy.spatial.distance import cdist

import pandas as pd
import numpy as np


def preprocess_altitude(flight_data, threshold=200, window_size=50):
    altitude_data = flight_data['altitude'].copy()
    n = len(altitude_data)
    print(n)

    i = 0
    while i < n - window_size + 1:
        window = altitude_data.iloc[i:i + window_size]
        min_height = window.min()
        max_height = window.max()

        if max_height - min_height < threshold:
            # 将窗口内的高度统一为最大值和最小值的平均数
            average_height = (max_height + min_height) / 2
            altitude_data.iloc[i:i + window_size] = average_height

        i += window_size

    flight_data['altitude'] = altitude_data

    return flight_data

def calculate_angles(flight_data, window_len=5, scale_factor=1000):
    angles = []
    coordinates = flight_data[['longitude', 'latitude', 'altitude']].to_numpy()
    num_points = len(coordinates)

    lon_range = coordinates[:, 0].max() - coordinates[:, 0].min()
    lat_range = coordinates[:, 1].max() - coordinates[:, 1].min()
    alt_range = coordinates[:, 2].max() - coordinates[:, 2].min()

    lon_scale = (1.0 / lon_range * scale_factor) if lon_range != 0 else 1.0
    lat_scale = (1.0 / lat_range * scale_factor) if lat_range != 0 else 1.0
    alt_scale = (1.0 / alt_range * scale_factor) if alt_range != 0 else 1.0
    print(coordinates[:, 2])

    for i in range(window_len, num_points - window_len):
        prev_point = coordinates[i - window_len]
        curr_point = coordinates[i]
        next_point = coordinates[i + window_len]

        prev_point_scaled = np.array([prev_point[0] * lon_scale, prev_point[1] * lat_scale, prev_point[2] * alt_scale])
        curr_point_scaled = np.array([curr_point[0] * lon_scale, curr_point[1] * lat_scale, curr_point[2] * alt_scale])
        next_point_scaled = np.array([next_point[0] * lon_scale, next_point[1] * lat_scale, next_point[2] * alt_scale])

        vector1 = curr_point_scaled - prev_point_scaled
        vector2 = next_point_scaled - curr_point_scaled
        # print(prev_point[2])

        if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
            angles.append(0)  # 如果向量无效，将角度设置为 0
            continue

        dot_product = np.dot(vector1, vector2)
        norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if norm_product == 0:
            angle = 0
        else:
            cos_theta = np.clip(dot_product / norm_product, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_theta))

        angles.append(angle)

    return angles



def smooth_flight_data(flight_data, angles, window_len=10, angle_threshold=20):
    flight_data['angle'] = angles

    large_angles_indices = flight_data[flight_data['angle'] > angle_threshold].index

    smoothed_altitude = flight_data['altitude'].copy()

    for i in range(len(smoothed_altitude)):
        if i < window_len // 2 or i > len(smoothed_altitude) - window_len // 2:
            continue
        window = smoothed_altitude[i - window_len // 2:i + window_len // 2 + 1]
        max_val = window.max()
        min_val = window.min()
        if max_val - min_val < 0.3:
            smoothed_altitude[i - window_len // 2:i + window_len // 2 + 1] = max_val

    flight_data['altitude'] = smoothed_altitude
    flight_data.loc[large_angles_indices, 'altitude'] = flight_data.loc[large_angles_indices, 'altitude']

    return flight_data


def visualize_flight_with_angles(csv_file, callsign, window_len=10, distance_threshold=80, angle_threshold=20):
    df = pd.read_csv(csv_file)

    flight_data = df[df['Callsign'] == callsign]

    if flight_data.empty:
        print(f"No data found for Callsign {callsign}")
        return

    flight_data.interpolate(method='linear', inplace=True)
    flight_data.fillna(method='bfill', inplace=True)
    flight_data.fillna(method='ffill', inplace=True)
    # preprocessed_data = preprocess_altitude(flight_data.copy())
    preprocessed_data = flight_data.copy()

    angles = calculate_angles(preprocessed_data, window_len)

    preprocessed_data = preprocessed_data.iloc[window_len:-window_len].copy()
    preprocessed_data['angle'] = angles

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(flight_data['longitude'], flight_data['latitude'], flight_data['altitude'], color='green', linestyle='-.',
             label='Original Trajectory')
    # ax1.plot(flight_data['longitude'][200:300], flight_data['latitude'][200:300], flight_data['altitude'][200:300], color='green', linestyle='-.', label='Original Trajectory')
    # ax1.scatter(flight_data['longitude'][::15], flight_data['latitude'][::15], flight_data['altitude'][::15], color='green', marker='o', s=5)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_zlabel('Altitude (feet)')
    ax1.set_title(f'Original Trajectory of Callsign: {callsign}')
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    start = 300
    end = 350
    # ax2.plot(preprocessed_data['longitude'][start:end], preprocessed_data['latitude'][start:end], preprocessed_data['altitude'][start:end], color='blue', linestyle='-.', label='Preprocessed Trajectory')

    for i in range(0, len(preprocessed_data), 5):
        # for i in range(start, end, 1):
        point = preprocessed_data.iloc[i]
        ax2.text(point['longitude'], point['latitude'], point['altitude'], f"{point['angle']:.1f}°", fontsize=8,
                 color='black')

    large_angles = preprocessed_data[preprocessed_data['angle'] > angle_threshold]
    window_len = 10
    preprocessed_data = smooth_flight_data(preprocessed_data, preprocessed_data['angle'], window_len, angle_threshold)

    # ax2.plot(preprocessed_data['longitude'][200:300], preprocessed_data['latitude'][200:300], preprocessed_data['altitude'][200:300], color='blue', linestyle='-.', label='Preprocessed Trajectory')
    ax2.plot(preprocessed_data['longitude'], preprocessed_data['latitude'], preprocessed_data['altitude'], color='blue',
             linestyle='-.', label='Preprocessed Trajectory')

    ax2.scatter(large_angles['longitude'], large_angles['latitude'], large_angles['altitude'], color='red', marker='s',
                s=5, label=f'Angles > {angle_threshold}°')

    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_zlabel('Altitude (feet)')
    ax2.set_title(f'Preprocessed Trajectory with Angles of Callsign: {callsign}')
    ax2.legend()

    # 显示图像
    plt.tight_layout()
    plt.show()


# df = pd.read_parquet('squawk7700_trajectories.parquet.gz', engine='pyarrow')

visualize_flight_with_angles(r"E:\pathformer-main\dataset\aircraft.csv", callsign=100007)
# 102603