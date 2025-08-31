import os

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random


import numpy as np
from scipy.spatial.distance import cdist

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def preprocess_altitude(flight_data, threshold=200, window_size=50):
    altitude_data = flight_data['altitude'].copy()
    n = len(altitude_data)

    i = 0
    while i < n - window_size + 1:
        window = altitude_data.iloc[i:i + window_size]
        min_height = window.min()
        max_height = window.max()

        if max_height - min_height < threshold:
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
            angles.append(0)
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

def sample_trajectory(preprocessed_data, seq_len, activity_init_points, patch_len):
    N = len(preprocessed_data)
    print(len(preprocessed_data))
    if N <= seq_len:
        return preprocessed_data
    activity_init_points = activity_init_points[::2]

    # 对活动点进行分组（连续或邻近的点分为一组）
    groups = []
    if not activity_init_points:
        return preprocessed_data[:seq_len]  # 无活动点时返回前seq_len个点

    sorted_points = sorted(activity_init_points)
    current_group = [sorted_points[0]]
    for point in sorted_points[1:]:
        if point - current_group[-1] <= 20:  # 连续或邻近点视为同一组
            current_group.append(point)
        else:
            groups.append(current_group)
            current_group = [point]
    groups.append(current_group)  # 添加最后一组

    # 提取每组的中心点
    simplified_points = []
    for group in groups:
        mid_idx = len(group) // 2
        simplified_points.append(group[mid_idx])

    sampled_indices = set()
    half_patch = patch_len//2
    past_point = 0
    range_index = set()

    for point in simplified_points:
        start = max(0, point - half_patch)
        end = min(N, point + half_patch + 1)
        print(start, end)
        print(point)
        range_index.add(start)
        range_index.add(end)
        # print(activity_init_points)
        # print(half_patch)
        # print(point - half_patch)

        for i in range(start, end):
            sampled_indices.add(i)
            past_point = point

    remaining_indices = sorted(set(range(N)) - sampled_indices)
    remaining_len = seq_len - len(sampled_indices)

    if remaining_len < 3:
        if len(sampled_indices) > 2:
            sampled_indices = list(sampled_indices)
            random.shuffle(sampled_indices)
            sampled_indices = sampled_indices[:len(sampled_indices) - 2]

        sampled_indices.extend([0, N - 1])

        while len(sampled_indices) < seq_len:
            sampled_indices.append(random.choice(remaining_indices))
    else:
        if remaining_len // 2 > 0:
            step_forward = max(1, len(remaining_indices) // (remaining_len // 2))
            additional_indices_forward = remaining_indices[::step_forward]
        else:
            additional_indices_forward = []

        if remaining_len // 2 > 0:
            step_backward = max(1, len(remaining_indices) // (remaining_len // 2))
            additional_indices_backward = remaining_indices[::-1][::step_backward]
        else:
            additional_indices_backward = []

        additional_indices = additional_indices_forward[:remaining_len // 2] + additional_indices_backward[
                                                                               :remaining_len // 2]
        sampled_indices.update(additional_indices[:remaining_len])

    sampled_indices = sorted(sampled_indices)
    sampled_data = preprocessed_data.iloc[sampled_indices]

    return sampled_data, simplified_points, range_index


def simplify_key_points(key_points, min_distance):
    if not key_points:
        return []

    simplified_points = [key_points[0]]
    last_point = key_points[0]

    for point in key_points[1:]:
        if point - last_point >= min_distance:
            simplified_points.append(point)
            last_point = point

    return simplified_points


def visualize_flight_with_angles(df, callsign, output_folder, window_len=10, angle_threshold=15):
    flight_data = df[df['Callsign'] == callsign]
    flight_data = flight_data.reset_index(drop=True)

    if flight_data.empty:
        print(f"No data found for Callsign {callsign}")
        return

    flight_data.interpolate(method='linear', inplace=True)
    flight_data.fillna(method='bfill', inplace=True)
    flight_data.fillna(method='ffill', inplace=True)

    preprocessed_data = flight_data.copy()
    preprocessed_data = preprocessed_data.reset_index(drop=True)

    angles = calculate_angles(preprocessed_data, window_len)

    preprocessed_data['angle'] = 0

    preprocessed_data.iloc[window_len:-window_len, preprocessed_data.columns.get_loc('angle')] = angles

    seq_len = 288
    activity_init_points = preprocessed_data.index[preprocessed_data['angle'] > angle_threshold].tolist()

    # activity_init_points = simplify_key_points(activity_init_points, 5)
    patch_len = 32
    sampled_data, center_points, range_index = sample_trajectory(preprocessed_data, seq_len, activity_init_points, patch_len)

    fig = plt.figure(figsize=(18, 12))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(flight_data['longitude'], flight_data['latitude'], flight_data['altitude'], color='green', linestyle='-.',
             label='Original Trajectory')
    # ax1.plot(flight_data['longitude'][200:300], flight_data['latitude'][200:300], flight_data['altitude'][200:300], color='green', linestyle='-.', label='Original Trajectory')
    # ax1.scatter(flight_data['longitude'][::15], flight_data['latitude'][::15], flight_data['altitude'][::15], color='green', marker='o', s=5)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_zlabel('Altitude (feet)')
    ax1.set_title(f'Original Trajectory of Callsign: {callsign}')
    ax1.legend()

    ax2 = fig.add_subplot(132, projection='3d')
    start = 300
    end = 350
    # ax2.plot(preprocessed_data['longitude'][start:end], preprocessed_data['latitude'][start:end], preprocessed_data['altitude'][start:end], color='blue', linestyle='-.', label='Preprocessed Trajectory')

    # for i in range(0, len(preprocessed_data), 5):
    # # for i in range(start, end, 1):
    #     point = preprocessed_data.iloc[i]
    #     ax2.text(point['longitude'], point['latitude'], point['altitude'], f"{point['angle']:.1f}°", fontsize=8, color='black')

    window_len = 32
    preprocessed_data = smooth_flight_data(preprocessed_data, preprocessed_data['angle'], window_len, angle_threshold)

    # ax2.plot(preprocessed_data['longitude'][200:300], preprocessed_data['latitude'][200:300], preprocessed_data['altitude'][200:300], color='blue', linestyle='-.', label='Preprocessed Trajectory')
    ax2.plot(preprocessed_data['longitude'], preprocessed_data['latitude'], preprocessed_data['altitude'], color='blue',
             linestyle='-.', label='Preprocessed Trajectory')

    # simplified_points = activity_init_points[::3]
    # simplified_points = activity_init_points
    simplified_points = center_points

    simplified_points_indices = preprocessed_data.index.intersection(simplified_points)
    large_angles_sampled = preprocessed_data.loc[simplified_points_indices]
    ax2.scatter(large_angles_sampled['longitude'], large_angles_sampled['latitude'], large_angles_sampled['altitude'],
                color='red', marker='s', s=2, label=f'Angles > {angle_threshold}°')

    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_zlabel('Altitude (feet)')
    ax2.set_title(f'Preprocessed Trajectory with Angles of Callsign: {callsign}')
    ax2.legend()

    ax3 = fig.add_subplot(133, projection='3d')

    # simplified_points_indices = preprocessed_data.index.intersection(range_index)
    # large_angles_sampled = preprocessed_data.loc[simplified_points_indices]
    # ax3.scatter(large_angles_sampled['longitude'], large_angles_sampled['latitude'], large_angles_sampled['altitude'],
    #             color='black', marker='x', s=15)

    for i, start in enumerate(range(0, len(sampled_data), patch_len)):
        segment = sampled_data.iloc[start:start + patch_len]
        ax3.plot(segment['longitude'], segment['latitude'], segment['altitude'], color='blue', linestyle='-', alpha=0.6)
        ax3.scatter(segment['longitude'].iloc[0], segment['latitude'].iloc[0], segment['altitude'].iloc[0],
                    color='black', marker='x', s=15)

    simplified_points_indices = preprocessed_data.index.intersection(simplified_points)
    large_angles_sampled = preprocessed_data.loc[simplified_points_indices]
    ax3.scatter(large_angles_sampled['longitude'], large_angles_sampled['latitude'], large_angles_sampled['altitude'],
                color='red', marker='s', s=5, label=f'Simplified Angles > {angle_threshold}°')

    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_zlabel('Altitude (feet)')
    ax3.set_title(f'Sampled Trajectory with Angles of Callsign: {callsign}')
    ax3.legend()


    output_path = os.path.join(output_folder, f"{callsign}.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def process_all_flights(csv_file, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(csv_file)
    callsigns = df['Callsign'].unique()[0:10]

    for callsign in tqdm(callsigns):
        visualize_flight_with_angles(df, callsign, output_folder)


csv_file_path = 'aircraft.csv'
output_folder_path = 'simple_plot'
process_all_flights(csv_file_path, output_folder_path)