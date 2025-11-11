import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# Assuming df is your DataFrame and already loaded
df = pd.read_parquet('squawk7700_trajectories.parquet.gz', engine='pyarrow')

# Base directory to save images
base_dir = 'all_flight_trajectories'
os.makedirs(base_dir, exist_ok=True)

# Iterate through each unique flight_id
for flight_id, flight_data in tqdm(df.groupby('flight_id'), desc='Processing flights'):
    squawk = flight_data['squawk'].iloc[0]

    # Create a figure and a 3D axis for each flight
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the trajectory
    ax.plot(flight_data['longitude'], flight_data['latitude'], flight_data['altitude'])

    # Set labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude')
    ax.set_title(f'3D Trajectory of Flight ID: {flight_id} (Squawk: {squawk})')

    # Save the plot as an image
    filepath = os.path.join(base_dir, f"{flight_id}.png")
    plt.savefig(filepath)

    # Close the plot to free memory
    plt.close(fig)
