import numpy as np
import pandas as pd
from Tracking import Tracking

def add_sensor_noise(x_true, y_true, sigma_x, sigma_y):
    """
    Function to add Gaussian noise to a 2D ground truth measurement.

    Args:
    x_true: Ground truth x-coordinate
    y_true: Ground truth y-coordinate
    sigma_x: Standard deviation of the noise in the x direction
    sigma_y: Standard deviation of the noise in the y direction

    Returns:
    x_noisy: Noisy x-coordinate
    y_noisy: Noisy y-coordinate
    """

    # Add Gaussian noise
    x_noisy = x_true + np.random.normal(0, sigma_x)
    y_noisy = y_true + np.random.normal(0, sigma_y)

    return x_noisy, y_noisy


root = '../data/'


# Read and parse CSV data
tracks_meta_df = pd.read_csv(root + '32_tracksMeta.csv')
tracks_df = pd.read_csv(root + '32_tracks.csv')

df = pd.merge(tracks_df, tracks_meta_df[['trackId', 'class']], on='trackId', how='left')

# Constants
SENSOR_RANGE = 85

# Filter rows where class is 'car' or 'pedestrian'
car_pedestrian_df = df[df['class'].isin(['car', 'pedestrian'])]

# Group the DataFrame by car ID
grouped_df = car_pedestrian_df.groupby('trackId')

# Get unique car IDs
unique_car_ids = car_pedestrian_df[car_pedestrian_df['class'] == 'car']['trackId'].unique()

# Iterate through unique car IDs
for car_id in unique_car_ids:
    car_df = car_pedestrian_df[(car_pedestrian_df['class'] == 'car') & (car_pedestrian_df['trackId'] == car_id)]
    pedestrian_df = car_pedestrian_df[(car_pedestrian_df['class'] == 'pedestrian') & (car_pedestrian_df['frame'].isin(car_df['frame']))]

    vruTracking = None
    last_pedestrian_track_id = -1
    # Iterate through pedestrian rows for the current car
    for _, pedestrian_row in pedestrian_df.iterrows():

        pedestrian_track_id = pedestrian_row['trackId']
        pedestrian_x_center = pedestrian_row['xCenter']
        pedestrian_y_center = pedestrian_row['yCenter']

        # Access the current car data as well
        car_x_center = car_df.iloc[0]['xCenter']
        car_y_center = car_df.iloc[0]['yCenter']

        # Perform any desired operations with the car and pedestrian data
        print(f"Processing car with track ID {car_id} and pedestrian with track ID {pedestrian_track_id}")
        print(f"Car center: ({car_x_center}, {car_y_center}), Pedestrian center: ({pedestrian_x_center}, {pedestrian_y_center})")

        if (pedestrian_track_id != last_pedestrian_track_id):
            vruTracking = Tracking(pedestrian_track_id, pedestrian_x_center, pedestrian_y_center)
        else:
            vruTracking.update(pedestrian_x_center, pedestrian_y_center)

        last_pedestrian_track_id = pedestrian_track_id