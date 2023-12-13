import pandas as pd
import numpy as np
import ast
import math

def engineer_features(dataframe):
    
    # Create acceleration feature by obtaining the acceleration magnitude
    
    # Obtain rows of all accelerometer axes
    accelerometer_x_rows = []
    accelerometer_y_rows = []
    accelerometer_z_rows = []
    accelerometer_axes_rows = [accelerometer_x_rows, accelerometer_y_rows, accelerometer_z_rows]
    accelerometer_axes_names = ['accelerometer_x', 'accelerometer_y', 'accelerometer_z']
    
    for index, accelerometer_axis_name in enumerate(accelerometer_axes_names):
        for accelerometer_axis_row in dataframe[accelerometer_axis_name]:
            accelerometer_axis_row_float = [float(x) for x in ast.literal_eval(accelerometer_axis_row)]
            accelerometer_axes_rows[index].append(accelerometer_axis_row_float)
    
    # Calculate overall acceleration magnitude from the rows of all accelerometer axes
    acceleration_magnitude_rows = []
    
    for row in range(0, 1):
        magnitude_row = []
        for element in range(0, 6):
            magnitude = math.sqrt(pow(accelerometer_x_rows[row][element], 2) + pow(accelerometer_y_rows[row][element], 2) + pow(accelerometer_z_rows[row][element], 2))
            magnitude_row.append(magnitude)
        acceleration_magnitude_rows.append(magnitude_row)
    
    # Add new acceleration feature to dataframe
    dataframe['acceleration'] = acceleration_magnitude_rows
    
    # Create yaw_speed feature by obtaining the gyroscope_z magnitude
    # Gyroscope_z is yaw velocity (angular velocity of z-axis), so the magnitude of gyroscope_z is yaw speed
    yaw_speed_rows = []
    
    # Obtain magnitude for all gyroscope_z values using absolute function
    for gyroscope_z_row in dataframe['gyroscope_z']:
        gyroscope_z_row_float = [float(x) for x in ast.literal_eval(gyroscope_z_row)]
        yaw_speed_row = [abs(x) for x in gyroscope_z_row_float]
        yaw_speed_rows.append(yaw_speed_row)
    
    # Add new yaw_speed feature to dataframe
    dataframe['yaw_speed'] = yaw_speed_rows
    
    # Create steer_change feature based on changes in steer value
    steer_change_rows = []
    
    for steer_row in dataframe['steer']:
        steer_row_float = [float(x) for x in ast.literal_eval(steer_row)]
        
        steer_change_row = np.diff(steer_row_float).tolist()
        # Obtain steer change magnitude using absolute function
        steer_change_row_abs = [abs(x) for x in steer_change_row]
        steer_change_rows.append(steer_change_row_abs)
    
    # Add new steer_change feature to dataframe
    dataframe['steer_change'] = steer_change_rows
    
    # Create positive_brake_change feature based on increments in brake threshold value
    positive_brake_change_rows = []
    
    for brake_row in dataframe['brake']:
        brake_row_float = [float(x) for x in ast.literal_eval(brake_row)]
        
        brake_change_row = np.diff(brake_row_float).tolist()
        # Keep only positive brake changes (pressing of brake pedal)
        positive_brake_change_row = [x for x in brake_change_row if x > 0]
        positive_brake_change_rows.append(positive_brake_change_row)
    
    # Add new positive_brake_change feature to dataframe
    dataframe['positive_brake_change'] = positive_brake_change_rows
    
    # Remove old features that have been engineered from dataframe
    dataframe = dataframe.drop(columns=['accelerometer_x', 'accelerometer_y', 'accelerometer_z', 'gyroscope_z', 'steer', 'brake'])
    
    # Move abnormality column to the end
    abnormality_column = dataframe.pop('abnormality')
    dataframe.insert(len(dataframe.columns), "abnormality", abnormality_column)
    
    return dataframe