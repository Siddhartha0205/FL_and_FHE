import pandas as pd
import numpy as np
import ast
import math

def aggregate_features(dataframe):
    
    # Define aggregation functions
    # row_is_string indicates if the value of a dataframe column row is a string
    # If it is a string, convert it to a list of floats before aggregation
    # Features will have column row values in string type by default unless engineered
    
    # Aggregate list of values in all column rows to obtain a single mean value for each row
    def aggregate_mean(column_name, row_is_string=True):
        mean_list = []
        for row in dataframe[column_name]:
            if row_is_string:
                row = [float(x) for x in ast.literal_eval(row)]
            mean_value = np.mean(row)
            mean_list.append(mean_value)
        return mean_list
    
    # Aggregate list of values in all column rows to obtain a single maximum value for each row
    def aggregate_max(column_name, row_is_string=True):
        max_list = []
        for row in dataframe[column_name]:
            if row_is_string:
                row = [float(x) for x in ast.literal_eval(row)]
            max_value = np.max(row)
            max_list.append(max_value)
        return max_list
    
    # Aggregate list of values in all column rows to obtain the count of maximum value occurences for each row
    def aggregate_max_count(column_name, max_value, row_is_string=True):
        max_count_list = []
        for row in dataframe[column_name]:
            if row_is_string:
                row = [float(x) for x in ast.literal_eval(row)]
            max_value_count = row.count(max_value)
            max_count_list.append(max_value_count)
        return max_count_list
    
    # Aggregate speed feature to mean_speed and max_speed, then add them to dataframe
    dataframe['mean_speed'] = aggregate_mean('speed')
    dataframe['max_speed'] = aggregate_max('speed')
    
    # Aggregate acceleration feature to mean_acceleration and max_acceleration, then add them to dataframe
    dataframe['mean_acceleration'] = aggregate_mean('acceleration', row_is_string=False)
    dataframe['max_acceleration'] = aggregate_max('acceleration', row_is_string=False)
    
    # Aggregate yaw_speed feature to mean_yaw_speed and max_yaw_speed, then add them to dataframe
    dataframe['mean_yaw_speed'] = aggregate_mean('yaw_speed', row_is_string=False)
    dataframe['max_yaw_speed'] = aggregate_max('yaw_speed', row_is_string=False)
    
    # Aggregate throttle feature to mean_throttle and max_throttle_count, then add them to dataframe
    dataframe['mean_throttle'] = aggregate_mean('throttle')
    dataframe['max_throttle_count'] = aggregate_max_count('throttle', 1.0)
    
    # Aggregate steer_change feature to mean_steer_change and max_steer_change, then add them to dataframe
    dataframe['mean_steer_change'] = aggregate_mean('steer_change', row_is_string=False)
    dataframe['max_steer_change'] = aggregate_max('steer_change', row_is_string=False)
    
    # Aggregate positive brake change feature to mean_positive_brake_change and max_positive_brake_change, then add them to dataframe
    dataframe['mean_positive_brake_change'] = aggregate_mean('positive_brake_change', row_is_string=False)
    dataframe['max_positive_brake_change'] = aggregate_max('positive_brake_change', row_is_string=False)
    
    # Remove old features that have been aggregated from dataframe
    dataframe = dataframe.drop(columns=['speed', 'acceleration', 'yaw_speed', 'throttle', 'steer_change', 'positive_brake_change'])
    
    # Move abnormality column to the end
    abnormality_column = dataframe.pop('abnormality')
    dataframe.insert(len(dataframe.columns), "abnormality", abnormality_column)
    
    return dataframe