import pandas as pd

def create_new_features(reduced_data):
    
    # Create a new feature 'TotalMinutes' by summing 'DayMins' and 'RoamMins'
    reduced_data['TotalMinutes'] = reduced_data['DayMins'] + reduced_data['RoamMins']
    
    return reduced_data

def select_features(reduced_data, feature_list):
    # Select features from the dataset
    return reduced_data[feature_list]
