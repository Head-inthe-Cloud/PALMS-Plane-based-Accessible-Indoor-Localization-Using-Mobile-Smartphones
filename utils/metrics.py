import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import csv
import os
from tqdm import tqdm
from collections import defaultdict


class Metrics:
    def __init__(self, tolerance=1, resolution=0.1, temp_data_path=None):
        self.tolerance = tolerance  # The accaptance distance
        self.resolution = resolution
        self.temp_data_path = temp_data_path # csv path, Save intermediate data from each update, avoid data loss from sudden stop
        self.reset()

    def reset(self):
        """Reset all metrics to initial state."""
        self.data = defaultdict(lambda: {
            'distance': [],
            'angular_diff': [],
            'total_count': 0
        })
    
    def load_temp_data(self):
        """Load previously saved metric data from temporary CSV file.
        
        Requires temp_data_path to be set during initialization.
        """
        if self.temp_data_path is None:
            print('You need to provide a temp_data_path before loading')
            return
        temp_metrics = read_temp_data(self.temp_data_path)
        for temp_metric in tqdm(temp_metrics, desc='Loading saved metric data'):
            scene_id = temp_metric['scene_id']
            for key in ['obs_path', 'rank', 'confidence', 'ambiscore', 'distance', 'angular_diff']:
                temp_data = temp_metric[key]
                if key in ['rank', 'confidence', 'ambiscore', 'distance', 'angular_diff']:
                    temp_data = float(temp_data)
                self.data[scene_id][key].append(temp_data)
            self.data[scene_id]['total_count'] += 1


    def update(self, scene_id, pose_pred, label):
        '''
        params:
            scene_id: the name of the scene, it could be building name
            pose_pred: the predicted pose [x, y, theta], in meters and radians
            label: the labeled pose [x, y, theta], in meters and radians
        '''

        location = label[:2]
        gt_theta = label[2]
        
        pred_x, pred_y = pose_pred[:2]
        pred_theta = pose_pred[2]
        gt_x, gt_y = location

        # Calculate distance
        distance = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)

        # Calculate angular distance
        angular_diff = np.abs(np.arctan2(np.sin(gt_theta - pred_theta), np.cos(gt_theta - pred_theta)))

        self.data[scene_id]['distance'].append(distance)
        self.data[scene_id]['angular_diff'].append(angular_diff)
        self.data[scene_id]['total_count'] += 1
        
        # Save data at temporary location
        if self.temp_data_path is not None:
            if not os.path.exists(self.temp_data_path):
                # Write header
                with open(self.temp_data_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['scene_id', 'distance', 'angular_diff'])
                    
            with open(self.temp_data_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([scene_id, distance, angular_diff])
                

    def get_metrics(self):
        """Compute and return localization accuracy metrics.
        
        Calculates recall rates at various distance thresholds (0.1m, 0.5m, 1m, 2m, 5m)
        and combined distance+angle metrics.
        
        Returns:
            dict: Dictionary mapping scene_id to metrics dictionary containing:
                - recall_0.1m: Recall at 0.1m threshold
                - recall_0.5m: Recall at 0.5m threshold
                - recall_1m: Recall at 1m threshold
                - recall_2m: Recall at 2m threshold
                - recall_5m: Recall at 5m threshold
                - recall_1m_30deg: Recall at 1m and 30° threshold
        """
        result = {}
        for scene_id, values in self.data.items():
            total = values['total_count']
            if total == 0:
                continue
            
            success_count_0_1m = np.sum(np.array(values['distance']) <= 0.1)
            success_count_0_5m = np.sum(np.array(values['distance']) <= 0.5)
            success_count_1m = np.sum(np.array(values['distance']) <= 1)
            success_count_2m = np.sum(np.array(values['distance']) <= 2)
            success_count_5m = np.sum(np.array(values['distance']) <= 5)
            success_count_1m_30deg = np.sum((np.array(values['distance']) <= 1) & (np.array(values['angular_diff']) <= np.radians(30)))

            result[scene_id] = {
                'recall_0.1m': success_count_0_1m / total,
                'recall_0.5m': success_count_0_5m / total,
                'recall_1m': success_count_1m / total,
                'recall_2m': success_count_2m / total,
                'recall_5m': success_count_5m / total,
                'recall_1m_30deg': success_count_1m_30deg / total
            }

        return result
    

def RMSE(pred, label):
    '''
    Calculate the RMSE between two sets of 2D coordinates
    
    Parameters:
        pred - numpy array of shape (n, 2)
        label - numpy array of shape (n, 2)
    
    Returns:
        score - the RMSE score, float
    '''
    # Compute the squared differences
    squared_diff = np.square(pred - label)
    
    # Average the squared differences
    mean_squared_diff = np.mean(squared_diff)
    
    # Take the square root of the average squared differences
    score = np.sqrt(mean_squared_diff)
    
    return score


def read_temp_data(csv_path):
    """Read temporary metric data from CSV file.
    
    Args:
        csv_path: Path to the CSV file containing metric data.
        
    Returns:
        list: List of dictionaries containing metric data rows.
    """
    data = []
    if not os.path.exists(csv_path):
        print(f'File {csv_path} does not exist.')
        return data

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data