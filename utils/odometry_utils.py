import numpy as np


def position_to_velocity(points, time_intervals, starting_point=np.array([0, 0])):
    """Convert position trajectory to velocity estimates.
    
    Args:
        points: Array of 2D positions, shape (n, 2).
        time_intervals: Array of time intervals between consecutive positions, shape (n,).
        starting_point: Initial position before the first point. Defaults to [0, 0].
        
    Returns:
        np.ndarray: Array of velocities, shape (n, 2).
    """
    assert len(points) == len(time_intervals), f'The length of points({len(points)}) must equal to the length of time_intervals({len(time_intervals)})'
    pre_point = starting_point   # Assuming all tracking data start at (0, 0), and the first step is not far from the origin
    velocities = []
    for i in range(len(points)):
        displacement = points[i] - pre_point
        velocities.append(displacement / time_intervals[i])
        pre_point = points[i]

    assert len(velocities) == len(points), 'The number of velocities should be equal the number of points'
    return np.array(velocities)


def velocity_to_position(velocities, time_intervals):
    """Convert velocity estimates to position trajectory.
    
    Args:
        velocities: Array of 2D velocities, shape (n, 2).
        time_intervals: Array of time intervals, shape (n,).
        
    Returns:
        np.ndarray: Array of positions, shape (n, 2), starting from origin.
    """
    assert len(velocities) == len(time_intervals), f'The length of points({len(velocities)}) must equal to the length of time_intervals({len(time_intervals)})'
    current_point = np.array([0.0, 0.0])   # Assuming all tracking data start at (0, 0), and the first step is not far from the origin
    points = []
    for i in range(len(velocities)):
        displacement = velocities[i] * time_intervals[i]
        current_point = current_point + displacement 
        points.append(current_point)

    assert len(points) == len(points), 'The number of velocities should be equal the number of points'
    return np.array(points)


def modify_tracking_data(tracking_data, time_intervals, scale_factor):
    """Apply scale correction to tracking data via velocity scaling.
    
    Extracts velocities from position trajectory, applies a scale factor, and
    reconstructs the trajectory with scaled velocities.
    
    Args:
        tracking_data: Array of 2D positions, shape (n, 2).
        time_intervals: Array of time intervals, shape (n,).
        scale_factor: Scaling factor to apply to velocities.
        
    Returns:
        np.ndarray: Modified position trajectory with scaled velocities applied.
    """
    velocities = position_to_velocity(points=tracking_data, time_intervals=time_intervals)
    velocities *= scale_factor
    new_tracking_data = velocity_to_position(velocities=velocities, time_intervals=time_intervals)
    return new_tracking_data