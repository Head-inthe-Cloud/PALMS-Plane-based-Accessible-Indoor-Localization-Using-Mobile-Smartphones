import numpy as np
from utils.geometry_utils import alpha_shape, rotate_segments_to_landscape, segments_to_binary_map
from utils.geometry_utils import points_to_path_distances, point_to_point_distances, find_optimal_rotation
from utils.geometry_utils import apply_transformation_to_points, get_R_from_orientations
from utils.odometry_utils import *
from utils.visualization import visualize
from palms_src.particle_filter import particle_filter
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt

class PF_Simulator():
    """
    Particle Filter Simulator for PALMS sequential localization.
    
    This class manages the particle filter-based localization system that tracks
    a user's position over time using odometry data and observation heatmaps.
    It supports both ARKit and RoNIN tracking data sources.
    
    Attributes:
        config (dict): Configuration dictionary containing resolution, PF settings, etc.
        vector_map (list): List of line segments representing the floor plan walls.
        binary_map (np.ndarray): Binary occupancy grid of the floor plan.
        map_mask (np.ndarray): Boolean mask indicating valid map regions.
        AR_tracking_data (np.ndarray): ARKit odometry trajectory data.
        RoNIN_tracking_data (np.ndarray): RoNIN odometry trajectory data (if used).
        AR_PF_data (np.ndarray): Ground truth trajectory for evaluation.
        velocities (np.ndarray): Velocity estimates from odometry.
        pf (particle_filter): Particle filter instance.
        metrics (dict): Dictionary storing localization metrics.
        
    Example:
        >>> config = {
        ...     'resolution': 0.1,
        ...     'tracking_data_source': 'ARKit',
        ...     'pf_config': {...}
        ... }
        >>> simulator = PF_Simulator(vector_map, tracking_data, config)
        >>> result = simulator.run_pf_global(heatmaps=heatmaps, obs_thetas=thetas)
    """
    def __init__(self, vector_map, tracking_data, config=None, 
                 map_transform=None, binary_map=None, map_mask=None,
                 crop_steps=0):
        """
        Initialize the particle filter simulator.
        
        Args:
            vector_map (list): List of line segments representing floor plan walls.
                Each segment is a numpy array of shape (2, 2) with [start, end] points.
            tracking_data (dict): Dictionary containing odometry tracking data with keys:
                - 'ARKit_raw_2D': ARKit trajectory positions
                - 'ARKit_PF': Ground truth trajectory for evaluation
                - 'RoNIN_raw_2D': RoNIN trajectory positions (optional)
                - 'time_intervals': Time intervals between positions
                - 'starting_vector': Initial direction vector
            config (dict): Configuration dictionary. Required keys:
                - 'resolution': Map resolution in meters per pixel
                - 'tracking_data_source': 'ARKit' or 'RoNIN'
                - 'pf_config': Particle filter configuration dictionary
            map_transform (np.ndarray, optional): 3x3 transformation matrix for map alignment.
                If None, map will be rotated to landscape orientation.
            binary_map (np.ndarray, optional): Pre-computed binary occupancy grid.
                If None, will be computed from vector_map.
            map_mask (np.ndarray, optional): Pre-computed map mask.
                If None, will be computed using alpha shape.
            crop_steps (int, optional): Number of initial steps to crop from tracking data.
                Default is 0.
        """
        self.config = config
        self.use_AR_tracking_data = config['tracking_data_source'] == 'ARKit' 
        self.resolution = config['resolution']
        self.pf_config = config['pf_config']
        self.pf_init_method = self.pf_config['pf_init_method']
        self.uniform_drift = self.pf_config['init_method'] == 'uniform'
        self.gif_path = None

        self.metrics = {'trace_length': [],
                        'converge_time': [],
                        'converge_distance': [],
                        'RMSE': [],
                        'RMSE_last_10': [],
                        'conv_success': [],
                        'success_1m': [],
                        'end_distance': [],
        }               
        self.n_simulations = 0     # Total number of simulations ran, used for Metric calculation

        if map_transform is None:
            self.vector_map, self.map_transform, _ = rotate_segments_to_landscape(vector_map)
        else:
            self.vector_map = vector_map
            self.map_transform = map_transform

        self.binary_map = segments_to_binary_map(self.vector_map, cell_size=self.resolution) if binary_map is None else binary_map
        self.map_mask = alpha_shape(self.vector_map, visualize=False) if map_mask is None else map_mask

        # The angle to rotate the observation and tracking data for
        self.obs_theta = 0
        self.obs_transform = np.eye(3)
        
        # Read tracking data
        self.starting_vector = np.array(tracking_data['starting_vector'])
        self.AR_tracking_data = np.array(tracking_data['ARKit_raw_2D']) * -1 # For some reason AR_tracking_data is reversed, so we invert it back
        self.RoNIN_tracking_data = None
        self.AR_PF_data = np.array(tracking_data['ARKit_PF'])
        self.time_intervals = np.array(tracking_data['time_intervals'])

        # Remove the first 6 steps from tracking data
        self.AR_tracking_data = self.AR_tracking_data[6:]
        self.time_intervals = self.time_intervals[6:]

        self.tracking_data_starting_point = np.array([0, 0])
        if crop_steps > 0:
            self.AR_PF_data = self.AR_PF_data[crop_steps:]
            
            # We use +6 here becase tracking data have 6 extra steps in the beginning compared to AR PF data
            self.time_intervals = self.time_intervals[crop_steps:]
            if self.use_AR_tracking_data:
                self.tracking_data_starting_point = self.AR_tracking_data[crop_steps-1]
                self.AR_tracking_data = self.AR_tracking_data[crop_steps:]
            else:
                self.tracking_data_starting_point = self.RoNIN_tracking_data[crop_steps-1]
                self.RoNIN_tracking_data = self.RoNIN_tracking_data[crop_steps:]
            

        # print(f'AR {len(self.AR_tracking_data)}, RoNIN {len(self.RoNIN_tracking_data)}, AR_PF {len(self.AR_PF_data)}, Time {len(self.time_intervals)}')
        # Rotate AR_PF_data to align with the floor plan
        self.AR_PF_data = apply_transformation_to_points(self.AR_PF_data, self.map_transform)

        if not self.use_AR_tracking_data:
            self.RoNIN_tracking_data = np.array(tracking_data['RoNIN_raw_2D'])[6:]
            # Rotate RoNIN tracking data to align with AR tracking data
            T = np.eye(3)
            R = find_optimal_rotation(self.RoNIN_tracking_data, self.AR_tracking_data)
            T[:2, :2] = R
            self.RoNIN_tracking_data = apply_transformation_to_points(self.RoNIN_tracking_data, T)

            # Note: Scale correction for RoNIN data. The scale_factor=1.1 corrects for systematic
            # scale errors in RoNIN tracking. If your RoNIN data uses a different scale calibration,
            # adjust this factor accordingly. Alternative scale factor (1.09/1.27) is commented out
            # for reference but may be needed for different data collection setups.
            self.RoNIN_tracking_data = modify_tracking_data(self.RoNIN_tracking_data, time_intervals=self.time_intervals, scale_factor=1.1)

        # Position to volocity
        if self.use_AR_tracking_data:
            self.velocities = position_to_velocity(self.AR_tracking_data, self.time_intervals, self.tracking_data_starting_point)
        else:
            self.velocities = position_to_velocity(self.RoNIN_tracking_data, self.time_intervals, self.tracking_data_starting_point)

        self.pf = None
        self.initial_particles = None    # For visualization purpose only

    def rotate_obs(self, theta):
        """Rotate observation and tracking data by given angle.
        
        Applies rotation transformation to align observations and tracking data
        with the map coordinate frame.
        
        Args:
            theta: Rotation angle in radians.
        """
        R, _ = get_R_from_orientations(theta=theta)
        self.obs_transform[:2, :2] = R
        # Rotate tracking data to match the rotated map
        self.starting_vector = apply_transformation_to_points(self.starting_vector, self.obs_transform)
        self.tracking_data_starting_point = apply_transformation_to_points(np.array([self.tracking_data_starting_point]), self.obs_transform)[0]
        # Position to volocity
        if self.use_AR_tracking_data:
            self.AR_tracking_data = apply_transformation_to_points(self.AR_tracking_data, self.obs_transform)
            self.velocities = position_to_velocity(self.AR_tracking_data, self.time_intervals, self.tracking_data_starting_point)
        else:
            self.RoNIN_tracking_data = apply_transformation_to_points(self.RoNIN_tracking_data, self.obs_transform)
            self.velocities = position_to_velocity(self.RoNIN_tracking_data, self.time_intervals, self.tracking_data_starting_point)


    def run_pf_local(self, visualize_palms=False):
        """Run particle filter with local initialization.
        
        Initializes particles near the starting point and runs the particle filter
        to track the user's position over time.
        
        Args:
            visualize_palms: If True, displays visualization during tracking.
            
        Returns:
            dict: Dictionary containing:
                - 'trace': Array of estimated positions over time
                - 'dispersions': Array of particle dispersion values
                - 'distances': Array of distances to ground truth
                - 'avg_drifts': Array of average drift values
                - 'convergence_idx': Index where convergence occurred (if any)
        """
        # Initialize pf locally
        self.pf = particle_filter(vector_map=self.vector_map, binary_map=self.binary_map, map_mask=self.map_mask, global_init=False, pf_init_method='basic', start_point=self.starting_vector[0], pf_config=self.pf_config)
        self.initial_particles = self.pf.extract_points()

        # Run PF
        if visualize_palms:
            pf_local_visual_list = self.initial_particles
            visualize(segments_1=self.vector_map, points_1=pf_local_visual_list)
            pf_result = self.run_pf_with_visual(self.pf)
            visualize(segments_1=self.vector_map, points_2=pf_result['trace'])
        else:
            pf_result = self.run_pf_without_visual(self.pf)

        return pf_result


    def run_pf_global(self, visualize_palms=False, obs_thetas=None, heatmaps=None):
        """Run particle filter with global initialization.
        
        Initializes particles globally across the map, optionally using heatmaps
        and observation orientations to guide initialization.
        
        Args:
            visualize_palms: If True, displays visualization during tracking.
            obs_thetas: Optional list of observation orientations in radians.
                Used to initialize particles with orientation constraints.
            heatmaps: Optional list of heatmaps (one per orientation).
                Used to sample initial particle locations.
                
        Returns:
            dict: Dictionary containing:
                - 'trace': Array of estimated positions over time
                - 'dispersions': Array of particle dispersion values
                - 'distances': Array of distances to ground truth
                - 'avg_drifts': Array of average drift values
                - 'convergence_idx': Index where convergence occurred (if any)
        """
        if obs_thetas is None:
            self.pf = particle_filter(vector_map=self.vector_map, binary_map=self.binary_map, map_mask=self.map_mask, 
                                      global_init=True, pf_init_method=self.pf_init_method, uniform_drift=self.uniform_drift, 
                                      pf_config=self.pf_config)
            particle_visual_list = self.pf.extract_points()
            self.initial_particles = [particle_visual_list]
        else:
            if heatmaps is None:
                # Note: Fallback initialization when heatmaps are not provided.
                # This uses uniform sampling with orientation constraints based on obs_thetas.
                # For best results, provide heatmaps from PALMS/PALMS+ layout matching.
                # Uniformly sample location if heatmaps are not provided
                self.pf = particle_filter(vector_map=self.vector_map, binary_map=self.binary_map, map_mask=self.map_mask, 
                                          global_init=True, pf_init_method=self.pf_init_method, drift_theta=obs_thetas[0], 
                                          heatmap=None, uniform_drift=self.uniform_drift, 
                                          pf_config=self.pf_config, group=0)
                particle_visual_list = self.pf.extract_points()
                self.initial_particles = [particle_visual_list]
                total_particle_list = self.pf.particle_list

                for i in range(1, len(obs_thetas)):
                    self.pf.drift_theta = obs_thetas[i]  # Update drift direction
                    self.pf.group = i  # Set particle group
                    particle_list = self.pf.initialize_pf_global()
                    total_particle_list += particle_list
                    particle_visual_list = self.pf.extract_points(particle_list)
                    if visualize_palms and self.gif_path is None:
                        visualize(segments_1=self.vector_map / self.resolution, points_1=particle_visual_list/self.resolution, heatmap=heatmaps[i])
                    self.initial_particles.append(particle_visual_list)

            else:
                # Use heatmaps to sample locations, and use obs_thetas to initialize PF
                self.pf = particle_filter(vector_map=self.vector_map, binary_map=self.binary_map, map_mask=self.map_mask, 
                                          global_init=True, pf_init_method=self.pf_init_method, drift_thetas=obs_thetas, 
                                          heatmaps=heatmaps, uniform_drift=self.uniform_drift,
                                          pf_config=self.pf_config, group=0)
                
                self.initial_particles = [self.pf.extract_points(group=group) for group in range(len(obs_thetas))]
                total_particle_list = self.pf.particle_list

                # TEMP: Vis
                # for i in range(1, len(obs_thetas)):
                #     particle_visual_list = self.pf.extract_points(group=i)
                #     if visualize_palms and self.gif_path is None:
                #         visualize(segments_1=self.vector_map / self.resolution, points_1=particle_visual_list/self.resolution, heatmap_1=heatmaps[i])
                
        # Show map with PF
        if visualize_palms:
            # visualize(segments_1=self.vector_map, points_group=self.initial_particles)
            pf_result = self.run_pf_with_visual(self.pf)
        else:
            pf_result = self.run_pf_without_visual(self.pf)

        return pf_result
        

    def run_pf_with_visual(self, particle_filter, animation_interval=100):
        """
        Runs a particle filter simulation with visualization (any num_groups).
        Returns dict with trace, dispersions, distances, avg_drifts, convergence_idx.
        """
        trace = []
        dispersions = []
        avg_drifts = []
        distances = np.array([])

        stage1_converged = False
        main_group = None
        stage2_converged = False
        convergence_idx = None
        dispersion_threshold = 3

        mean_shift_counter = 0
        mean_shift_interval = 5
        mean_shift_threshold = 0.5
        max_label = None

        figure, ax = plt.subplots()
        for seg in self.vector_map:
            plt.plot(*zip(*seg), color='black')

        def _set_scatter_data(line, pts):
            if len(pts) > 0:
                xs, ys = zip(*pts)
                line.set_data(xs, ys)
            else:
                line.set_data([], [])

        if self.uniform_drift:
            particles_line, = plt.plot([], [], 'o', color='orange', markersize=1, label='Particles')
            particles_lines = [particles_line]  # unify interface
        else:
            # Dynamically create one artist per group
            num_groups = int(getattr(particle_filter, "num_groups", 4))
            cmap_colors = list(plt.cm.tab20.colors)  # enough distinct colors
            particles_lines = [
                plt.plot([], [], 'o',
                        color=cmap_colors[g % len(cmap_colors)],
                        markersize=1, label=f'Ori {g}')[0]
                for g in range(num_groups)
            ]

        centroid, = plt.plot([], [], 'ro', markersize=5)
        gt, = plt.plot([], [], 'o', color='#00FF00', markersize=5)
        gt_path, = plt.plot([], [], '-', color='#00FF00', linewidth=2)

        i = 0
        frames = len(self.time_intervals)

        def update(frame):
            nonlocal i, stage1_converged, stage2_converged, convergence_idx
            nonlocal main_group, mean_shift_counter, max_label, distances

            if i < frames:
                # Update PF
                particle_filter.process_noise_and_update(self.velocities[i], self.time_intervals[i])

                dispersion = particle_filter.get_dispersion()
                avg_drift = particle_filter.get_average_drift()

                if self.uniform_drift:
                    # Stage 1 (single group): use dispersion threshold
                    if not stage1_converged and dispersion <= dispersion_threshold:
                        stage1_converged = True
                        # split particles evenly among groups later if needed
                        particle_filter.particle_num //= max(1, getattr(particle_filter, "num_groups", 1))
                        print(f'Stage 1 @ {np.sum(self.time_intervals[:i])}')

                    pf_visual_list = particle_filter.extract_points()
                    _set_scatter_data(particles_lines[0], pf_visual_list)

                else:
                    num_groups = int(getattr(particle_filter, "num_groups", len(particles_lines)))
                    # Count and draw each group
                    particle_counts = np.zeros(num_groups, dtype=int)
                    for g in range(num_groups):
                        pf_visual_list = particle_filter.extract_points(group=g)
                        particle_counts[g] = len(pf_visual_list)
                        _set_scatter_data(particles_lines[g], pf_visual_list)

                    # Stage 1: dominant group > 80%
                    if not stage1_converged and np.any(particle_counts > particle_counts.sum() * 0.8):
                        stage1_converged = True
                        particle_filter.particle_num //= max(1, num_groups)
                        main_group = int(np.argmax(particle_counts))
                        print(f'Stage 1 @ {np.sum(self.time_intervals[:i])}')

                    # If dominant group changes post Stage 1, reset Stage 2 trackers
                    if stage1_converged:
                        temp_group = int(np.argmax(particle_counts))
                        if temp_group != main_group:
                            main_group = temp_group
                            stage2_converged = False
                            convergence_idx = None
                            max_label = None
                            mean_shift_counter = 0

                # Stage 2 convergence via mean-shift clustering
                if stage1_converged and not stage2_converged:
                    if mean_shift_counter % mean_shift_interval == 0:
                        if not self.uniform_drift and main_group is None:
                            main_group = 0
                        _max_label, _max_count, _max_prop = particle_filter.mean_shift(
                            group=main_group if not self.uniform_drift else None
                        )
                        if _max_prop >= mean_shift_threshold:
                            stage2_converged = True
                            convergence_idx = i
                            max_label = _max_label
                            mean_shift_counter = 0
                    mean_shift_counter += 1

                # After convergence: refresh cluster occasionally
                if stage2_converged:
                    if mean_shift_counter % 10 == 0:
                        _max_label, _max_count, _max_prop = particle_filter.mean_shift(
                            group=main_group if not self.uniform_drift else None
                        )
                        max_label = _max_label
                    mean_shift_counter += 1

                # Resample & center estimate
                if stage1_converged:
                    particle_filter.resample()
                    if self.uniform_drift:
                        max_cluster_particles = particle_filter.extract_points(label=max_label)
                    else:
                        max_cluster_particles = particle_filter.extract_points(group=main_group, label=max_label)

                    if len(max_cluster_particles) <= 0:
                        _max_label, _max_count, _max_prop = particle_filter.mean_shift(
                            group=main_group if not self.uniform_drift else None
                        )
                        max_label = _max_label
                        if self.uniform_drift:
                            max_cluster_particles = particle_filter.extract_points(label=max_label)
                        else:
                            max_cluster_particles = particle_filter.extract_points(group=main_group, label=max_label)

                    particle_center = particle_filter.mean_position(max_cluster_particles)
                else:
                    adjust_weight = not self.uniform_drift
                    particle_center = particle_filter.resample(adjust_weight=adjust_weight)

                if particle_center is None:
                    # PF failed; stop animation safely
                    animation.event_source.stop()
                    return

                centroid.set_data([particle_center[0]], [particle_center[1]])

                pf_calibration_steps = len(self.AR_tracking_data) - len(self.AR_PF_data)
                if i >= pf_calibration_steps:
                    gt_data_point = self.AR_PF_data[i - pf_calibration_steps].tolist()
                    gt.set_data([gt_data_point[0]], [gt_data_point[1]])
                    gt_path.set_data(*zip(*self.AR_PF_data[:i - pf_calibration_steps + 1]))

                trace.append(particle_center)
                dispersions.append(dispersion)
                avg_drifts.append(avg_drift)

                ax.relim()
                ax.autoscale_view()
                i += 1

            # Return artists for blitting
            return (*particles_lines, centroid, gt, gt_path) if not self.uniform_drift else (particles_lines[0], centroid, gt, gt_path)

        animation = FuncAnimation(figure, update, frames=frames, interval=animation_interval, repeat=False)
        plt.axis('equal')

        if self.gif_path is not None:
            print(f'Saving GIF to {self.gif_path}')
            writer = PillowWriter(fps=10)
            animation.save(self.gif_path, writer=writer)
        else:
            plt.show()

        plt.close(figure)

        dispersions = np.array(dispersions)
        avg_drifts = np.array(avg_drifts)

        distances = point_to_point_distances(trace, self.AR_PF_data)
        assert len(trace) == len(dispersions) == len(distances), \
            f'Error: Length of trace, dispersions, and distances must be equal. \n Got {len(trace)}, {len(dispersions)}, {len(distances)}'

        _ = self.calculate_metrics(trace, dispersions, distances)

        return {
            'trace': trace,
            'dispersions': dispersions,
            'distances': distances,
            'avg_drifts': avg_drifts,
            'convergence_idx': convergence_idx
        }
    

    def run_pf_without_visual(self, particle_filter):
        """Run particle filter simulation without visualization.
        
        Executes the particle filter tracking loop without displaying intermediate
        visualizations, which is faster for batch processing.
        
        Args:
            particle_filter: Initialized particle filter instance.
            
        Returns:
            dict: Dictionary containing:
                - 'trace': Array of estimated positions over time
                - 'dispersions': Array of particle dispersion values
                - 'distances': Array of distances to ground truth
                - 'avg_drifts': Array of average drift values
                - 'convergence_idx': Index where convergence occurred (if any)
        """
        trace = []
        dispersions = []
        avg_drifts = []

        dispersion_threshold = 3
        
        stage1_converged = False
        main_group = None
        stage2_converged = False
        convergence_idx = None

        mean_shift_counter = 0  # A counter used to track when to use mean shift
        mean_shift_interval = 5
        mean_shift_threshold = 0.5 # Consider convergence if the proportion of the largest cluster from meanshift is larger than this
        max_label = None

        for i in range(len(self.time_intervals)):
            particle_filter.process_noise_and_update(self.velocities[i], self.time_intervals[i])

            dispersion = particle_filter.get_dispersion()
            avg_drift = particle_filter.get_average_drift()
            
            # Check stage 1 convergence
            if self.uniform_drift:
                particle_counts = particle_filter.particle_num
                if not stage1_converged and dispersion <= dispersion_threshold:
                    stage1_converged = True
                    particle_filter.particle_num //= particle_filter.num_groups
            else:
                particle_counts = np.array([len(particle_filter.extract_points(group=i)) for i in range(particle_filter.num_groups)])
                # Check stage 1 convergence
                if not stage1_converged and any(particle_counts > sum(particle_counts) * 0.8):
                    stage1_converged = True
                    particle_filter.particle_num //= particle_filter.num_groups
                    main_group = np.argmax(particle_counts)
                    # print(f'Stage 1 @ {np.sum(self.time_intervals[:i])}')
                
                # Check if main group changed
                if stage1_converged:
                    temp_group = np.argmax(particle_counts)
                    if temp_group != main_group:
                        main_group = temp_group
                        stage2_converged = False
                        convergence_idx = None
                        max_label = None
                        mean_shift_counter = 0

            # Check stage 2 convergence
            if stage1_converged and not stage2_converged:
                if mean_shift_counter % mean_shift_interval == 0:
                    # Run meanshift 
                    if not self.uniform_drift:
                        main_group = np.argmax(particle_counts)
                    _max_label, _max_count, _max_proportion = particle_filter.mean_shift(group=main_group)
                    if _max_proportion >= mean_shift_threshold:
                        stage2_converged = True
                        convergence_idx = i
                        max_label = _max_label
                        mean_shift_counter = 0

                        # print(f'Stage 2 @ {np.sum(self.time_intervals[:i])}')

                mean_shift_counter += 1

            # After convergence, use a larger interval to update mean_shift
            if stage2_converged:
                if mean_shift_counter % 10 == 0:
                    # Run meanshift 
                    main_group = np.argmax(particle_counts)
                    _max_label, _max_count, _max_proportion = particle_filter.mean_shift(group=main_group)
                    max_label = _max_label

                mean_shift_counter += 1

            # Resample
            if stage1_converged:
                particle_filter.resample()
                max_cluster_particles = particle_filter.extract_points(group=main_group, label=max_label)

                if len(max_cluster_particles) <= 0:
                    _max_label, _max_count, _max_proportion = particle_filter.mean_shift(group=main_group)
                    max_label = _max_label
                    max_cluster_particles = particle_filter.extract_points(group=main_group, label=max_label)

                particle_center = particle_filter.mean_position(max_cluster_particles)
            else:
                adjust_weight = not self.uniform_drift
                particle_center = particle_filter.resample(adjust_weight=adjust_weight)  # Adjust weight for each group based on particle count
            
            if particle_center is None:
                # Particle filter failed, no active particles, terminate early
                break
                
            trace.append(particle_center)
            dispersions.append(dispersion)
            avg_drifts.append(avg_drift)

        trace = np.array(trace)
        dispersions = np.array(dispersions)
        avg_drifts = np.array(avg_drifts)

        distances = point_to_point_distances(trace, self.AR_PF_data)

        assert len(trace) == len(dispersions) == len(distances), f'Error: Length of trace, dispersions, and distances must be equal. \n Got {len(trace)}, {len(dispersions)}, {len(distances)}'

        _ = self.calculate_metrics(trace, dispersions, distances)

        return {'trace': trace, 'dispersions': dispersions, 'distances': distances, 'avg_drifts': avg_drifts, 'convergence_idx': convergence_idx}


    def calculate_metrics(self, trace, dispersions, distances):
        """Calculate localization performance metrics.
        
        Computes metrics including convergence time, distance, RMSE, and success rates
        based on the particle filter trace and ground truth data.
        
        Args:
            trace: Array of estimated positions from particle filter.
            dispersions: Array of particle dispersion values over time.
            distances: Array of distances between estimated and ground truth positions.
            
        Returns:
            dict: Dictionary containing computed metrics:
                - 'trace_length': Length of the trace
                - 'converge_time': Time to convergence (if converged)
                - 'converge_distance': Distance traveled before convergence
                - 'RMSE': Root mean square error over entire trace
                - 'RMSE_last_10': RMSE over last 10 steps
                - 'end_distance': Final distance to ground truth
                - 'success_1m': Whether last 10 steps all within 1m
                - 'conv_success': Whether convergence occurred
        """
        metrics = {
            'trace_length': len(trace),
            'converge_time': np.nan,
            'converge_distance': np.nan,
            'RMSE': np.nan,
            'RMSE_last_10': np.nan,
            'end_distance': np.nan,  # The distance in the end
            'success_1m': False,    # last 10 steps all fall within 1m
            'conv_success': False,    # Convergence success
        }

        dispersion_threshold = 3
        converge_index = None
        for idx, d in enumerate(dispersions):
            if d < dispersion_threshold:
                converge_index = idx
                metrics['conv_success'] = True
                break

        if converge_index is not None:
            # time is sum of intervals up to converge_index
            metrics['converge_time'] = float(np.sum(self.time_intervals[:converge_index]))

            # distance along GT path up to convergence (respect your 6-step offset)
            if converge_index > 6:
                seg1 = self.AR_PF_data[:converge_index - 1 - 6]
                seg2 = self.AR_PF_data[1:converge_index - 6]
                if len(seg1) and len(seg1) == len(seg2):
                    metrics['converge_distance'] = float(
                        np.sum(point_to_point_distances(seg1, seg2))
                    )

        # RMSE over finite distances, ignore first 6 
        tail = np.asarray(distances[6:], dtype=float)
        finite = np.isfinite(tail)
        if np.any(finite):
            metrics['RMSE'] = float(np.sqrt(np.mean(tail[finite] ** 2)))

        # Consider PF early termination as failure
        if len(trace) - 6 < len(self.AR_PF_data):
            metrics['conv_success'] = False

        # If the last 10 steps falls with 1m, consider success
        print('distances', distances[-10:])
        if np.all(distances[-10:] <= 1):
            metrics['success_1m'] = True
        
        # Record RMSE of the last 10 steps
        last10 = distances[-10:]
        finite = np.isfinite(last10)
        last10 = last10[finite]
        if np.any(last10):
            rmse_last10 = np.sqrt(np.mean(np.square(last10)))
            metrics['RMSE_last_10'] = rmse_last10

        metrics['end_distance'] = distances[-1]
        
        # record
        for k, v in metrics.items():
            self.metrics[k].append(v)
        self.n_simulations += 1
        return metrics

    def get_avg_metrics(self):
        """Calculate average metrics across all simulations.
        
        Aggregates metrics from multiple simulation runs and computes averages,
        filtering out invalid (inf/nan) values.
        
        Returns:
            dict: Dictionary with averaged metrics, prefixed with 'avg_':
                - 'avg_trace_length': Average trace length
                - 'avg_converge_time': Average convergence time
                - 'avg_converge_distance': Average convergence distance
                - 'avg_RMSE': Average RMSE
                - 'avg_RMSE_last_10': Average RMSE over last 10 steps
                - 'avg_end_distance': Average final distance
                - 'avg_success_1m': Average success rate (within 1m)
                - 'avg_conv_success': Average convergence success rate
        """
        avg_metrics = {}
        for k, vals in self.metrics.items():
            arr = np.asarray(vals, dtype=float)
            mask = np.isfinite(arr)  # drops inf and nan
            avg_metrics['avg_' + k] = float(np.mean(arr[mask])) if np.any(mask) else np.nan
        return avg_metrics
    