import numpy as np
from utils.geometry_utils import Bresenham, alpha_shape, get_segment_orientation_at_xy, get_segment_orientations, filter_segments_by_distance, get_intersection_point
from sklearn.cluster import MeanShift

class Particle:
    """Represents a single particle in the particle filter.
    
    Attributes:
        x: X coordinate in meters.
        y: Y coordinate in meters.
        drift: Orientation drift angle in radians.
        weight: Particle weight (probability).
        group: Group identifier for multi-group initialization.
        label: Cluster label from mean-shift clustering.
        step_length_scaler: Scaling factor for step length.
    """
    def __init__(self, x, y, drift, weight, group=None, label=None, step_length_scaler=1):
        """Initialize a particle.
        
        Args:
            x: Initial X coordinate.
            y: Initial Y coordinate.
            drift: Initial orientation drift.
            weight: Initial particle weight.
            group: Optional group identifier.
            label: Optional cluster label.
            step_length_scaler: Initial step length scaling factor.
        """
        self.x = x
        self.y = y
        self.drift = drift
        self.weight = weight
        self.group = group
        self.label = label
        self.step_length_scaler = step_length_scaler
    
    def update_using_velocity(self, mag, mag_noise, angle, drift_change, step_length_scaler_change, time_interval):
        """Update particle position using velocity with noise.
        
        Args:
            mag: Velocity magnitude.
            mag_noise: Noise added to magnitude.
            angle: Velocity direction angle.
            drift_change: Change in orientation drift.
            step_length_scaler_change: Change in step length scaler.
            time_interval: Time interval for motion update.
        """
        pre_x = self.x
        pre_y = self.y

        self.drift += drift_change
        self.step_length_scaler += step_length_scaler_change

        self.x = pre_x + (mag * time_interval * self.step_length_scaler + mag_noise) * np.cos(angle + self.drift)
        self.y = pre_y + (mag * time_interval * self.step_length_scaler + mag_noise) * np.sin(angle + self.drift)


class particle_filter:
    """Particle filter for 2D localization using vector maps and binary occupancy grids.
    
    Implements a particle filter for tracking position and orientation in indoor
    environments using floor plan maps and odometry data.
    """
    def __init__(self, vector_map, binary_map, map_mask=None, global_init=False, 
                 pf_init_method=True, start_point=None, heatmaps=None, drift_thetas=[0], 
                 uniform_drift=False, resolution=0.1, pf_config=None, group=None) -> None:
        
        if pf_config is None:
            self.particle_num = 500
            self.mag_sigma = 0.712 * 0.5
            self.angle_sigma = 0.01
            self.drift_sigma = 0.015
            self.resample_radius = 0.1
        else:
            self.particle_num = pf_config['particle_num']
            self.mag_sigma = pf_config['mag_sigma']
            self.angle_sigma = pf_config['angle_sigma']
            self.drift_sigma = pf_config['drift_sigma']
            self.resample_radius = pf_config['resample_radius']
            self.step_length_sigma = pf_config['step_length_sigma']


        self.vector_map = vector_map
        self.binary_map = binary_map
        self.heatmaps = heatmaps
        self.global_init = global_init
        self.start_point = start_point
        self.resolution = resolution
        self.map_mask = map_mask if map_mask is not None else alpha_shape(self.vector_map, visualize=False) 
        # Use map mask on heat map
        self.heatmaps = np.array([np.where(self.map_mask, heatmap, 0) for heatmap in heatmaps])
        
        self.pf_init_method = pf_init_method
        self.drift_thetas = drift_thetas     # Use this to set particle's direction, particle direction = velocity direction + drift_thetas + random noise
        self.uniform_drift = uniform_drift

        self.num_groups = len(self.heatmaps) if self.heatmaps is not None else 1

        self.particle_list = self.initialize_pf_global() if global_init else self.initialize_pf_local()

        self.BH = Bresenham(cell_size=resolution)

        self.update_count = 0   # Counts how many updates have been done


    def initialize_pf_global(self, use_mask=True):
        """Initialize particles globally using heatmaps or uniform sampling.
        
        Args:
            use_mask: If True, use map mask to constrain particle locations.
            
        Returns:
            list: List of Particle objects initialized globally.
        """
        assert self.heatmaps is not None, 'Error, you need a heat map to perform initialization'
        particle_list = []

        max_x = np.max(self.vector_map[:, :, 0])
        max_y = np.max(self.vector_map[:, :, 1])
        min_x = np.min(self.vector_map[:, :, 0])
        min_y = np.min(self.vector_map[:, :, 1])

        range_x = max_x - min_x
        range_y = max_y - min_y

        if self.pf_init_method == 'percentile':
            # Extract top 99 percentile positions
            percentile = 0.99
            flat_heatmaps = np.array([heatmap.flatten() for heatmap in self.heatmaps]).flatten()
            kth = int(flat_heatmaps.shape[0] * percentile)
            top_val_indices = np.argpartition(flat_heatmaps, kth=kth)[kth:]
            top_val_indices = np.unravel_index(top_val_indices, self.heatmaps.shape)
            binary_heatmaps = np.zeros_like(self.heatmaps)
            binary_heatmaps[top_val_indices] = 1

            random_with_replacement = self.particle_num > np.count_nonzero(binary_heatmaps) 
            candidate_locations = np.array(np.where(binary_heatmaps == 1)).T

            selected_indices = np.random.choice(len(candidate_locations), size=self.particle_num, replace=random_with_replacement)
            selected_locations = candidate_locations[selected_indices]

            for group, y, x in selected_locations:
                x = x * self.resolution
                y = y * self.resolution
                if self.uniform_drift:
                    drift = np.random.random() * 2 * np.pi - np.pi       # drift = -pi ~ pi
                else:
                    # drift = self.drift_thetas[group] + np.random.random() * np.pi / 4 - np.pi / 8   # drift = -pi/8 ~ pi/8
                    drift = self.drift_thetas[group]

                particle_list.append(Particle(x=x, y=y, drift=drift, weight=1, group=group))


        elif self.pf_init_method == 'random':
            # Flatten all heatmaps into one array
            flat_heatmaps = self.heatmaps.flatten().astype(np.float64)

            # Normalize to probabilities
            if flat_heatmaps.sum() == 0:
                probs = np.ones_like(flat_heatmaps) / flat_heatmaps.size
            else:
                probs = flat_heatmaps / flat_heatmaps.sum()

            # Draw particle_num samples from the full set of (group, y, x) indices
            indices = np.arange(flat_heatmaps.size)
            selected_flat = np.random.choice(
                indices, size=self.particle_num, replace=True, p=probs
            )

            # Convert back to (group, y, x)
            selected_locations = np.array(np.unravel_index(selected_flat, self.heatmaps.shape)).T

            # Create particles
            for group, y, x in selected_locations:
                x = x * self.resolution
                y = y * self.resolution
                if self.uniform_drift:
                    drift = np.random.random() * 2 * np.pi - np.pi   # drift ∈ [-π, π]
                else:
                    drift = self.drift_thetas[group]

                particle_list.append(Particle(x=x, y=y, drift=drift, weight=1, group=group))
    
        else:
            raise NotImplementedError

        return particle_list


    def initialize_pf_local(self):
        """Initialize particles locally around a starting point.
        
        Requires start_point to be set. Particles are initialized with random
        drift angles around the starting location.
        
        Returns:
            list: List of Particle objects initialized locally.
        """
        assert self.start_point is not None, 'Error, you need a start point to initialize PF locally'

        particle_list = []
        for _ in range(self.particle_num):
            # drift = np.random.normal(loc=0, scale=0.5)
            drift = np.random.random() * np.pi / 2 - np.pi / 4
            particle_list.append(Particle(x=self.start_point[0], y=self.start_point[1], drift=drift, weight=1/self.particle_num))

        return particle_list
    

    def process_noise_and_update(self, velocity, time_interval, wall_strategy='weighted'):
        '''
            updates the particle filter using input velocity
            
            param:
                velocity: velocity represented in the form of [x, y]
                time_interval: used to calculate displacement
        '''
        assert wall_strategy in ['no_pass', 'weighted', 'scheduled', 'bounce'], f"Error: Wall strategy must be one of ['no_pass', 'weigthed', 'scheduled', 'bounce'], get {wall_strategy} instead"
        if np.all(velocity == 0):
             return
        
        mag = np.linalg.norm(velocity)
        angle = np.arctan2(velocity[1], velocity[0])

        for index in range(len(self.particle_list)):
            self.process_noise_and_update_single_particle(index, mag, angle, time_interval, wall_strategy)
            # Check if particle falls with in room (not implemented yet)
        
        self.update_count += 1


    def process_noise_and_update_single_particle(self, index, mag, angle, time_interval, wall_strategy='weighted'):
        assert wall_strategy in ['no_pass', 'weighted', 'scheduled', 'bounce'], f"Error: Wall strategy must be one of ['no_pass', 'weigthed', 'scheduled', 'bounce'], get {wall_strategy} instead"

        particle = self.particle_list[index]
        pre_x = particle.x
        pre_y = particle.y
        particle.update_using_velocity(mag=mag, 
                                        mag_noise=np.random.normal(loc=0, scale=self.mag_sigma), 
                                        angle=np.random.normal(loc=angle, scale=self.angle_sigma), 
                                        drift_change=np.random.normal(loc=0, scale=self.drift_sigma), 
                                        step_length_scaler_change=np.random.normal(loc=0, scale=self.step_length_sigma) if self.step_length_sigma is not None else 0,
                                        time_interval=time_interval)
        new_x = particle.x
        new_y = particle.y

        line = np.array([[pre_x, pre_y], [new_x, new_y]])
        
        # Check if particle is out of map
        if self.out_of_map_bound(line):
            self.particle_list[index].x = pre_x
            self.particle_list[index].y = pre_y
            self.particle_list[index].weight = 0
            return
        
        # Check if particle intersects with wall
        intersect, intersection_pt = self.intersect_with_map(line)
        if intersect:
            if wall_strategy == 'no_pass':
                self.particle_list[index].x = pre_x
                self.particle_list[index].y = pre_y
                self.particle_list[index].weight = 0
            elif wall_strategy == 'weighted':
                self.particle_list[index].x = new_x
                self.particle_list[index].y = new_y
                self.particle_list[index].weight *= 0.3
            elif wall_strategy == 'scheduled':
                self.particle_list[index].x = new_x
                self.particle_list[index].y = new_y
                if self.update_count < 40:
                    self.particle_list[index].weight *= 0.9
                else:
                    self.particle_list[index].weight *= 0
            elif wall_strategy == 'bounce':
                # Calculate angle
                # visualize(segments_1=filter_segments_by_distance(self.vector_map, point=intersection_pt), point_1=intersection_pt, points_2=np.array([[pre_x, pre_y], [new_x, new_y]]))
                seg_ori = get_segment_orientation_at_xy(segments=self.vector_map, point=intersection_pt)
                if seg_ori is None:
                    # Note: Edge case handling when particle intersects with 0 or multiple segments.
                    # Current behavior allows the particle to pass through, which may cause
                    # particles to move through walls in complex geometries. Consider adding
                    # more robust collision detection or rejection in such cases.
                    # If intersect with 0 or more than 1 segments, we just let it pass
                    self.particle_list[index].x = new_x
                    self.particle_list[index].y = new_y
                else:
                    path_ori = get_segment_orientations([line])
                    theta = seg_ori - path_ori

                    if np.abs(theta) > np.radians(90):
                        theta = (np.pi - np.abs(theta)) * -np.sign(theta)

                    if np.abs(theta) > np.radians(60):
                        # if the incident angle is too large, remove the particle
                        self.particle_list[index].x = pre_x
                        self.particle_list[index].y = pre_y
                        self.particle_list[index].weight = 0
                    else:
                        # update particle drift
                        self.particle_list[index].drift += float(theta)
                        # update particle location
                        self.process_noise_and_update_single_particle(index, mag, angle, time_interval, wall_strategy)


    def out_of_map_bound(self, line):
        """Check if a line segment is outside the map boundaries.
        
        Args:
            line: Line segment as array [[x1, y1], [x2, y2]].
            
        Returns:
            bool: True if any point is outside map bounds or mask.
        """
        x_cell = self.binary_map.shape[1]
        y_cell = self.binary_map.shape[0]
        # Check if a point is outside of the tight map bounds created using concave hull
        for point in line:
            x = point[0] / self.resolution
            y = point[1] / self.resolution
            if x < 0 or x >= x_cell or y < 0 or y >= y_cell:
                return True
            if self.map_mask[int(y), int(x)] == 0:
                return True
        return False


    def intersect_with_map(self, line, mode='raster'):
        assert mode in ['raster', 'vector'], "Error: mode should be either 'raster' or 'vector'"
        intersection_point = None
        if mode == 'raster':
            x_res_list, y_res_list = self.BH.seg(line[0][0], line[0][1], line[1][0], line[1][1])
            for x, y in zip(map(int, x_res_list), map(int, y_res_list)):
                # See if any of the traces intersects with the map wall
                # Notice that in Python, y (row) needs to be first in numpy arrays!
                if self.binary_map[y][x] == 1:
                    intersection_point = np.array([x * self.resolution, y * self.resolution])
                    return True, intersection_point
        if mode == 'vector':
            # Note: Current implementation uses brute-force search which can be slow for large maps.
            # Consider implementing Approximate Nearest Neighbor Field (ANNF) or spatial indexing
            # (e.g., KD-tree) to optimize segment intersection queries for better performance.
            mid_point = np.array([(max(line[0][0], line[1][0]) - min(line[0][0], line[1][0])) / 2 + min(line[0][0], line[1][0]),
                                  (max(line[0][1], line[1][1]) - min(line[0][1], line[1][1])) / 2 +  min(line[0][1], line[1][1])])
            vector_mag = np.linalg.norm(line[1] - line[0])
            segments = filter_segments_by_distance(self.vector_map, point=mid_point, range_limit=vector_mag)
            if len(segments) > 0:
                for segment in segments:
                    intersection_point = get_intersection_point(segment, line)
                    if intersection_point is not None:
                        return True, intersection_point


        return False, intersection_point


    def resample(self, adjust_weight=False):
        resampled_list = []
        active_particles = []

        # Determine number of groups dynamically (fallback to 1 if unknown)
        if hasattr(self, "num_groups") and isinstance(self.num_groups, int) and self.num_groups > 0:
            num_groups = self.num_groups
        else:
            # infer from existing particles
            groups = [p.group for p in self.particle_list if getattr(p, "group", None) is not None]
            num_groups = max(groups) + 1 if groups else 1

        particle_counts = np.zeros(num_groups, dtype=int)

        # Collect active particles and count per group
        for particle in self.particle_list:
            if particle.weight > 0:
                active_particles.append(particle)
                g = getattr(particle, "group", None)
                if g is not None and 0 <= g < num_groups:
                    particle_counts[g] += 1

        if len(active_particles) == 0:
            # Particle filter failure - no more active particles
            return None

        # Optionally adjust weights based on group counts (smoothed by gamma)
        if adjust_weight:
            gamma = 0.001
            total_active = float(len(active_particles))
            for p in active_particles:
                g = getattr(p, "group", None)
                if g is not None and 0 <= g < num_groups and total_active > 0:
                    new_weight = p.weight * (particle_counts[g] / total_active)
                    p.weight = p.weight * (1.0 - gamma) + new_weight * gamma
                # else: leave weight unchanged (e.g., group None or out of range)

        self.normalize_weights(active_particles)

        pre_sampled = self.low_variance_sampler(active_particles, self.particle_num)
        for p in pre_sampled:
            dx = np.random.normal(loc=0.0, scale=self.resample_radius)
            dy = np.random.normal(loc=0.0, scale=self.resample_radius)
            new_x = p.x + dx
            new_y = p.y + dy

            # If you want to avoid crossing walls, enable this block and your map intersection check
            # line = np.array([[p.x, p.y], [new_x, new_y]])
            # intersect, _ = self.intersect_with_map(line, mode='raster')
            # if intersect:
            #     new_x, new_y = p.x, p.y

            resampled_list.append(Particle(
                x=new_x,
                y=new_y,
                drift=p.drift,
                weight=p.weight,
                group=p.group,
                label=p.label,
                step_length_scaler=p.step_length_scaler
            ))

        assert len(resampled_list) == self.particle_num, "Incorrect resampled list length"
        self.particle_list = resampled_list

        # Return the current mean position of all (resampled) particles
        return self.mean_position(self.extract_points())
    

    def low_variance_sampler(self, input_particle_list, M):
        '''
        Resamples particles based on their weights using a low variance method.
        
        Args:
            input_particle_list: List of particles with weights.
            M: Number of particles to sample.
        
        Returns:
            List of resampled particles.
        '''
        result_particle_list = []
        r = np.random.uniform(0.0, 1.0 / M)
        c = input_particle_list[0].weight
        i = 0
        
        for m in range(M):
            U = r + m / M
            while U > c:
                i += 1
                c += input_particle_list[i].weight
            result_particle_list.append(input_particle_list[i])
    
        return result_particle_list
    

    def mean_shift(self, group=None):
        '''
        Runs meanshift on a group of particles, assign labels to these groups
        return:
            max_label: the label for the largest cluster
            max_count: the number of particles in the largest cluster
            max_proportion: the propoertion of the largest cluster in the group
        '''
        MS = MeanShift()
        particles, indices = self.extract_points(group=group, return_indices=True)
        MS.fit(particles)
        labels = MS.labels_
        assert len(indices) == len(labels)

        cluster_counts = {}
        for i, label in enumerate(labels):
            # Assign labels
            self.particle_list[indices[i]].label = label

            if label in cluster_counts:
                cluster_counts[label] += 1
            else:
                cluster_counts[label] = 1
        max_label = max(cluster_counts, key=cluster_counts.get)   
        max_count = cluster_counts[max_label]   
        max_proportion = max_count / len(labels)      

        return max_label, max_count, max_proportion


    def normalize_weights(self, particle_list):
        """Normalize particle weights so they sum to 1.
        
        Args:
            particle_list: List of Particle objects to normalize (modified in-place).
        """
        total_weight = sum([particle.weight for particle in particle_list])
        for particle in particle_list:
            particle.weight /= total_weight


    def extract_points(self, particle_list=None, group=None, label=None, return_indices=False):
        """Extract 2D positions from particles, optionally filtered by group or label.
        
        Args:
            particle_list: List of particles to extract from. If None, uses self.particle_list.
            group: If provided, only extract particles from this group.
            label: If provided, only extract particles with this label.
            return_indices: If True, also return indices of extracted particles.
            
        Returns:
            np.ndarray or tuple: Array of points shape (n, 2), or (points, indices) if return_indices=True.
        """
        if particle_list is None:
            particle_list = self.particle_list

        points = []
        indices = []
        for i, particle in enumerate(particle_list):
            if group is not None and particle.group != group:
                continue
            if label is not None and particle.label != label:
                continue

            points.append([particle.x, particle.y])
            indices.append(i)

        if return_indices:
            return np.array(points), np.array(indices)

        return np.array(points)
    

    def extract_weights(self, particle_list=None, group=None):
        """Extract weights from particles, optionally filtered by group.
        
        Args:
            particle_list: List of particles to extract from. If None, uses self.particle_list.
            group: If provided, only extract weights from this group.
            
        Returns:
            np.ndarray: Array of particle weights.
        """
        if particle_list is None:
            particle_list = self.particle_list
        weights = []
        for particle in particle_list:
            if group is not None and particle.group != group:
                continue
            weights.append(particle.weight)

        return np.array(weights)


    def mean_position(self, points):
        """Compute the mean position from a set of 2D points.
        
        Args:
            points: Array of 2D points, shape (n, 2).
            
        Returns:
            list or None: Mean position [x, y], or None if points is empty.
        """
        if len(points) == 0:
            return None
        mean_x = np.mean([point[0] for point in points])
        mean_y = np.mean([point[1] for point in points])

        return [mean_x, mean_y]
    

    def get_dispersion(self):
        """
        Weighted spatial dispersion of particles (weighted std of distances to weighted centroid).
        Robust to empty sets and zero/invalid weights.
        """
        pts = self.extract_points()
        if pts.size == 0:
            return np.inf  # no particles -> maximal dispersion

        w = self.extract_weights()
        # Align lengths
        if w.size != len(pts):
            w = np.ones(len(pts), dtype=float)

        # Keep only finite points and positive weights
        pts = np.asarray(pts, dtype=float)
        w = np.asarray(w, dtype=float)
        mask = np.isfinite(pts).all(axis=1) & np.isfinite(w) & (w > 0)
        if not np.any(mask):
            # no valid weighted points; treat as unweighted over finite points (if any)
            mask = np.isfinite(pts).all(axis=1)
            if not np.any(mask):
                return np.inf
            pts_valid = pts[mask]
            center = pts_valid.mean(axis=0)
            d = np.linalg.norm(pts_valid - center, axis=1)
            return float(d.std())

        pts_valid = pts[mask]
        w_valid = w[mask]

        sw = w_valid.sum()
        if not np.isfinite(sw) or sw <= 0:
            # fallback to uniform weights
            w_valid = np.ones(len(pts_valid), dtype=float) / len(pts_valid)
        else:
            w_valid = w_valid / sw

        # Weighted centroid
        center = np.average(pts_valid, axis=0, weights=w_valid)
        d = np.linalg.norm(pts_valid - center, axis=1)

        # Weighted std of distances
        mu = np.average(d, weights=w_valid)
        var = np.average((d - mu) ** 2, weights=w_valid)
        return float(np.sqrt(var))
    

    def get_average_drift(self):
        """Compute the average orientation drift across all particles.
        
        Returns:
            float: Average drift angle in radians.
        """
        return np.mean([particle.drift for particle in self.particle_list])