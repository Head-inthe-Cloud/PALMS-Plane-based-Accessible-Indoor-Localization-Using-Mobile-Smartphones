import math
import numpy as np
from utils.file_io import *
from utils.visualization import *
from utils.geometry_utils import normalize_heatmap
from matplotlib.path import Path
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter, shift
from sklearn.cluster import KMeans, DBSCAN, MeanShift
import cv2

class ConvCES():
    """
    Convolutional Certainly Empty Space (CES) for layout matching.
    
    This class implements the CES constraint for matching observed plane segments
    to floor plan layouts. It generates heatmaps that indicate likely locations
    where the observation could have been taken, based on the constraint that
    observed empty spaces must correspond to empty spaces in the floor plan.
    
    The CES kernel is created by convolving the observed segments with the floor
    plan, where high values indicate good matches. Gaussian weighting can be
    applied to emphasize matches near the observation point.
    
    Attributes:
        obs_segments (np.ndarray): Observed plane segments, shape (n, 2, 2).
        resolution (float): Spatial resolution in meters per pixel.
        mode (str): 'binary' or 'weighted' kernel mode.
        CES_kernel (np.ndarray): Certainly Empty Space kernel.
        seg_kernel (np.ndarray): Segment kernel.
        obs_point (np.ndarray): Observation point [x, y] in meters.
        gaussian_kernel_config (tuple): (kernel_size, sigma) for Gaussian weighting.
        
    Example:
        >>> obs_planes = np.array([[[0, 0], [1, 0]], [[1, 0], [1, 1]]])
        >>> ces = ConvCES(obs_planes, resolution=0.1, mode='weighted')
        >>> heatmap = ces.create_heatmap(binary_fp=floor_plan_map)
    """
    def __init__(self, obs_segments=None, resolution=0.1, T=None, mode='weighted', visualize_kernels=False, gaussian_kernel_config=(7, 3)):
        """
        Initialize the ConvCES object.

        Args:
            obs_segments (np.ndarray, optional): Observed plane segments.
                Shape (n, 2, 2) where each segment is [start_point, end_point].
                If None, kernels will not be initialized.
            resolution (float, optional): Spatial resolution in meters per pixel.
                Default is 0.1.
            T (np.ndarray, optional): 3x3 transformation matrix applied to obs_segments.
                Used to calculate the observation point. If None, observation point
                is assumed to be at origin (0, 0).
            mode (str, optional): Kernel mode - 'binary' or 'weighted'.
                'binary' uses binary segment representation.
                'weighted' applies distance-based weighting.
                Default is 'weighted'.
            visualize_kernels (bool, optional): If True, visualize the generated kernels.
                Default is False.
            gaussian_kernel_config (tuple, optional): (kernel_size, sigma) for Gaussian
                weighting. Default is (7, 3).
        """
        assert mode in ['binary', 'weighted'], 'mode must be either binary or weighted'
        self.mode = mode
        self.obs_segments = obs_segments
        self.kernel_shape = None
        self.CES_kernel = None
        self.seg_kernel = None
        self.obs_point = None
        self.resolution = resolution
        self.gaussian_kernel_config = gaussian_kernel_config

        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None

        
        if T is None:
            self.obs_point = np.array([0, 0])
        else:
            self.obs_point = np.array(np.matmul(T, np.array([0, 0, 1]).T))

            # Reduce dimension to go from homogeneous space back to euclidean
            if len(self.obs_point.shape) == 2:
                self.obs_point = self.obs_point[0, :2]
            else:
                self.obs_point = self.obs_point[:2]

        if obs_segments is None:
            return
        self.initialize_kernels(obs_segments, mode)
        
        if visualize_kernels:
            self.visualize_kernels()

    
    def apply_gaussian_weighting(self, kernel, sigma):
        """Create a Gaussian weight map and multiply it with the kernel.
        
        Creates a Gaussian weight map centered at (-min_x, -min_y) with the same
        shape as kernel and multiplies the kernel with this weight map.

        Args:
            kernel: The input CES kernel.
            sigma: Standard deviation in X and Y direction for the Gaussian.

        Returns:
            np.ndarray: The CES kernel weighted by the Gaussian.
        """

        # Get kernel shape
        h, w = kernel.shape

        # Create meshgrid
        x = np.linspace(0, w - 1, w) - (-self.min_x / self.resolution)  # Shift center to -min_x
        y = np.linspace(0, h - 1, h) - (-self.min_y / self.resolution)  # Shift center to -min_y
        X, Y = np.meshgrid(x, y)

        # Compute 2D Gaussian
        gaussian_weight = np.exp(-((X**2) / (2 * sigma**2) + (Y**2) / (2 * sigma**2)))

        # Normalize the Gaussian to max 1 (optional)
        gaussian_weight /= np.max(gaussian_weight)

        # Apply Gaussian weighting to kernel
        weighted_CES = kernel * gaussian_weight

        return weighted_CES
    
    def apply_square_mask(self, image, center, max_distance):
        """Apply a square mask to an image centered at a given point.

        Args:
            image: Input image (2D or 3D).
            center: (x, y) coordinates of the mask center.
            max_distance: Half the length of the square side.

        Returns:
            np.ndarray: Masked image with values outside the square set to 0.
        """
        h, w = image.shape[:2]
        cx, cy = center

        max_distance = int(max_distance)
        # Compute bounds
        x1 = max(cx - max_distance, 0)
        x2 = min(cx + max_distance, w)
        y1 = max(cy - max_distance, 0)
        y2 = min(cy + max_distance, h)

        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1

        # Apply mask
        if image.ndim == 3:
            mask = mask[:, :, np.newaxis]  # For color images

        return image * mask


    def erode_binary_mask(self, binary_img, radius_px=1):
        """Shrink the binary mask inward using morphological erosion.

        Args:
            binary_img: Binary image (uint8 or bool).
            radius_px: Erosion radius in pixels.

        Returns:
            np.ndarray: Eroded binary image.
        """
        # Convert to uint8 if needed
        if binary_img.dtype != np.uint8:
            binary_img = binary_img.astype(np.uint8)

        if radius_px < 1:
            return binary_img.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
        eroded = cv2.erode(binary_img, kernel, iterations=1)
        return eroded
    
    def initialize_kernels(self, obs_segments, mode='weighted'):
        """Initialize CES and segment kernels from observed plane segments.
        
        Creates kernels representing certainly empty space and observed segments,
        with optional Gaussian weighting and distance filtering.
        
        Args:
            obs_segments: Array of observed plane segments, shape (n, 2, 2).
            mode: Kernel mode: 'binary' or 'weighted'. Defaults to 'weighted'.
        """

        triangles = np.array([[self.obs_point, obs_segments[i, 0], obs_segments[i, 1]] for i in range(obs_segments.shape[0])])

        # Determine the bounds of the map
        self.min_x = triangles[:, :, 0].min()
        self.max_x = triangles[:, :, 0].max()
        self.min_y = triangles[:, :, 1].min()
        self.max_y = triangles[:, :, 1].max()

        # Create the grid
        x = np.arange(self.min_x, self.max_x + self.resolution, self.resolution)  # Keep the last point even if it can not be devided evenly
        y = np.arange(self.min_y, self.max_y + self.resolution, self.resolution)

        # Make the kernel with odd rows and columns
        if len(x) % 2 == 0:
            x = np.append(x, x[-1] + self.resolution)

        if len(y) % 2 == 0:
            y = np.append(y, y[-1] + self.resolution)

        xx, yy = np.meshgrid(x, y)
        self.kernel_shape = xx.shape
            
        points = np.vstack((xx.flatten(), yy.flatten())).T

        # Initialize the binary map
        self.CES_kernel = np.zeros(points.shape[0], dtype=int)

        # Mark pixels inside any triangle
        for triangle in triangles:
            path = Path(triangle)
            mask = path.contains_points(points)
            self.CES_kernel = np.logical_or(self.CES_kernel, mask)  # CES_kernel dtype = bool

        self.CES_kernel = self.CES_kernel.reshape(self.kernel_shape)

        # For seg_kernel
        self.seg_kernel = np.zeros(points.shape[0], dtype=int)
        # Initialize Bresenham with the cell size
        cell_size = self.resolution  # Example cell size
        bresenham = Bresenham(cell_size)
        self.seg_kernel = self.seg_kernel.reshape(self.kernel_shape)
        
        # Mark pixels on the path for each segment
        for segment in obs_segments:
            x0, y0 = segment[0]
            x1, y1 = segment[1]
            x_res_list, y_res_list = bresenham.seg(x0 - self.min_x, y0 - self.min_y, x1-self.min_x, y1-self.min_y)
            
            for x, y in zip(x_res_list, y_res_list):
                if 0 <= x < self.kernel_shape[1] and 0 <= y < self.kernel_shape[0]:
                    self.seg_kernel[int(y), int(x)] = 1

        self.CES_kernel = np.array(self.CES_kernel, dtype=float)
        self.seg_kernel = np.array(self.seg_kernel, dtype=float)

        # Add paddings
        padding_width = 10
        self.seg_kernel = np.pad(self.seg_kernel, pad_width=padding_width, mode='constant', constant_values=0)

        # Shrink factor
        shrink_factor = 0.7

        # Calculate the new size of the kernel
        old_size = self.CES_kernel.shape
        new_size = np.int16(np.array(old_size) * shrink_factor)
        shrunk_kernel = cv2.resize(self.CES_kernel, new_size[::-1], interpolation=cv2.INTER_NEAREST)

        # Calculate the relative position of (x, y) in the shrunk kernel
        x = int((self.obs_point[0] - self.min_x) / self.resolution) 
        y = int((self.obs_point[1] - self.min_y) / self.resolution)

        # Calculate the relative position of (x, y) in the shrunk kernel
        new_x = int(x * shrink_factor)
        new_y = int(y * shrink_factor)

        # Calculate the padding to keep (x, y) centered
        padding_top = y - new_y
        padding_bottom = old_size[0] - new_size[0] - padding_top
        padding_left = x - new_x
        padding_right = old_size[1] - new_size[1] - padding_left

        # Pad the shrunk kernel back to original size
        padded_kernel = np.pad(shrunk_kernel, 
                            ((padding_top, padding_bottom), (padding_left, padding_right)), 
                            mode='constant', constant_values=0)
        
        # Debugging only
        if False:
            fig, axs = plt.subplots(3)
            axs[0].imshow(self.CES_kernel, cmap='Greys', origin='lower')
            axs[0].scatter(x, y)
            axs[1].imshow(shrunk_kernel, cmap='Greys', origin='lower')
            axs[1].scatter(x, y)
            axs[2].imshow(padded_kernel, cmap='Greys', origin='lower')
            axs[2].scatter(x, y)
            plt.show()
            sys.exit()
        
        # Pad the kernel to be same size as seg_kernel
        self.CES_kernel = np.pad(padded_kernel, pad_width=padding_width, mode='constant', constant_values=0)
        self.min_x -= padding_width * self.resolution
        self.min_y -= padding_width * self.resolution

        if mode=='weighted':
            # Define the size of the Gaussian kernel
            kernel_size = (self.gaussian_kernel_config[0], self.gaussian_kernel_config[0])  # Kernel size. The values should be odd numbers.

            # Define the standard deviation in X and Y direction
            sigma_x = self.gaussian_kernel_config[1] # Standard deviation in X direction
            sigma_y = self.gaussian_kernel_config[1] # Standard deviation in Y direction. If 0, it's set to be equal to sigma_x.

            # Apply Gaussian filter to blur the seg_kernel
            self.seg_kernel = cv2.GaussianBlur(self.seg_kernel, kernel_size, sigmaX=sigma_x, sigmaY=sigma_y)
            # Use Gaussian weighting to weight segments closer to the observation point more
            self.seg_kernel = self.apply_gaussian_weighting(self.seg_kernel, sigma=10 / self.resolution)

            # Filter by distance
            self.CES_kernel = self.apply_square_mask(self.CES_kernel, center=(x, y), max_distance=10 / self.resolution)

            # Erode the kernel 
            self.CES_kernel = self.erode_binary_mask(self.CES_kernel, radius_px=1)

    def rotate(self, theta):
        """Apply rotation to the CES kernel and reinitialize.

        Args:
            theta: Angle to rotate the kernel by, in radians.
        """
        T = np.eye(3)
        R, _ = get_R_from_orientations(theta=theta)
        T[:2, :2] = R
        self.obs_segments = apply_transformation_to_segments(self.obs_segments, T)
        self.initialize_kernels(self.obs_segments, mode=self.mode)


    def conv2d(self, image, kernel):
        """Perform optimized 2D convolution using OpenCV.
        
        Args:
            image: 2D array of shape (h, w).
            kernel: 2D array of shape (n, m).
        
        Returns:
            np.ndarray: 2D array of shape (h, w) with convolution result.
        """
        # Flip the kernel for convolution, as OpenCV applies correlation (not convolution)
        # kernel_flipped = np.flipud(np.fliplr(kernel))
        
        # Perform convolution using OpenCV's filter2D function
        result = cv2.filter2D(np.float32(image), -1, kernel)
        assert result.shape == image.shape, 'The shape of the output should be the same as the shape of input'
        
        return result

    def create_heatmap(self, binary_fp, kernel='CES', visualize_heatmap=False, filler='min', alpha=10):
        """Create a heatmap by performing convolution over the floor plan.

        Args:
            binary_fp: Binary map representing the floor plan, shape (n, m).
            kernel: Kernel type to use: 'CES', 'seg', or 'com'. Defaults to 'CES'.
            visualize_heatmap: If True, visualize the resulting heatmap.
            filler: Filler value for heatmap: 'min' or 'max'. Defaults to 'min'.
            alpha: Weight for CES kernel in combined mode. Defaults to 10.

        Returns:
            np.ndarray: Heatmap of size (n, m) representing matching scores.
        """

        kernel_types = ['CES', 'seg', 'com']
        if kernel not in kernel_types:
            raise ValueError('Invalid kernel type. Expected one of: %s' % kernel_types)

        # Calculate margin sizes
        # This step introduces rasterization error due to the floor operation, so the output shape may not match the original shape
        top_margin_height = int((self.max_y - self.obs_point[1]) / self.resolution)
        bot_margin_height = int(np.abs(self.min_y - self.obs_point[1]) / self.resolution)
        right_margin_width = int((self.max_x - self.obs_point[0]) / self.resolution)
        left_margin_width = int(np.abs(self.min_x - self.obs_point[0]) / self.resolution)

        
        # Fill in the center
        if kernel == 'CES':
            heatmap = self.conv2d(binary_fp, self.CES_kernel)
        elif kernel == 'seg':
            # heatmap = self.conv2d(1-binary_fp, self.seg_kernel)
            heatmap = self.conv2d(binary_fp, self.seg_kernel)
        elif kernel == 'com':
            combined_kernel = -alpha * self.CES_kernel + normalize_heatmap(self.seg_kernel, as_prob_distribution=True)
            heatmap = self.conv2d(binary_fp, combined_kernel)
            heatmap = np.clip(heatmap, 0, None)

        if filler == 'max':
            filler_val = np.max(heatmap)
        else:
            filler_val = np.min(heatmap)

        kernel_height, kernel_width = self.kernel_shape

        # The heatmap is calculated regarding the center of the kernel
        # We want it to work with observation point instead, so we shift the heatmap

        # Compute center offset between kernel and observation point
        dy = bot_margin_height - kernel_height // 2
        dx = left_margin_width - kernel_width // 2

        # Manually shifting the heatmap to fix rasterization errors
        dx += -9
        dy += -9

        shifted_heatmap = shift(heatmap, shift=(dy, dx), order=1, mode='constant', cval=0.0)

        return shifted_heatmap


    def find_elbow_point(self, inertias, k_range):
        """Find the elbow point in a K-means inertia curve.
        
        Uses second derivative to identify the optimal number of clusters.
        
        Args:
            inertias: List of inertia values for different k values.
            k_range: Range of k values corresponding to inertias.
            
        Returns:
            int: Optimal k value at the elbow point.
        """
        # Calculate the differences between successive inertias
        deltas = np.diff(inertias)
        # Then, calculate the difference between successive deltas
        delta_deltas = np.diff(deltas)
        # The elbow point is where the second derivative (change in deltas) is maximum
        elbow_point_index = np.argmax(delta_deltas) + 1  # +1 due to the nature of diff reducing array length by 1
        return k_range[elbow_point_index]
    

    def find_kth_min_coords(self, heatmap, k, neighborhood_size=10, resolution=1):
        """Find the coordinates of the k minimum values in the heatmap.
        
        Filters out nearby points to avoid duplicates using neighborhood size threshold.
        
        Args:
            heatmap: Input heatmap array.
            k: Number of minimum points to return.
            neighborhood_size: Neighborhood size for deduplication in meters. Defaults to 10.
            resolution: Resolution of the heatmap in meters per pixel. Defaults to 1.
            
        Returns:
            np.ndarray: Array of k coordinates with minimum heatmap values.
        """

        k_multiplier = 10  # We multiply this number with k when performing argpartition, 
                           # since we eliminate overlaping coordinates, this makes sure we have k coordinates left
        assert k > 0, 'k has to be greater than 0'
        # Step 1: Flatten the heatmap
        flattened = heatmap.flatten()
        
        # Step 2: Find the indices of the kth smallest values
        if k <= len(flattened):
            # Using partition for efficiency. Note: this modifies the original array.
            # If preserving the original array is necessary, consider using np.argpartition on a copy.
            candidate_indices = np.argpartition(flattened, kth=k_multiplier*k-1)[:k_multiplier*k]  # find the top 2*k indices
        else:
            raise ValueError('k is larger than the number of elements in the heatmap')
        
        # Step 3: Convert 1D indices to 2D coordinates
        coords = np.unravel_index(candidate_indices, heatmap.shape)        
        # coords now contains two arrays: one for row indices and one for column indices.
        # Zip these arrays to get tuples of (row, col) coordinates.
        coords = np.dstack((coords[1], coords[0]))[0]

        # Step 4: filter out points that are close to each other
        kth_coords = [coords[0]]
        num_coords = 1
        for coord in coords[1:]:
            in_neighborhood = False
            for ref_coord in kth_coords:
                if point_to_point_distance(coord, ref_coord) < neighborhood_size / resolution:
                    in_neighborhood = True
                    break
            if not in_neighborhood:
                kth_coords.append(coord)
                num_coords += 1
            if num_coords >= k:
                break
        
        return np.array(kth_coords)

    def find_local_minima(self, heatmap, visualize_minima=False, neighborhood_size=10000):
        """Find local minima in the heatmap using minimum filter.
        
        Args:
            heatmap: Input heatmap array.
            visualize_minima: If True, visualize detected minima. Defaults to False.
            neighborhood_size: Size of neighborhood for minimum filter. Defaults to 10000.
            
        Returns:
            np.ndarray: Array of local minima coordinates, shape (n, 2).
        """
        # Apply minimum filter
        min_filtered = minimum_filter(heatmap, size=neighborhood_size)

        # Find local minima
        local_min_mask = (heatmap == min_filtered)
        y, x = np.where(local_min_mask)

        # Visualization
        if visualize_minima:
            plt.imshow(heatmap, cmap='viridis')
            plt.colorbar()
            plt.scatter(x, y, color='r')
            plt.title('Local Minima in heatmap')
            plt.show()
        
        local_minima = np.column_stack((x, y))
    
        return local_minima


    def cluster(self, points, method='meanshift'):
        method_types = ['kmeans', 'db', 'meanshift']
        if method not in method_types:
            raise ValueError('Invalid clustering method type. Expected one of: %s' % method_types)

        if method == 'kmeans':
            # Perform KMeans clustering
            k_min=1
            k_max=20
            assert k_max >= k_min, 'k_max needs to be greater than or equal to k_min'

            # Use Elbow method to find the best k
            if k_max == k_min:
                k = k_max
            else:
                k_range = range(k_min, k_max + 1)
                inertias = []
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(points)
                    inertias.append(kmeans.inertia_)
                
                k = self.find_elbow_point(inertias, list(k_range))
                # plt.plot(inertias)
                # plt.show()
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(points)
            # Extract the cluster centroids
            centroids = kmeans.cluster_centers_
            return centroids

        elif method == 'db': 
            # Radius R
            # R = 2.0 / self.resolution  # This will be used as the 'eps' parameter in DBSCAN
            R = 2.0

            # Perform DBSCAN clustering
            db = DBSCAN(eps=R, min_samples=1, metric='euclidean').fit(points)

            # Get cluster labels for each point
            labels = db.labels_

            # Number of clusters
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Ignoring noise if present
            print(f'Number of clusters: {n_clusters}')
            print(f'Cluster labels: {labels}')

            print('Looking for centroids')
            centroids = []
            for label in set(labels):
                cluster_points = points[labels == label]
                centroids.append(np.mean(cluster_points, axis=0))
            
            centroids = np.array(centroids)

            return centroids
        
        elif method == 'meanshift':
            meanshift = MeanShift()
            meanshift.fit(points)
            cluster_centers = meanshift.cluster_centers_
            labels = meanshift.labels_
            return cluster_centers

    def inference(self, fp, visualize_heatmap=False, resolution=0.1):
        """Run inference to find likely observation locations in the floor plan.
        
        Converts floor plan to binary map, creates heatmap, and finds local minima.
        
        Args:
            fp: Floor plan as array of line segments, shape (n, 2, 2).
            visualize_heatmap: If True, visualize the heatmap. Defaults to False.
            resolution: Map resolution in meters per pixel. Defaults to 0.1.
            
        Returns:
            np.ndarray: Array of local minima coordinates representing likely locations.
        """
        # Turns floor plan into binary
        # Note: Floor plan dimensions calculation. Currently uses max coordinates which may
        # not account for negative coordinates or proper bounding box. Consider using
        # min/max range for more robust dimension calculation.
        fp_width = math.ceil(np.max(fp[:, :, 0]) / resolution)
        fp_height = math.ceil(np.max(fp[:, :, 1]) / resolution)
        binary_fp = segments_to_binary_map(fp, width=fp_width, height=fp_height, cell_size=resolution)
        # Creates heat map
        heatmap = self.create_heatmap(binary_fp, visualize_heatmap=False)
        # Find local minima
        local_minima = self.find_local_minima(heatmap, visualize_minima=True)

        return local_minima


    def visualize_kernels(self):
        """Visualize the CES and segment kernels.
        
        Displays the kernels and their combination for debugging purposes.
        """
        plt.imshow(self.CES_kernel, cmap='Greys', origin='lower')
        # plt.plot(-self.min_x / self.resolution, -self.min_y / self.resolution, 'ro')
        plt.show()
        plt.imshow(self.seg_kernel, cmap='Greys', origin='lower')
        # plt.plot(-self.min_x / self.resolution, -self.min_y / self.resolution, 'ro')
        plt.show()

        combined_kernel = -1 * self.CES_kernel + normalize_heatmap(self.seg_kernel)
        plt.imshow(combined_kernel, cmap='Greys', origin='lower')
        plt.show()

        fig, axs = plt.subplots(3, 1)
        axs[0].plot(-self.min_x / self.resolution, -self.min_y / self.resolution, 'ro')
        axs[1].plot(-self.min_x / self.resolution, -self.min_y / self.resolution, 'ro')
        axs[2].plot(-self.min_x / self.resolution, -self.min_y / self.resolution, 'ro')
        axs[0].imshow(self.CES_kernel, cmap='Greys', origin='lower')
        axs[1].imshow(self.seg_kernel, cmap='Greys', origin='lower')
        axs[2].imshow(self.CES_kernel + self.seg_kernel > 0, cmap='Greys', origin='lower')
        plt.show()


    def sharpen_heatmap(self, heatmap):
        """Apply sigmoid sharpening to enhance heatmap contrast.
        
        Args:
            heatmap: Input heatmap array.
            
        Returns:
            np.ndarray: Sharpened heatmap with enhanced contrast.
        """
        return 2 / (1 + np.exp(-(5 * heatmap - 5)))
    