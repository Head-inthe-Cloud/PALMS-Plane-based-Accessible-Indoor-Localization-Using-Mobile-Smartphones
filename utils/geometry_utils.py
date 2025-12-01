import numpy as np
from shapely.vectorized import contains
from shapely.geometry import LineString, Polygon
import matplotlib.pyplot as plt
import math
import alphashape


class Plane:
    '''
    Holds an ARKit vertical plane and projects it to 2D
    '''
    def __init__(self, index, id, anchor_transform, plane_center, plane_extent, detected_time, updated_time):
        self.index = index
        self.id = id
        self.anchor_transform = np.matrix(anchor_transform)  # 4 x 4 numpy matrix
        self.plane_center = np.array(plane_center) # [cx, cy, cz]
        self.plane_extent = np.array(plane_extent) # [width, height, rotation_on_y_axis]
        self.detected_time = detected_time
        self.updated_time = updated_time
        self.T_sp = None   # Transforms coordinates from camera reference frame to session reference frame
    
    # Outputs the coordinates of the two end points in the form of [[x1, y1], [x2, y2]]
    def to_2D(self):
        _ = self.get_plane_center()

        # The plane lies on the x-y plane
        p1_p = np.array([[self.plane_extent[0] / 2, 0, 0, 1]]).T    # Shape: (4, 1)
        p2_p = np.array([[-self.plane_extent[0] / 2, 0, 0, 1]]).T   #  Shape: (4, 1)
        p1_s = np.dot(self.T_sp, p1_p) # Transform the point from the plane reference frame to session reference frame
        p2_s = np.dot(self.T_sp, p2_p)

        # Project to 2D
        # This is tricky, because we want the plane coordinates align with the camera pose system such that:
        #   When camera pose yaw = 0, the planes should be in the direction of yaw = 0, which is the +x direction.
        #   Since in the original 3D coordinates, the "front" direction corresponds to -z, and right is +x, we do the following conversion:
        #   (x', y') = (-z, -x), where x and z are axis from the original 3D coodrinates, and x' and y' are the resulting 2D coordinates
        return np.array([[-p1_s[2, 0], -p1_s[0, 0]], [-p2_s[2, 0], -p2_s[0, 0]]])  # (x', y') = (-z, -x), y is the vertical direction


    # Return center of the plane in session frame, also calculates the transformation matrix form plane frame to session frame
    def get_plane_center(self):
        # Rotation
        theta = self.plane_extent[2] # rotation_on_y_axis
        # transformation matrix form plane reference frame to the anchor reference frame
        T_ap = np.array([
            [ np.cos(theta), 0, np.sin(theta), self.plane_center[0]],
            [ 0,             1, 0,             self.plane_center[1]],
            [-np.sin(theta), 0, np.cos(theta), self.plane_center[2]],
            [ 0,             0, 0,             1]
        ])
        
        T_ap = np.matrix(T_ap)

        T_sa = self.anchor_transform
        self.T_sp = np.dot(T_sa, T_ap)

        Cc = np.array([[0, 0, 0, 1]]).T

        return np.dot(self.T_sp, Cc)


    def __str__(self):
        return 'Plane Object. Index: {}, ID: {}'.format(self.index, self.id)


class Segment:
    '''
    A 2D segment represented by two points
    '''
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.normal = None
        self.orientation = None

    def __str__(self):
        return f'Segment Object ({self.x1}, {self.y1}), ({self.x2}, {self.y2})'


class Bresenham:
    '''
    Implements the Bresenham line-drawing algorithm for grid-based traversal.

    This class is used to discretize a straight line segment between two points
    into a sequence of grid cells, which is commonly applied in path planning,
    raycasting, occupancy grid mapping, and computer graphics. The algorithm
    incrementally determines which cells a line passes through, ensuring
    connectivity without gaps.

    Attributes:
        cell_size (float): The resolution of the grid cells. Input coordinates
            are scaled and rounded according to this value.

    Methods:
        point_rounder(value):
            Rounds a continuous coordinate into a discrete grid index
            based on the cell size, clamping negative results to 0.

        seg(x0, y0, x1, y1):
            Computes the grid cells along a line segment from (x0, y0) to (x1, y1).
            Returns two lists containing the discrete x and y indices of the cells
            traversed by the line.
    '''
    
    def __init__(self, cell_size):
        self.cell_size = cell_size

    def point_rounder(self, value):
        round_res = np.floor(value / self.cell_size)
        if round_res < 0:
            return 0.0
        return round_res

    def seg(self, x0, y0, x1, y1):
        x0 = self.point_rounder(x0)
        y0 = self.point_rounder(y0)
        x1 = self.point_rounder(x1)
        y1 = self.point_rounder(y1)

        dx = np.abs(x1 - x0)
        dy = np.abs(y1 - y0)
        x = x0
        y = y0
        sx = -1.0 if x0 > x1 else 1.0
        sy = -1.0 if y0 > y1 else 1.0

        x_res_list = []
        y_res_list = []

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                x_res_list.append(x)
                y_res_list.append(y)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx

                    # enhance the wall
                    x_res_list.append(x)
                    y_res_list.append(y)
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                x_res_list.append(x)
                y_res_list.append(y)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy

                    # enhance the wall
                    x_res_list.append(x)
                    y_res_list.append(y)
                y += sy

        x_res_list.append(x)
        y_res_list.append(y)
        return x_res_list, y_res_list


def get_orientation_diff_from_segments(seg1, seg2, mode='radian'):
    '''
    Compute the signed orientation difference between two 2D line segments.

    The function calculates the angle between two line segments `seg1` and `seg2`
    defined by their endpoints in 2D. The angle is measured from `seg2` to `seg1`
    following the right-hand rule (positive counter-clockwise, negative clockwise). 
    The orientation is determined using the cross product of the direction vectors.

    Args:
        seg1 (np.ndarray): A 2x2 array representing the first line segment 
            [[x1, y1], [x2, y2]].
        seg2 (np.ndarray): A 2x2 array representing the second line segment 
            [[x1, y1], [x2, y2]].
        mode (str, optional): Output angle unit. 
            - 'radian' (default): Returns the angle in radians.
            - 'degree': Returns the angle in degrees.

    Returns:
        float: Signed angle from `seg2` to `seg1` in the specified unit.
               Positive values indicate counter-clockwise rotation, 
               negative values indicate clockwise rotation.

    Notes:
        - Both `seg1` and `seg2` must be non-degenerate (length > 0).
        - If the vectors are collinear, the result will be 0 or π (depending on direction).
    '''
    v1 = np.array([seg1[1, 0] - seg1[0, 0], seg1[1, 1] - seg1[0, 1]])
    v2 = np.array([seg2[1, 0] - seg2[0, 0], seg2[1, 1] - seg2[0, 1]])
    theta = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    cross_product = np.cross(v1, v2)
    if cross_product < 0:
        theta = -theta

    if mode == 'degree':
        theta = np.degrees(theta)
    return theta


def angular_diff(angle1, angle2, mode='radian'):
    '''
    Finds the absolute difference between two orientations in radians.

    Parameters:
    - angle1: The first orientation in radians.
    - angle2: The second orientation in radians.

    Returns:
    - The absolute difference between the two orientations in radians,
      constrained to the range [0, π].
    '''
    # Normalize angles to the range [0, 2π)
    angle1 = angle1 % (2 * np.pi)
    angle2 = angle2 % (2 * np.pi)
    
    # Calculate the absolute difference
    diff = np.abs(angle1 - angle2)
    
    # Ensure the difference is within the range [0, π]
    if diff > np.pi:
        diff = 2 * np.pi - diff
        
    if mode == 'degree':
        diff / np.pi * 180

    return diff


def get_segments_distance(seg1, seg2, mode):
    """Compute distance between two line segments using specified method.
    
    Args:
        seg1: First line segment as array [[x1, y1], [x2, y2]].
        seg2: Second line segment as array [[x1, y1], [x2, y2]].
        mode: Distance computation method. Options:
            - 'point': Minimum point-to-segment distance (returns 0 if segments intersect).
            - 'ICL': Distance function from ICL paper considering position, orientation, and length.
            - 'NL': Similar to ICL but ignores length difference.
            
    Returns:
        float: Distance between segments according to the specified mode.
    """
    # Naive Solution
    # ------------------------------------
    # If segments intersect, return 0
    if mode == 'point':
        if is_segments_intersect(seg1, seg2):
            return 0
        distances = []
        distances.append(point_to_segment_distance(seg1[0], seg2))
        distances.append(point_to_segment_distance(seg1[1], seg2))
        distances.append(point_to_segment_distance(seg2[0], seg1))
        distances.append(point_to_segment_distance(seg2[1], seg1))
        return min(distances)

    # Distance function from the ICL paper
    # Reference: https://www.sciencedirect.com/science/article/pii/S0898122100002303
    # Note: Current implementation may not fully handle cases where floor plan segments
    # and observed segments have significantly different lengths. Consider adding length
    # normalization or weighted matching based on segment length similarity.
    # ------------------------------------
    elif mode == 'ICL':
        # l1 = length(L1), l2 = length(L2)
        # D(L1, L2) = (l1 + l2) / 2 * (||O1 - O2||^2 + (l1*l2/12) * ||V1-V2||^2 + (l1-l2)^2 / 12)
        # O1 and O2 are the center of the segments, V1 and V2 are unit vectors that represent directions
        alpha = 1
        beta = 1
        gamma = 1

        l1 = np.linalg.norm(seg1)
        l2 = np.linalg.norm(seg2)
        O1 = (seg1[0] + seg1[1]) / 2
        O2 = (seg2[0] + seg2[1]) / 2
        V1 = seg1 / l1
        V2 = seg2 / l2

        dis_component = np.sum((O1 - O2)**2)
        ori_component = (l1 * l2 / 12) * np.sum((V1 - V2)**2)
        len_component = (l1 - l2) ** 2 / 12

        # print(f'Distance: {dis_component}, Orientation: {ori_component}, Length: {len_component}')
        D = (l1 + l2) / 2 * (alpha * dis_component + beta * ori_component + gamma * len_component)
        return D
    
    elif mode == 'NL':
        # Similar to ICL, but ignore length diff
        l1 = np.linalg.norm(seg1)
        l2 = np.linalg.norm(seg2)
        O1 = (seg1[0] + seg1[1]) / 2
        O2 = (seg2[0] + seg2[1]) / 2
        V1 = seg1 / l1
        V2 = seg2 / l2

        D = (np.sum((O1 - O2)**2) + np.sum((V1 - V2)**2))
        return D


def is_segments_intersect(seg1, seg2):
    """Check if two line segments intersect.
    
    Args:
        seg1: First line segment as array [[x1, y1], [x2, y2]].
        seg2: Second line segment as array [[x1, y1], [x2, y2]].
        
    Returns:
        bool: True if segments intersect, False otherwise.
    """
    x11 = seg1[0, 0]
    y11 = seg1[0, 1]
    x12 = seg1[1, 0]
    y12 = seg1[1, 1]
    x21 = seg2[0, 0]
    y21 = seg2[0, 1]
    x22 = seg2[1, 0]
    y22 = seg2[1, 1]

    dx1 = x12 - x11
    dy1 = y12 - y11
    dx2 = x22 - x21
    dy2 = y22 - y21
    delta = dx2 * dy1 - dy2 * dx1
    if delta == 0: return False
    s = (dx1 * (y21 - y11) + dy1 * (x11 - x21)) / delta
    t = (dx2 * (y11- y21) + dy2 * (x21 - x11)) / (-delta)

    return (0 <= s <= 1) and (0 <= t <= 1)


def point_to_segment_distance(point, seg):
    """Compute the shortest distance from a point to a line segment.
    
    Args:
        point: 2D point as array [x, y].
        seg: Line segment as array [[x1, y1], [x2, y2]].
        
    Returns:
        float: Distance from point to segment.
    """
    px, py = point[0], point[1]
    x1, y1, x2, y2 = seg[0, 0], seg[0, 1], seg[1, 0], seg[1, 1]
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:  # the segment's just a point
        return math.hypot(px - x1, py - y1)

    # Calculate the t that minimizes the distance.
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

    # See if this represents one of the segment's
    # end points or a point in the middle.
    if t < 0:
        dx = px - x1
        dy = py - y1
    elif t > 1:
        dx = px - x2
        dy = py - y2
    else:
        near_x = x1 + t * dx
        near_y = y1 + t * dy
        dx = px - near_x
        dy = py - near_y

    return math.hypot(dx, dy)


def point_to_path_distance(p1, path_points):
    '''
    Compute the shortest distance from a 2D point to a polyline path.

    This function measures the distance from a query point `p1` to a path defined 
    by a sequence of 2D points. If the path contains only one point, the function 
    returns the point-to-point distance. If the path has multiple points, it 
    identifies the two nearest path vertices to `p1` and computes the perpendicular 
    (or endpoint) distance from the query point to the line segment connecting them.

    Args:
        p1 (array-like): A 2D point `[x, y]` representing the query location.
        path_points (array-like): An array of shape (N, 2) containing the polyline 
            path coordinates.

    Returns:
        float: The shortest distance from `p1` to the path.

    Notes:
        - Uses Euclidean distance (`L2` norm).
        - For paths with multiple vertices, only the two closest vertices are 
          considered when forming the candidate segment for distance computation.
    '''
    p1 = np.asarray(p1, dtype=float).reshape(2)
    path = np.asarray(path_points, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError('path_points must be an array-like of shape (N, 2)')
    n = path.shape[0]
    if n == 0:
        raise ValueError('path_points is empty')
    if n == 1:
        return point_to_point_distance(p1, path[0])

    # Distances from p1 to every point on the path
    diffs = path - p1
    dists = np.hypot(diffs[:, 0], diffs[:, 1])

    # Indices of the two closest points
    idx = np.argpartition(dists, 2)[:2]
    p2, p3 = path[idx[0]], path[idx[1]]

    # Distance from p1 to the segment (p2, p3)
    seg = np.vstack([p2, p3])
    return point_to_segment_distance(p1, seg)


def points_to_path_distances(points, path_points):
    """Compute distances from multiple points to a polyline path.
    
    For each point, finds the two nearest path vertices and computes the distance
    to the line segment connecting them.
    
    Args:
        points: Array of 2D points, shape (n, 2).
        path_points: Array of path vertices, shape (m, 2).
        
    Returns:
        list: List of distances, one per input point.
    """
    distances = []
    for p in points:
        dist = point_to_path_distance(p, path_points)
        distances.append(dist)
    return distances


def point_to_point_distance(p1, p2):
    """Compute Euclidean distance between two 2D points.
    
    Args:
        p1: First point as array [x, y] or tuple (x, y).
        p2: Second point as array [x, y] or tuple (x, y).
        
    Returns:
        float: Euclidean distance between the two points.
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_intersection_point(seg1, seg2):
    """Find the intersection point of two line segments.
    
    Args:
        seg1: First line segment as array [[x1, y1], [x2, y2]].
        seg2: Second line segment as array [[x1, y1], [x2, y2]].
        
    Returns:
        np.ndarray or None: Intersection point [x, y] if segments intersect, None otherwise.
    """
    xdiff = (seg1[0][0] - seg1[1][0], seg2[0][0] - seg2[1][0])
    ydiff = (seg1[0][1] - seg1[1][1], seg2[0][1] - seg2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det(*seg1), det(*seg2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y])


def apply_transformation_to_segments(segments, T):
    """Apply a 3x3 homogeneous transformation matrix to line segments.
    
    Args:
        segments: Array of line segments, shape (n, 2, 2).
        T: 3x3 homogeneous transformation matrix.
        
    Returns:
        np.ndarray: Transformed segments, same shape as input.
    """
    segments_transformed = np.copy(segments)
    for i in range(segments.shape[0]):
        for j in range(2):
            point_aug = np.array([[segments[i, j, 0], segments[i, j, 1], 1]]).T
            segments_transformed[i, j] = np.dot(T, point_aug).T[0, :-1]
    return segments_transformed


def apply_transformation_to_points(points, T):
    """Apply a 3x3 homogeneous transformation matrix to 2D points.
    
    Args:
        points: Array of 2D points, shape (n, 2).
        T: 3x3 homogeneous transformation matrix.
        
    Returns:
        np.ndarray: Transformed points, same shape as input.
    """
    points_transformed = np.zeros_like(points)
    for i in range(points.shape[0]):
        point_aug = np.array([[points[i, 0], points[i, 1], 1]]).T
        points_transformed[i] = np.dot(T, point_aug).T[0, :-1]
    return points_transformed


def filter_segments_by_distance(segments, point, range_limit=20):
    """Filter segments that are within a specified distance from a point.
    
    Args:
        segments: Array of line segments, shape (n, 2, 2).
        point: Reference point as array [x, y].
        range_limit: Maximum distance threshold.
        
    Returns:
        np.ndarray: Filtered segments within the distance threshold.
    """
    filtered_segments = []
    for seg in segments:
        if point_to_segment_distance(point, seg) < range_limit:
            filtered_segments.append(seg)
    return np.array(filtered_segments)


def filter_segments_by_length(segments, min_length):
    """Filter segments that meet a minimum length requirement.
    
    Args:
        segments: Array of line segments, shape (n, 2, 2).
        min_length: Minimum segment length threshold.
        
    Returns:
        np.ndarray: Filtered segments with length >= min_length.
    """
    filtered_segments = []
    for seg in segments:
        if get_segment_length(seg) > min_length:
            filtered_segments.append(seg)
    return np.array(filtered_segments)


def get_segment_length(seg):
    """Compute the length of a line segment.
    
    Args:
        seg: Line segment as array [[x1, y1], [x2, y2]].
        
    Returns:
        float: Euclidean length of the segment.
    """
    return np.sqrt((seg[0, 0] - seg[1, 0])**2 + (seg[0, 1] - seg[1, 1])**2)


def order_segments_by_distance(segments, reference_seg, mode='point'):
    """Sort segments by their distance to a reference segment.
    
    Args:
        segments: Array of line segments to sort, shape (n, 2, 2).
        reference_seg: Reference segment for distance computation.
        mode: Distance computation mode (see get_segments_distance).
        
    Returns:
        list: Segments sorted by distance to reference segment (closest first).
    """
    result = sorted(segments, key=lambda segment: get_segments_distance(reference_seg, segment, mode))
    return result


def get_segment_orientations(segments):
    """Compute orientation angles for a set of line segments.
    
    Orientations are constrained to the range [-π/2, π/2] by wrapping angles
    outside this range.
    
    Args:
        segments: Array of line segments, shape (n, 2, 2).
        
    Returns:
        np.ndarray: Array of orientations in radians, shape (n,).
    """
    orientations = []
    for segment in segments:
        vector = segment[1] - segment[0]
        # Use arctan to find orientation g
        orientation = np.arctan2(vector[1], vector[0])  # This value is within the range [-pi/2, pi/2]
        if orientation < -np.pi/2:
            orientation += np.pi
        elif orientation > np.pi/2:
            orientation -= np.pi

        orientations.append(orientation)
    return np.array(orientations)


def find_principal_orientations(segments, num_bins=360, threshold=np.radians(1)):
    '''
    Estimate the dominant orthogonal orientations of a set of line segments.

    This function computes the orientations of all given line segments, builds 
    a histogram of their distribution (weighted by segment length), and identifies 
    the two principal orthogonal orientations that best explain the data. The 
    method bins orientations between -π/2 and π/2, sums weighted contributions 
    within a tolerance window, and selects the orientation pair with the highest 
    combined score.

    Args:
        segments (list or np.ndarray): List/array of line segments, each defined 
            by two endpoints with shape (2, 2).
        num_bins (int, optional): Number of orientation bins (default: 360). 
            Higher values provide finer resolution.
        threshold (float, optional): Angular tolerance in radians for counting 
            a segment toward a bin (default: 1 degree in radians).

    Returns:
        np.ndarray: Array of shape (2,) containing the two principal orientations 
        (in radians). The second orientation is orthogonal (π/2 rotated) to the first.

    Notes:
        - Segments are weighted by their lengths when populating orientation bins.
        - Orientations are constrained to the range [-π/2, π/2].
        - This is commonly used for detecting dominant layout directions in 
          structured environments (e.g., Manhattan-world assumptions).
    '''
    orientations = get_segment_orientations(segments)
    threshold = threshold
    bins = [0] * num_bins
    # Iterate over -pi/2 to pi/2
    deltas = np.arange(0, 180, 180 / num_bins)
    for delta_idx, delta in enumerate(deltas):
        theta = -np.pi/2 + np.radians(delta)
        theta_1 = theta - threshold
        theta_2 = theta + threshold

        if theta_1 < -np.pi/2:
            theta_1 = (theta_1 // (-np.pi/2)) * (np.pi/2) - theta_1 % (np.pi/2)
        if theta_2 > np.pi/2:
            theta_2 = (theta_2 // (np.pi/2)) * (-np.pi/2) + theta_2 % (np.pi/2)
        
        for idx, orientation in enumerate(orientations):
            if theta_1 <= orientation <= theta_2:
                # bins[delta] += 1                         # Only count number of segments, ignore length
                bins[delta_idx] += get_segment_length(segments[idx])   # Weighted by segment length

    best_ori = 0
    best_ori_score = 0
    for idx_1 in range(int(num_bins // 2 + num_bins % 2)):
        idx_2 = int((idx_1 + 90 / (180 / num_bins)) % num_bins)
        score = bins[idx_1] + bins[idx_2]
        if score > best_ori_score:
            best_ori = np.radians(idx_1 * (180 / num_bins)) - np.pi/2
            best_ori_score = score

    ortho_ori = best_ori + np.pi/2
    if ortho_ori > np.pi/2:
        ortho_ori = (ortho_ori // (np.pi/2)) * (-np.pi/2) + ortho_ori % (np.pi/2)

    return np.array([best_ori, ortho_ori])


def find_top_relative_orientations(segments1, segments2, k=4, num_bins=360, visualize_hist=False, nms_radius=10):
    '''
    Find the top-k relative orientation differences between two sets of line segments.

    This function is similar in spirit to `find_principal_orientations`, but instead of 
    finding dominant orthogonal directions within a single set of segments, it compares 
    two segment sets by building orientation histograms (weighted by segment length) 
    and computing their cross-correlation by performing a convolution. The peaks in this correlation indicate the 
    most likely relative rotations between the two sets of orientations.

    Args:
        segments1 (list or np.ndarray): First set of line segments, each of shape (2, 2).
        segments2 (list or np.ndarray): Second set of line segments, each of shape (2, 2).
        k (int, optional): Number of top relative orientations to return. Defaults to 3.
        num_bins (int, optional): Number of orientation bins over [-90°, 90°). 
            Defaults to 360, which corresponds to 0.5° per bin.
        nms_radius (int, optional): Number of orientations bins around the selected bin to supress using Non-max Supression

    Returns:
        np.ndarray: Array of shape (k,) containing the top-k relative orientation 
        differences in radians between `segments1` and `segments2`.

    Notes:
        - Segments are weighted by their lengths when building histograms.
        - The result indicates *relative* rotation angles that best align the 
          dominant orientations of `segments1` with those of `segments2`.
        - See `find_principal_orientations` for related logic on estimating 
          dominant orientations within a single set of segments.
    '''
    assert k % 2 == 0, 'k needs to be even'

    # Step 1: Create histogrames
    ori1 = get_segment_orientations(segments1)
    ori2 = get_segment_orientations(segments2)

    length1 = [get_segment_length(segment) for segment in segments1]
    length2 = [get_segment_length(segment) for segment in segments2]

    hist1 = [0] * num_bins
    hist2 = [0] * num_bins

    ori1_deg = (np.rad2deg(np.array(ori1)) + 90) % 180
    ori2_deg = (np.rad2deg(np.array(ori2)) + 90) % 180

    bin_width = 180 / num_bins
    bin_indices1 = (ori1_deg // bin_width).astype(int)
    bin_indices2 = (ori2_deg // bin_width).astype(int)

    for idx, bin in enumerate(bin_indices1):
        hist1[bin] += length1[idx]

    for idx, bin in enumerate(bin_indices2):
        hist2[bin] += length2[idx]

    hist1 = np.array(hist1)
    hist2 = np.array(hist2)

    # Step 2: perform convolution
    features = np.zeros_like(hist1)
    for delta_idx in range(num_bins):
        kernel = hist2.copy()
        kernel = np.roll(kernel, delta_idx)
        features[delta_idx] = np.sum(hist1 * kernel)

    # Visualization
    if visualize_hist:
        angles = np.linspace(0, 180, num_bins, endpoint=False)  # Bin centers
        width = 180 / num_bins

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        axs[0].bar(angles, hist1, width=width, align='edge', color='blue')
        axs[0].set_title('Histogram 1')
        axs[0].set_xlabel('Orientation (degrees)')
        axs[0].set_ylabel('Weighted count')

        axs[1].bar(angles, hist2, width=width, align='edge', color='green')
        axs[1].set_title('Histogram 2')
        axs[1].set_xlabel('Orientation (degrees)')
        axs[1].set_ylabel('Weighted count')

        plt.tight_layout()
        plt.show()

        # Show features
        plt.bar(angles, features, width=width, align='edge', color='blue')
        plt.show()


    # Non-max Supression
    features_copy = features.copy()
    selected_bins = []

    # Number of values to pick (same as original: k/2)
    num_to_pick = int(k / 2)

    for _ in range(num_to_pick):
        max_idx = np.argmax(features_copy)
        max_val = features_copy[max_idx]
        if max_val <= 0:
            break  # no more positive values

        selected_bins.append(max_idx)

        # Suppress neighborhood around the max index
        start = max_idx - nms_radius
        end = max_idx + nms_radius + 1
        features_copy[start:end] = 0

        # Handle wrap-around (circular bins)
        if start < 0:
            features_copy[start:] = 0
            features_copy[:end] = 0
        if end > len(features_copy):
            features_copy[start:] = 0
            features_copy[:end - len(features_copy)] = 0
    
    top_bins = np.array(selected_bins, dtype=int)
    top_thetas = np.radians(top_bins * bin_width)
    rev_top_thetas = (top_thetas + np.radians(180)) % (2 * np.pi)

    top_k_thetas = np.concatenate((top_thetas, rev_top_thetas))
    return top_k_thetas


def get_R_from_orientations(ori_1=None, ori_2=None, theta=None):
    '''
    Compute a 2D rotation matrix from two orientations or a given angle.

    This function generates a 2x2 rotation matrix `R` that rotates a vector 
    oriented at `ori_1` (in radians) into alignment with `ori_2`. If `theta` 
    is provided directly, it is used as the rotation angle. Otherwise, the 
    rotation angle is computed as `theta = ori_2 - ori_1`.

    Args:
        ori_1 (float, optional): Initial orientation in radians, constrained 
            to [-π/2, π/2]. Used only if `theta` is None.
        ori_2 (float, optional): Target orientation in radians, constrained 
            to [-π/2, π/2]. Used only if `theta` is None.
        theta (float, optional): Rotation angle in radians. If provided, this 
            overrides the difference `ori_2 - ori_1`.

    Returns:
        tuple:
            - R (np.ndarray): A 2x2 rotation matrix.
            - theta (float): The rotation angle in radians that transforms 
              `ori_1` into `ori_2`.

    Notes:
        - If both `ori_1` and `ori_2` are provided and `theta` is None, 
          the function computes the relative rotation.
        - If `theta` is provided, `ori_1` and `ori_2` are ignored.
    '''
    if theta is None:
        theta = ori_2 - ori_1
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    return R, theta


def get_rotation_matrix_from_two_vectors(a, b):
    """Compute rotation matrix that rotates vector a to align with vector b.
    
    Uses Rodrigues' rotation formula. Handles edge cases where vectors are
    parallel or anti-parallel.
    
    Args:
        a: Source 3D vector.
        b: Target 3D vector.
        
    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    
    if np.isclose(c, 1.0):
        return np.eye(3)  # No rotation needed
    elif np.isclose(c, -1.0):
        # 180-degree rotation around any axis orthogonal to a
        axis = np.array([1, 0, 0]) if not np.allclose(a, [1, 0, 0]) else np.array([0, 1, 0])
        v = np.cross(a, axis)
        v = v / np.linalg.norm(v)
        return rotation_matrix_from_axis_angle(v, np.pi)

    skew = np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0]
    ])
    R = np.eye(3) + skew + skew @ skew * ((1 - c) / (np.linalg.norm(v) ** 2))
    return R


def rotation_matrix_from_axis_angle(axis, angle):
    """Compute rotation matrix from axis-angle representation using Rodrigues' formula.
    
    Args:
        axis: Rotation axis as 3D vector (will be normalized).
        angle: Rotation angle in radians.
        
    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    I = np.eye(3)
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return R


def get_theta_from_R(rotation_matrix):
    '''
    Calculate the rotation angle from a 2x2 rotation matrix.

    Args:
        rotation_matrix (np.array): A 2x2 rotation matrix.

    Returns:
        float: The rotation angle in radians.
    '''
    # Check if the matrix is 2x2
    if rotation_matrix.shape != (2, 2):
        raise ValueError('The input must be a 2x2 matrix.')

    # Ensure it's a rotation matrix by checking if its transpose is its inverse
    if not np.allclose(np.linalg.inv(rotation_matrix), np.transpose(rotation_matrix)):
        raise ValueError('The input must be a valid rotation matrix.')

    # Calculate the angle using the arctangent of the matrix elements
    angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    
    return angle


def find_optimal_rotation(A, B):
    """Find optimal rotation matrix between two point sets using SVD.
    
    Computes the rotation that minimizes the sum of squared distances between
    corresponding points in sets A and B after centering.
    
    Args:
        A: Source point set, shape (n, 2) or (n, 3).
        B: Target point set, shape (n, 2) or (n, 3), same size as A.
        
    Returns:
        np.ndarray: Optimal rotation matrix (2x2 or 3x3 depending on input dimension).
    """
    # Center the points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # Compute the covariance matrix
    H = np.dot(A_centered.T, B_centered)

    # Perform SVD
    U, S, Vt = np.linalg.svd(H)

    # Compute the optimal rotation matrix
    R = np.dot(Vt.T, U.T)

    return R


def angle_to_unit_arrow(theta):
    """Convert an angle to a unit direction vector.
    
    Args:
        theta: Angle in radians.
    
    Returns:
        list: Unit direction vector [cos(theta), sin(theta)].
    """
    x = np.cos(theta)
    y = np.sin(theta)
    return [x, y]


def sample_points_from_segments(segments, sample_density=0.1, segment_labels=None):
    '''
    Sample evenly spaced points along a collection of 2D line segments.

    This function discretizes line segments into regularly spaced points with
    a specified sampling density. It can either return all sampled points 
    together with their originating segment indices, or group sampled points 
    according to provided segment label groups (e.g., orientation clusters).

    Args:
        segments (np.ndarray): Array of shape (N, 2, 2) representing N line 
            segments, where each segment is defined by two endpoints [[x1, y1], [x2, y2]].
        sample_density (float, optional): Distance between consecutive sampled 
            points along a segment. Defaults to 0.1 (e.g., 10 cm if working in meters).
        segment_labels (list of list[int], optional): Groups of segment indices, 
            where each inner list specifies which segments belong to that group. 
            If provided, sampled points are grouped accordingly. Defaults to None.

    Returns:
        tuple:
            - points (np.ndarray or list[np.ndarray]): 
                * If `segment_labels` is None: A single array of shape (M, 2) 
                  containing all sampled points.
                * If `segment_labels` is provided: A list of arrays, one per 
                  group, each of shape (Mi, 2) with sampled points.
            - labels (np.ndarray or None): 
                * If `segment_labels` is None: Array of shape (M,) indicating 
                  the segment index each point originated from.
                * If `segment_labels` is provided: Returns None.

    Notes:
        - Endpoints of each segment are always included in the sampled points.
        - When `segment_labels` is used, segments not belonging to any group 
          are ignored.
        - Sampling density is enforced as an integer subdivision of the segment 
          length, so spacing may slightly differ near the endpoints.
    '''
    if segment_labels is None:
        points = []
        labels = []
        for seg_idx, segment in enumerate(segments):
            vector = segment[1] - segment[0]
            length = get_segment_length(segment)
            num_points = int(np.ceil(length / sample_density))
            dx = vector[0] / num_points
            dy = vector[1] / num_points
            for i in range(num_points + 1):
                point = [segment[0, 0] + dx * i, segment[0, 1] + dy * i]
                points.append(point)
                labels.append(seg_idx)
        
            
        return np.array(points), labels

    else:
        points = [[] for _ in range(len(segment_labels))]
        for seg_idx, segment in enumerate(segments):
            # Find which group the segment belongs to
            segment_label = None
            for i in range(len(segment_labels)):
                if seg_idx in segment_labels[i]:
                    segment_label = i
                    break

            # If the segment does not belong to any orientation group, we ignore it
            if segment_label is None:
                continue

            # Sample points from this segment
            vector = segment[1] - segment[0]
            length = get_segment_length(segment)
            num_points = int(length // sample_density)
            dx = vector[0] / num_points
            dy = vector[1] / num_points
            for i in range(num_points + 1):
                point = [segment[0, 0] + dx * i, segment[0, 1] + dy * i]
                points[segment_label].append(point)
        points = [np.array(sub_points) for sub_points in points]
        
        return points, None

    
def segments_to_binary_map(segments, cell_size, filler_value=0, width=None, height=None, to_bot_left=False):
    '''
    Convert line segments into a 2D binary occupancy map.

    Each segment is rasterized onto a grid using the Bresenham algorithm, with 
    resolution defined by `cell_size`. The output is a binary map where cells 
    touched by any segment are marked as 1, and others are filled with 
    `filler_value`.

    Args:
        segments (np.ndarray): Array of shape (N, 2, 2) containing line segments.
        cell_size (float): Resolution of grid cells.
        filler_value (int, optional): Default value for non-segment cells (0 or 1). Defaults to 0.
        width (int, optional): Predefined map width in cells. If None, computed from bounds.
        height (int, optional): Predefined map height in cells. If None, computed from bounds.
        to_bot_left (bool, optional): If True, shifts segments so the minimum x,y 
            starts at the bottom-left corner. Defaults to False.

    Returns:
        np.ndarray: 2D binary occupancy map (height x width).
    '''
    BH = Bresenham(cell_size)

    segments_copy = segments.copy()
    min_x = np.min(segments_copy[:, :, 0])
    min_y = np.min(segments_copy[:, :, 1])
    max_x = np.max(segments_copy[:, :, 0])
    max_y = np.max(segments_copy[:, :, 1])

    if to_bot_left:
        segments_copy[:, :, 0] -= min_x
        segments_copy[:, :, 1] -= min_y

    if width is None or height is None:
        width = int(np.ceil((max_x - min_x) / cell_size))
        height = int(np.ceil((max_y - min_y) / cell_size))


    if filler_value == 0:
        binary_map = np.zeros((height, width), dtype=np.uint8)
    else:
        binary_map = np.ones((height, width), dtype=np.uint8) * filler_value

    for segment in segments_copy:
        x0, y0 = segment[0]
        x1, y1 = segment[1]
        x_res_list, y_res_list = BH.seg(x0, y0, x1, y1)
        for x, y in zip(x_res_list, y_res_list):
            if 0 <= x < width and 0 <= y < height:
                binary_map[int(y), int(x)] = 1

    return binary_map


def is_point_in_fov(point, location, direction, fov, radius):
    '''
    Check whether a 2D point lies within a field of view (FOV).

    The function tests if a given point is both within a maximum radius from 
    an observer location and inside the angular bounds of the observer's FOV. 
    The FOV is centered on a given viewing `direction` and spans an angular 
    range of `fov` radians.

    Args:
        point (array-like): The target point `[x, y]`.
        location (array-like): Observer's position `[x, y]`.
        direction (float): Viewing direction in radians, measured from the 
            positive x-axis.
        fov (float): Field of view angle in radians (centered at `direction`).
        radius (float): Maximum distance from the observer to consider.

    Returns:
        bool: True if the point lies within both the radius and FOV bounds, 
        False otherwise.

    Notes:
        - The FOV is symmetric about `direction` with half-angle `fov / 2`.
        - Angular difference is normalized to the range [-π, π].
    '''
    vec = np.array(point) - np.array(location)
    dist = np.linalg.norm(vec)
    if dist > radius:
        return False
    angle = np.arctan2(vec[1], vec[0])
    delta = ((angle - direction + np.pi) % (2 * np.pi)) - np.pi
    return abs(delta) <= fov / 2


def build_fov_polygon(location, direction, fov, radius, num_points=100):
    """
    Construct a polygon representing a field-of-view (FOV) sector.

    The polygon is defined as a circular sector centered at `location`,
    spanning the angular range [`direction - fov/2`, `direction + fov/2`],
    with the given `radius`.

    Args:
        location (tuple): (x, y) coordinates of the observer.
        direction (float): Central viewing direction in radians.
        fov (float): Field of view angle in radians.
        radius (float): Radius of the FOV sector.
        num_points (int, optional): Number of points used to approximate the arc. Default is 100.

    Returns:
        shapely.geometry.Polygon: Polygon representing the FOV sector.
    """
    angles = np.linspace(direction - fov / 2, direction + fov / 2, num_points)
    arc_points = [
        (location[0] + radius * np.cos(a), location[1] + radius * np.sin(a))
        for a in angles
    ]
    coords = [location] + arc_points + [location]
    return Polygon(coords)


def filter_segments_fov_clipping(vector_map, location, direction, fov):
    """
    Filter and clip segments to retain only those within a field of view (FOV) sector.

    Each segment in `vector_map` is tested against the circular FOV sector
    centered at `location`. Segments fully inside are kept, while partially
    intersecting ones are clipped to fit the FOV boundary. The radius of the FOV
    is automatically set to the maximum distance from `location` to any point in the map.

    Args:
        vector_map (np.ndarray): Array of shape (n, 2, 2) representing n line segments.
        location (tuple): (x, y) coordinates of the observer.
        direction (float): Central viewing direction in radians.
        fov (float): Field of view angle in radians.

    Returns:
        np.ndarray: Array of shape (m, 2, 2) of accepted or clipped segments,
                    translated so that `location` becomes the origin.
    """
    # Compute radius automatically
    all_points = vector_map.reshape(-1, 2)
    distances = np.linalg.norm(all_points - np.array(location), axis=1)
    radius = distances.max() + 1e-3  # small buffer

    fov_sector = build_fov_polygon(location, direction, fov, radius)
    kept_segments = []

    for seg in vector_map:
        p1, p2 = seg

        in1 = is_point_in_fov(p1, location, direction, fov, radius)
        in2 = is_point_in_fov(p2, location, direction, fov, radius)

        line = LineString([p1, p2])

        if in1 and in2:
            kept_segments.append(np.array([p1, p2]))
        else:
            clipped = line.intersection(fov_sector)
            if clipped.is_empty:
                continue
            if clipped.geom_type == 'LineString':
                coords = np.array(clipped.coords)
                if len(coords) == 2:
                    kept_segments.append(coords)
            elif clipped.geom_type == 'MultiLineString':
                for part in clipped:
                    coords = np.array(part.coords)
                    if len(coords) == 2:
                        kept_segments.append(coords)

    kept_segments = np.array(kept_segments)
    kept_segments = translate_segments(kept_segments, -location[0], -location[1])
    return kept_segments


def merge_segments(segments, distance_threshold=0.2, angle_threshold=np.pi/180 * 10):
    '''
    Merge nearby and similarly oriented line segments into longer continuous segments.

    This function iteratively groups line segments that are close to each other 
    and have small orientation differences, then merges each group into a single 
    representative segment. This helps simplify vector maps by reducing redundant 
    or fragmented segments.

    Args:
        segments (array-like): Collection of line segments of shape (N, 2, 2), 
            where each segment is defined by two endpoints [[x1, y1], [x2, y2]].
        distance_threshold (float, optional): Maximum allowed distance between 
            two segments to consider them for merging. Defaults to 0.2.
        angle_threshold (float, optional): Maximum allowed angular difference 
            (in radians) between segments to consider them parallel. 
            Defaults to 10 degrees in radians.

    Returns:
        np.ndarray: Array of merged line segments with shape (M, 2, 2), 
        where M ≤ N.

    Notes:
        - Uses `get_segments_distance` with mode='point' to measure proximity.
        - Uses `get_orientation_diff_from_segments` to check angular alignment.
        - Each group of compatible segments is merged using `merge_segments_helper`.
        - The process is greedy: segments are merged in order and removed 
          once processed.
    '''
    segments = list(segments)
    
    merged_segments = []
    while len(segments) > 0:
        segment = segments[0]
        segments = segments[1:]
        segment_group = [segment]
        for i, other_segment in enumerate(segments):
            distance = get_segments_distance(segment, other_segment, mode='point')
            angle_diff = get_orientation_diff_from_segments(segment, other_segment, mode='radian')
            if distance < distance_threshold and abs(angle_diff) < angle_threshold:
                segment_group.append(other_segment)
                segments.pop(i)
        merged_segments.append(merge_segments_helper(segment_group))
    return np.array(merged_segments)


def merge_segments_helper(segments):
    """Merge a group of nearby segments into a single representative segment.
    
    Computes average orientation and projects all endpoints onto the average
    direction to find the furthest endpoints.
    
    Args:
        segments: List of line segments to merge, each shape (2, 2).
        
    Returns:
        np.ndarray: Merged segment as [[x1, y1], [x2, y2]].
    """
    orientations = get_segment_orientations(segments)
    # Calculate average angle
    average_ori = np.mean(orientations)
    
    # Calculate direction vector from average angle
    ori_vector = np.array([np.cos(average_ori), np.sin(average_ori)])
    
    # Project endpoints onto the average direction
    projections = []
    for segment in segments:
        x1, y1 = segment[0]
        x2, y2 = segment[1]
        projections.append((np.array([x1, y1]), np.dot(np.array([x1, y1]), ori_vector)))
        projections.append((np.array([x2, y2]), np.dot(np.array([x2, y2]), ori_vector)))
    
    # Find the furthest points along the projection
    projections.sort(key=lambda p: p[1])
    furthest_points = [projections[0][0], projections[-1][0]]
    
    return np.array(furthest_points)


def rotate_segments(segments, ori_1=None, ori_2=None, theta=None):
    """Rotate line segments by a specified angle.
    
    Args:
        segments: Array of line segments, shape (n, 2, 2).
        ori_1: Initial orientation in radians (used if theta is None).
        ori_2: Target orientation in radians (used if theta is None).
        theta: Rotation angle in radians. If provided, overrides ori_1 and ori_2.
        
    Returns:
        np.ndarray: Rotated segments, same shape as input.
    """
    if theta is None:
        theta = ori_2 - ori_1
    R, theta = get_R_from_orientations(theta=theta)
    T = np.eye(3)
    T[:2, :2] = R
    segments_T = apply_transformation_to_segments(segments, T=T)
    return segments_T


def translate_segments(segments, x, y):
    '''
    Translates a list of 2D line segments by (x, y) using a transformation matrix.

    Args:
    - segments (list): List of line segments in the form [[[x1, y1], [x2, y2]], ...].
    - x (float): Translation offset in the x-direction.
    - y (float): Translation offset in the y-direction.

    Returns:
    - translated_segments (list): List of translated segments.
    '''
    # Define the 3x3 translation matrix for homogeneous coordinates
    T = np.array([
        [1, 0, x],  # Translate in X
        [0, 1, y],  # Translate in Y
        [0, 0, 1]   # Homogeneous coordinate
    ])

    # Apply transformation using the helper function
    translated_segments = apply_transformation_to_segments(segments, T=T)

    return translated_segments


def rotate_segments_to_landscape(segments):
    '''
    Takes a set of segments, usually a vector floor plan, and rotate it according to its main orientation
    Such that the width is greater than the height, aka landscape position

    Parameters:
        segments - the segments to be rotated, shape (n, 2, 2)
    Returns:
        segments_T - the rotated segments
        T - the transformation matrix
        theta - the angle the segments were rotated by, in radians
    '''
    # Find the main orientation
    main_orientations = find_principal_orientations(segments, num_bins=360, threshold=np.radians(0.5))
    original_orientation = main_orientations[1]
    R, theta = get_R_from_orientations(original_orientation, 0)
    T = np.eye(3)
    T[:2, :2] = R

    # Aligns with the 0 degree
    segments_T = apply_transformation_to_segments(segments, T=T)

    # Check if width is greater than height, otherwise, rotate it +- 90 degrees, which ever is closer to the original orientation
   
    # Find the max and min values among the x-coordinates of the rotated points

    # Compute the span
    x_coords = np.concatenate((segments_T[:, 0, 0], segments_T[:, 1, 0]))
    y_coords = np.concatenate((segments_T[:, 0, 1], segments_T[:, 1, 1]))
    max_x = np.max(x_coords)
    min_x = np.min(x_coords)
    max_y = np.max(y_coords)
    min_y = np.min(y_coords)
    span_x = max_x - min_x
    span_y = max_y - min_y

    if span_y > span_x:
        final_orientation = np.pi / 2

        if angular_diff(original_orientation, np.pi / 2) > np.pi / 2:
            final_orientation *= -1
        
        R, theta = get_R_from_orientations(original_orientation, final_orientation)
        T = np.eye(3)
        T[:2, :2] = R
        segments_T = apply_transformation_to_segments(segments, T=T)
        x_coords = np.concatenate((segments_T[:, 0, 0], segments_T[:, 1, 0]))
        y_coords = np.concatenate((segments_T[:, 0, 1], segments_T[:, 1, 1]))
        min_x = np.min(x_coords)
        min_y = np.min(y_coords)

    # Translate the map such that it starts from 0
    T_t = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0,  1]
    ])
    segments_T = apply_transformation_to_segments(segments_T, T_t)

    T = T_t @ T

    return segments_T, T, theta


def alpha_shape(segments: np.ndarray, alpha=0.15, cell_size=0.1, visualize=False) -> np.ndarray:
    '''
    Compute the alpha shape (concave hull) of a set of points.
    Parameters:
    points (np.array): Array of points.
    alpha (float): Alpha parameter.
    Returns:
    shapely.geometry.Polygon: The alpha shape.
    '''
    sampled_points = sample_points_from_segments(segments=segments, sample_density=1)[0]
    
    mask = alphashape.alphashape(sampled_points, alpha)

    if visualize:
        try:
            x, y = mask.exterior.xy
            for seg in segments:
                plt.plot(*zip(*seg), color='black')

            plt.plot(*zip(*sampled_points), 'ro', markersize=1)

            plt.plot(x, y)
            plt.fill(x, y, alpha=0.5)
            plt.axis('equal')
            plt.show()
        except:
            print('No mask found')
            return
    
    points = segments.reshape((segments.shape[0] * 2, 2))
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    x_coords = np.arange(min_x, max_x, cell_size)
    y_coords = np.arange(min_y, max_y, cell_size)
    xv, yv = np.meshgrid(x_coords, y_coords)
    points = np.vstack((xv.flatten(), yv.flatten())).T

    binary_mask = contains(mask, points[:, 0], points[:, 1])
    binary_mask = binary_mask.reshape(xv.shape)

    return binary_mask


def point_to_point_distances(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances between corresponding points.
    
    Args:
        points1: First array of 2D points, shape (n, 2).
        points2: Second array of 2D points, shape (n, 2), same size as points1.
        
    Returns:
        np.ndarray: Array of distances, shape (n,).
    """
    assert len(points1) == len(points2), f'Error: the lengths of the arrays of points must be the same, the first array has length {len(points1)}, while the second one is {len(points2)}'

    distances = []
    for i in range(len(points1)):
        distances.append(np.linalg.norm(points2[i] - points1[i]))
    
    return np.array(distances)


def get_segment_orientation_at_xy(segments, point, resolution=0.1):
    """Find the orientation of the segment nearest to a given point.
    
    Args:
        segments: Array of line segments, shape (n, 2, 2).
        point: Query point as array [x, y].
        resolution: Distance threshold for finding nearby segments.
        
    Returns:
        float or None: Orientation angle in radians (-π/2 to π/2) if exactly one
            segment is found within the threshold, None otherwise.
    """
    filtered_segments = filter_segments_by_distance(segments, point, range_limit=resolution)
    n = len(filtered_segments)
    if n == 1:
        return(get_segment_orientations(filtered_segments)[0])
    return None
    

def normalize_heatmap(heatmap, linear_normalization=True, as_prob_distribution=False):
    '''
    Normalize a heatmap with optional linear scaling and probability distribution conversion.

    This function takes a heatmap (2D or N-dimensional array) and optionally applies 
    linear normalization to rescale values into the [0, 1] range. It can also 
    convert the normalized heatmap into a valid probability distribution where 
    all values sum to 1 and very small values are clamped to a minimum epsilon 
    to avoid zeros.

    Args:
        heatmap (array-like): Input heatmap, can be 2D or higher-dimensional.
        linear_normalization (bool, optional): If True, linearly scales values 
            to the [0, 1] range using min-max normalization. Defaults to True.
        as_prob_distribution (bool, optional): If True, normalizes values to sum 
            to 1, converting the heatmap into a probability distribution. 
            Very small probabilities are clipped to an epsilon for numerical stability. 
            Defaults to False.

    Returns:
        np.ndarray: Normalized heatmap, same shape as input.

    Notes:
        - If all values in the heatmap are constant, the function returns the 
          original array (or a uniform distribution if `as_prob_distribution=True`).
        - When `as_prob_distribution=True`, the final heatmap values are guaranteed 
          to sum to 1 and be strictly positive.
    '''
    normalized_heatmap = np.asarray(heatmap, dtype=float).copy()
    if linear_normalization:
        min_val = np.min(heatmap)
        max_val = np.max(heatmap)
        if min_val == max_val:
            if as_prob_distribution:
                return np.ones_like(heatmap) * 1 / heatmap.size
            return heatmap
        normalized_heatmap = (heatmap - min_val) / (max_val - min_val)

    if as_prob_distribution:
        eps = 1e-8
        total = np.sum(normalized_heatmap)
        normalized_heatmap /= total  # All sum to 1

        # Makes the lowest probability equal to eps, while still all sum to 1
        mask = normalized_heatmap < eps
        inverse_mask = ~mask
        normalized_heatmap[inverse_mask] *= 1 - (np.sum(mask) * eps)
        normalized_heatmap[mask] += eps
        
    return normalized_heatmap
