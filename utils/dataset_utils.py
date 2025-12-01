import cv2
import numpy as np
from shapely.geometry import Point, LineString

# Some of the code here are from the LASER Github, please refer to
# https://github.com/zillow/laser

def is_polygon_clockwise(lines):
    """Check if a polygon defined by line segments is oriented clockwise.
    
    Args:
        lines: Array of line segments representing polygon edges, shape (n, 2, 2).
        
    Returns:
        bool: True if polygon is clockwise, False if counter-clockwise.
    """
    return np.sum(np.cross(lines[:, 0], lines[:, 1], axis=-1)) > 0


def poly_verts_to_lines_append_head(verts):
    """Convert polygon vertices to line segments, closing the polygon.
    
    Appends the first vertex to the end to create a closed polygon.
    
    Args:
        verts: Array of polygon vertices, shape (n, 2).
        
    Returns:
        np.ndarray: Array of line segments, shape (n, 2, 2).
    """
    n_verts = verts.shape[0]
    if n_verts == 0:
        return
    assert n_verts > 1
    verts = np.concatenate([verts, verts[0:1]], axis=0)  # append head to tail
    lines = np.stack([verts[:-1], verts[1:]], axis=1)  # N,2,2
    return lines


def convert_lines_to_vertices(lines):
    """Convert line segments to polygon vertex lists.
    
    Connects line segments end-to-end to form closed polygons.
    
    Args:
        lines: Array of line segments, shape (n, 2, 2).
        
    Returns:
        list: List of polygons, each polygon is a list of vertex coordinates.
    """
    polygons = []
    lines = np.array(lines)

    polygon = None
    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, 0)

        lineID, juncID = np.where(lines == polygon[-1])
        vertex = lines[lineID[0], 1 - juncID[0]]
        lines = np.delete(lines, lineID, 0)

        if vertex in polygon:
            polygons.append(polygon)
            polygon = None
        else:
            polygon.append(vertex)

    return polygons


def read_s3d_floorplan(annos):
    '''
    Parse structured 3D (S3D) annotations to extract floorplan geometry.

    Args:
        annos (dict): Annotation dictionary containing semantics, planes, lines, and junctions.

    Returns:
        tuple: (n_rooms, room_lines, door_lines, window_lines)
            - n_rooms (int): Number of detected rooms.
            - room_lines (np.ndarray): Line segments for room boundaries.
            - door_lines (np.ndarray): Line segments for doors.
            - window_lines (np.ndarray): Line segments for windows.
    '''
    # extract the floor in each semantic for floorplan visualization
    planes = []
    for semantic in annos['semantics']:
        for planeID in semantic['planeID']:
            if annos['planes'][planeID]['type'] == 'floor':
                planes.append({'planeID': planeID, 'type': semantic['type']})

        if semantic['type'] == 'outwall':
            outerwall_planes = semantic['planeID']

    # extract hole vertices
    lines_holes = []
    for semantic in annos['semantics']:
        if semantic['type'] in ['window', 'door']:
            for planeID in semantic['planeID']:
                lines_holes.extend(
                    np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
                )
    lines_holes = np.unique(lines_holes)

    # junctions on the floor
    junctions = np.array([junc['coordinate'] for junc in annos['junctions']])
    junction_floor = np.where(np.isclose(junctions[:, -1], 0))[0]

    # construct each polygon
    polygons = []
    for plane in planes:
        lineIDs = np.where(np.array(annos['planeLineMatrix'][plane['planeID']]))[
            0
        ].tolist()
        junction_pairs = [
            np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist()
            for lineID in lineIDs
        ]
        polygon = convert_lines_to_vertices(junction_pairs)
        polygons.append([polygon[0], plane['type']])
    '''
    outerwall_floor = []
    for planeID in outerwall_planes:
        lineIDs = np.where(np.array(annos['planeLineMatrix'][planeID]))[0].tolist()
        lineIDs = np.setdiff1d(lineIDs, lines_holes)
        junction_pairs = [np.where(np.array(annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs]
        for start, end in junction_pairs:
            if start in junction_floor and end in junction_floor:
                outerwall_floor.append([start, end])

    outerwall_polygon = convert_lines_to_vertices(outerwall_floor)
    polygons.append([outerwall_polygon[0], 'outwall'])
    '''

    junctions = np.array([junc['coordinate'][:2] for junc in annos['junctions']])
    door_lines = []
    window_lines = []
    room_lines = []
    n_rooms = 0
    for (polygon, poly_type) in polygons:
        polygon = junctions[np.array(polygon)] / 1000.0  # mm to meter
        lines = poly_verts_to_lines_append_head(polygon)
        if not is_polygon_clockwise(lines):
            lines = poly_verts_to_lines_append_head(np.flip(polygon, axis=0))
        if poly_type == 'door':
            door_lines.append(lines)
        elif poly_type == 'window':
            window_lines.append(lines)
        else:
            n_rooms += 1
            room_lines.append(lines)

    room_lines = np.concatenate(room_lines, axis=0)
    door_lines = (
        np.zeros((0, 2, 2), float)
        if len(door_lines) == 0
        else np.concatenate(door_lines, axis=0)
    )
    window_lines = (
        np.zeros((0, 2, 2), float)
        if len(window_lines) == 0
        else np.concatenate(window_lines, axis=0)
    )

    return n_rooms, room_lines, door_lines, window_lines


# Origianlly from https://github.com/fuenwang/Equirec2Perspec
# Modifyied in https://github.com/zillow/laser, which we then modified upon
def pano2persp(img, fov, yaw, pitch, roll, size, RADIUS=128):
    '''
    Sample from an equirectangular panorama for a perspective view.

    Args:
        img (np.ndarray): Input panorama image (HxWxC).
        fov (float): Horizontal field of view in radians.
        yaw (float): Yaw angle (rotation around vertical axis, in raduabs).
        pitch (float): Pitch angle (rotation around lateral axis, in radians).
        roll (float): Roll angle (rotation around forward axis, in radians).
        size (tuple): Output perspective image size (height, width).
        RADIUS (int, optional): Sphere radius used for projection. Default is 128.

    Returns:
        np.ndarray: Perspective-view image warped from the panorama.
    '''
    equ_h, equ_w = img.shape[:2]
    equ_cx = (equ_w - 1) / 2.0
    equ_cy = (equ_h - 1) / 2.0

    height, width = size
    wFOV = fov

    f = width / (2 * np.tan(wFOV / 2))
    hFOV = 2 * np.arctan(height / (2 * f))

    c_x = (width - 1) / 2.0
    c_y = (height - 1) / 2.0

    wangle = (np.radians(180) - wFOV) / 2.0
    w_len = 2 * RADIUS * np.sin(wFOV / 2.0) / np.sin(wangle)
    w_interval = w_len / (width - 1)

    hangle = (np.radians(180) - hFOV) / 2.0
    h_len = 2 * RADIUS * np.sin(hFOV / 2.0) / np.sin(hangle)
    h_interval = h_len / (height - 1)
    x_map = np.zeros([height, width], np.float32) + RADIUS
    y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
    z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
    D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = np.zeros([height, width, 3], float)
    xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
    xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
    xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * (yaw - np.radians(180)))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * (-pitch))
    [R3, _] = cv2.Rodrigues(np.dot(R2 @ R1, x_axis) * (-roll))

    xyz = xyz.reshape(height * width, 3).T
    xyz = ((R3 @ R2 @ R1) @ xyz).T
    lat = np.arcsin(xyz[:, 2] / RADIUS)
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    lon = ((lon / np.pi + 1) * equ_cx).reshape(height, width)
    lat = ((-lat / np.pi * 2 + 1) * equ_cy).reshape(height, width)

    persp = cv2.remap(
        img,
        lon.astype(np.float32),
        lat.astype(np.float32),
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_WRAP,
    )
    return persp


def get_intrinsics_from_pano2persp(fov, image_size):
    '''
    Recover camera intrinsic matrix K from pano2persp parameters.
    
    Args:
        fov (float): horizontal field of view in radians
        image_size (tuple): (height, width) of the perspective image

    Returns:
        K (np.ndarray): 3x3 camera intrinsic matrix
    '''
    height, width = image_size
    fov = fov
    f_x = (width / 2) / np.tan(fov / 2)

    # pano2persp calculates vertical FOV as:
    # hFOV = (height / width) * wFOV
    fov_y = (height / width) * fov
    f_y = (height / 2) / np.tan(fov_y / 2)

    c_x = (width - 1) / 2.0
    c_y = (height - 1) / 2.0

    K = np.array([
        [f_x,   0,  c_x],
        [0,   f_y,  c_y],
        [0,     0,   1 ]
    ], dtype=np.float32)

    return K


def get_camera_pose_from_direction(direction, pitch=0):
    '''
    Compute a 4x4 camera-to-world transformation matrix from the direction in the floor plan reference frame and a pitch.
    The convention we use here is 0 degree in floor plan reference frame (+x direction) correspond to yaw angle of 0.
    By defining it this way, we are saying that the initial forward direction in the observation session reference frame should 
        align with the +x direction on the floor plan.

    Args:
        direction (float): Yaw angle in radians in floor plan (world) reference frame.
        pitch (float): Pitch angle in radians (positive looks upward).

    Returns:
        np.ndarray: 4x4 camera-to-world pose matrix.
    '''
    yaw = direction
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi # Normalize it to (-pi, pi)

    # Rotation around Y axis (yaw)
    R_yaw = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [ 0,               1, 0              ],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    # Rotation around camera's local X axis (pitch)
    # negative pitch to match pano2persp (down is positive)
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)],
    ])

    # Total rotation: apply yaw first, then pitch in the local (rotated) frame
    R = R_yaw @ R_pitch

    # Assemble transformation matrix (no translation)
    T = np.eye(4)
    T[:3, :3] = R
    return T


def are_colinear(line1, line2, tol=1e-4):
    """Check if two line segments are approximately colinear.
    
    Args:
        line1: First line segment (shapely LineString).
        line2: Second line segment (shapely LineString).
        tol: Tolerance for colinearity check.
        
    Returns:
        bool: True if lines are colinear within tolerance.
    """
    # Convert to vectors
    p1, p2 = np.array(line1.coords)
    q1, q2 = np.array(line2.coords)

    v1 = p2 - p1
    v2 = q2 - q1

    # Normalize
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    dot_product = np.dot(v1, v2)
    return abs(abs(dot_product) - 1) < tol  # direction is aligned or opposite


def subtract_colinear_doors(roomLines, doorLines, tol=1e-3, overlap_thresh=0.8):
    '''
    Subtracts colinear overlapping door segments from room boundary lines.

    Args:
        roomLines (np.ndarray): Array of room boundary line segments (m x 2 x 2).
        doorLines (np.ndarray): Array of door line segments (n x 2 x 2), grouped in sets of 4.
        tol (float): Distance tolerance for considering segments overlapping. Default is 1e-3.
        overlap_thresh (float): Minimum overlap ratio (unused in current implementation). Default is 0.8.

    Returns:
        tuple:
            - updated_lines (np.ndarray): Room boundary segments with door overlaps removed.
            - remaining_door_lines (np.ndarray): Door line segments that were not subtracted.
    '''
    updated_lines = []
    used_door_indices = set()

    # For each set of room lines in a set of 4, find the two long room lines and two short lines
    assert len(doorLines) % 4 == 0, 'The number of door lines should be multiples of 4'
    long_segment_indices = []
    for i in range(0, len(doorLines), 4):
        group = doorLines[i:i+4]
        
        # Compute lengths of each segment in the group
        lengths = [np.linalg.norm(seg[0] - seg[1]) for seg in group]
        
        # Get indices of the two longest segments in the group
        group_indices = list(range(i, i+4))
        sorted_indices = sorted(zip(group_indices, lengths), key=lambda x: x[1], reverse=True)
        long_indices = [sorted_indices[0][0], sorted_indices[1][0]]
        
        long_segment_indices.extend(long_indices)

    # Remove overlapping door segments from room segments
    for room_seg_coords in roomLines:
        room_seg = LineString(room_seg_coords)
        segments_to_subtract = []

        for i, door_seg_coords in enumerate(doorLines):
            # Skip short segments
            if i not in long_segment_indices:
                continue

            door_seg = LineString(door_seg_coords)
            if room_seg.distance(door_seg) < tol and are_colinear(room_seg, door_seg):
                segments_to_subtract.append(door_seg)
                used_door_indices.add(i)

        if segments_to_subtract:
            diff = room_seg
            for door in segments_to_subtract:
                diff = diff.difference(door, 0.05)

            if diff.is_empty:
                continue
            elif diff.geom_type == 'LineString':
                updated_lines.append(np.array(diff.coords))
            elif diff.geom_type == 'MultiLineString':
                for part in diff.geoms:
                    updated_lines.append(np.array(part.coords))
        else:
            updated_lines.append(room_seg_coords)

    remaining_door_lines = [doorLines[i] for i in range(len(doorLines)) if i not in used_door_indices]

    return np.array(updated_lines), np.array(remaining_door_lines)


def door_connects_two_rooms(door_line, room_lines, tol=1e-3):
    """Check if a door line connects two different room boundaries.
    
    Args:
        door_line: Door line segment (shapely LineString).
        room_lines: Array of room boundary line segments.
        tol: Distance tolerance for considering endpoints near room lines.
        
    Returns:
        bool: True if door connects at least two different room boundaries.
    """
    p1, p2 = map(Point, door_line.coords)

    touching_segments = []
    for i, room_coords in enumerate(room_lines):
        room_line = LineString(room_coords)
        if p1.distance(room_line) < tol or p2.distance(room_line) < tol:
            touching_segments.append(i)

    return len(set(touching_segments)) >= 2


def find_doors_connecting_rooms(roomLines, doorLines, tol=0.05):
    """Find door segments that connect two different room boundaries.
    
    Args:
        roomLines: Array of room boundary line segments.
        doorLines: Array of door line segments.
        tol: Distance tolerance for connection check.
        
    Returns:
        np.ndarray: Array of door segments that connect rooms.
    """
    connecting_doors = []

    for door_coords in doorLines:
        door_line = LineString(door_coords)
        if door_connects_two_rooms(door_line, roomLines, tol):
            connecting_doors.append(door_coords)

    return np.array(connecting_doors)

