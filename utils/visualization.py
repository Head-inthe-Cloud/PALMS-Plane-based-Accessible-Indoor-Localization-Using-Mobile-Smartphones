import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from utils.geometry_utils import Bresenham, get_R_from_orientations, apply_transformation_to_segments, apply_transformation_to_points
from utils.file_io import *
import open3d as o3d

def draw_arrow(point, arrow):
    """Draw an arrow on the current matplotlib plot.
    
    Args:
        point: Starting point as array [x, y].
        arrow: Direction vector as array [dx, dy] (will be normalized).
    """
    p = np.array(point, dtype=float)       # starting point
    a = np.array(arrow, dtype=float)       # direction vector

    if np.linalg.norm(a) > 0:
        a = a / np.linalg.norm(a)          # normalize

    arrow_len = 2  # adjust as needed
    plt.arrow(
        p[0], p[1],
        a[0] * arrow_len, a[1] * arrow_len,
        head_width=2, head_length=2,
        fc='r', ec='r'
    )


def visualize(segments_1=None, 
              segments_2=None, 
              segments_3=None, 
              l1=None, 
              l2=None, 
              L1=None, 
              L2=None, 
              T=None, 
              points_1=None, 
              points_2=None,
              points_group=None,
              point_1=None,
              point_2=None,
              arrow_1=None,
              arrow_2=None,
              heatmap_1 = None,
              heatmap_2 = None,
              img_1 = None,
              img_2 = None,
              title=None, 
              output_path=None,
              block=True):
    """Visualize geometric data including segments, points, heatmaps, and images.
    
    Flexible visualization function that can display multiple types of geometric
    primitives and data overlays on a single plot.
    
    Args:
        segments_1: Line segments to plot in black.
        segments_2: Line segments to plot in purple.
        segments_3: Line segments to plot in orange.
        l1, l2: Line segments to plot in red (optionally transformed by T).
        L1, L2: Line segments to plot in green.
        T: Transformation matrix to apply to l1 and l2.
        points_1: Points to plot as red circles.
        points_2: Points to plot as green circles.
        points_group: List of point groups (up to 4) to plot in different colors.
        point_1: Single point to plot as blue circle.
        point_2: Single point to plot as orange circle.
        arrow_1: Arrow to draw from point_1.
        arrow_2: Arrow to draw from point_2.
        heatmap_1: Heatmap to overlay with plasma colormap.
        heatmap_2: Second heatmap to overlay.
        img_1: Image to display.
        img_2: Second image to display.
        title: Plot title.
        output_path: If provided, save figure to this path instead of displaying.
        block: Whether to block execution when displaying plot.
    """
    # x_values = np.array([x for seg in map_data for x, _ in seg])
    # y_values = np.array([y for seg in map_data for _, y in seg])

    # # Calculate the center of the data
    # center_x = sum(x_values) / len(x_values)
    # center_y = sum(y_values) / len(y_values)

    # Create the plot
    if segments_1 is not None:
        for seg in segments_1:
            plt.plot(*zip(*seg), color='black')

    if segments_2 is not None:
        for seg in segments_2:
            plt.plot(*zip(*seg), color='purple')

    if segments_3 is not None:
        for seg in segments_3:
            plt.plot(*zip(*seg), color='orange')

    if l1 is not None and l2 is not None:
        if T is not None: 
            for seg in apply_transformation_to_segments(np.array([l1, l2]), T):
                plt.plot(*zip(*seg), color='red')
        else: 
            for seg in [l1, l2]:
                plt.plot(*zip(*seg), color='red')


    if L1 is not None and L2 is not None:
        for seg in [L1, L2]:
            plt.plot(*zip(*seg), color='green')

    if l1 is not None and L1 is not None:
        plt.plot(*zip(*l1), color='red')
        plt.plot(*zip(*L1), color='green')
    
    if points_1 is not None:
        plt.plot(*zip(*points_1), 'ro', markersize=3)
    
    if points_2 is not None:
        plt.plot(*zip(*points_2), 'go', markersize=3)

    if points_group is not None:
        assert len(points_group) <= 4, 'We can only handle 4 groups right now'
        colors=['orange', 'blue', 'gray', 'purple']
        for i in range(len(points_group)):
            plt.plot(*zip(*points_group[i]), 'o', color=colors[i], markersize=1)

    
    # if points_1 is not None and points_2 is not None:
    #     for p1, p2 in zip(points_1, points_2):
    #         plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='green')

    if point_1 is not None:
        plt.plot(point_1[0], point_1[1], 'bo', markersize=2)

    if point_2 is not None:
        plt.plot(point_2[0], point_2[1], 'o', color='orange', markersize=2)

    if point_1 is not None and arrow_1 is not None:
        draw_arrow(point_1, arrow_1)

    if point_2 is not None and arrow_2 is not None:
        draw_arrow(point_2, arrow_2)

    if img_1 is not None:
        plt.imshow(img_1, origin='lower')
    
    if img_2 is not None:
        plt.imshow(img_2, origin='lower')

    if heatmap_1 is not None:
        im = plt.imshow(heatmap_1, cmap='plasma', origin='lower', alpha=0.5)
        plt.colorbar(im, label='Heatmap Intensity')
    
    if heatmap_2 is not None:
        im = plt.imshow(heatmap_2, cmap='plasma', origin='lower', alpha=0.5)
        plt.colorbar(im, label='Heatmap Intensity')

    if title is not None:
        plt.title = title

    plt.axis('equal')
    if output_path is not None:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show(block=block)


def show_images(images, titles=None, rotate=False):
    '''
    Display up to 4 images side by side.

    Args:
        images (list of np.ndarray): List of images to display.
        titles (list of str, optional): Optional list of titles.
    '''
    num_images = min(len(images), 4)
    plt.figure(figsize=(5 * num_images, 5))

    for i in range(num_images):
        image = images[i]
        if rotate:
            image = np.rot90(image, k=3)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image)
        if titles and i < len(titles):
            plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def show_pcd(pcd, window_name='Showing a single point cloud with the axes'):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([pcd, frame], window_name=window_name)


def show_pcds(pcds, window_name=None):
    if window_name is None:
        n = len(pcds)
        window_name = f'Showing {n} point clouds'
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries(pcds + [frame], window_name=window_name)


def visualize_pcds(pcd_list, output_path, width=800, height=600):
    # Create an offscreen renderer
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    
    # Set up the scene
    render.scene.set_background([1, 1, 1, 1])  # White background
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = 'defaultUnlit'  # or 'defaultLit' for realistic lighting

    # Add all the point clouds
    for idx, pcd in enumerate(pcd_list):
        render.scene.add_geometry(f'pcd_{idx}', pcd, material)

    # Set the camera nicely to fit all objects
    full_bbox = pcd_list[0].get_axis_aligned_bounding_box()
    for pcd in pcd_list[1:]:
        full_bbox += pcd.get_axis_aligned_bounding_box()

    center = full_bbox.get_center()
    extent = np.linalg.norm(full_bbox.get_extent())
    render.setup_camera(60.0, full_bbox, center)

    # Render the scene
    img = render.render_to_image()

    # Save the image
    o3d.io.write_image(output_path, img)

    # Release resources
    render.release()
    print(f'Saved multiple PCDs visualization to {output_path}')


def visualize_histogram(data, bins=20):
    # Create a histogram
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Data')
    plt.grid(True)

    # Show the plot
    plt.show()


def visualize_orientations(orientations, center=[0, 0], length=10):
    """Convert orientations to line segments for visualization.
    
    Args:
        orientations: List of orientations in radians, range -π/2 to π/2.
        center: Center point [x, y] for the line segments. Defaults to [0, 0].
        length: Half-length of each line segment. Defaults to 10.
        
    Returns:
        np.ndarray: Array of line segments, shape (n, 2, 2).
    """
    segments = []
    for ori in orientations:
        x = np.cos(ori)
        y = np.sin(ori)
        x1 = x * length + center[0]
        y1 = y * length + center[1]
        x2 = -x * length + center[0]
        y2 = -y * length + center[1]
        segments.append([[x1, y1], [x2, y2]])

    return np.array(segments)


def visualize_kernel_overlap(binary_map, kernel, gt_loc, kernel_center):
    '''
    Overlays a kernel on a binary map centered at the ground truth location and visualizes the overlap.

    Args:
        binary_map (np.ndarray): 2D binary map (0: free, 1: occupied).
        kernel (np.ndarray): 2D kernel (e.g., gaussian or shape).
        gt_loc (tuple): (x, y) ground truth location in pixel coordinates.
        kernel_center (tuple): (cx, cy) center of the kernel (e.g., for a 15x15 kernel, it's (7, 7)).
    '''
    map_h, map_w = binary_map.shape
    kh, kw = kernel.shape
    cx, cy = kernel_center
    x, y = int(gt_loc[0]), int(gt_loc[1])

    # Compute where the kernel will go on the map
    top_left_x = x - cx
    top_left_y = y - cy
    bottom_right_x = top_left_x + kw
    bottom_right_y = top_left_y + kh

    # Prepare canvas for overlay
    kernel_mask = np.zeros_like(binary_map, dtype=np.float32)

    # Compute valid region to paste kernel (in case it goes out of bounds)
    x1_k = max(0, -top_left_x)
    y1_k = max(0, -top_left_y)
    x2_k = min(kw, map_w - top_left_x)
    y2_k = min(kh, map_h - top_left_y)

    x1_m = max(0, top_left_x)
    y1_m = max(0, top_left_y)
    x2_m = x1_m + (x2_k - x1_k)
    y2_m = y1_m + (y2_k - y1_k)

    # Place kernel on the canvas
    kernel_mask[y1_m:y2_m, x1_m:x2_m] = kernel[y1_k:y2_k, x1_k:x2_k]

    # Highlight overlap: binary_map * kernel
    overlap = kernel_mask * binary_map

    # Crop for visualization
    crop_radius = max(kh, kw)
    crop_x1 = max(0, x - crop_radius)
    crop_x2 = min(map_w, x + crop_radius)
    crop_y1 = max(0, y - crop_radius)
    crop_y2 = min(map_h, y + crop_radius)

    binary_crop = binary_map[crop_y1:crop_y2, crop_x1:crop_x2]
    kernel_crop = kernel_mask[crop_y1:crop_y2, crop_x1:crop_x2]
    overlap_crop = overlap[crop_y1:crop_y2, crop_x1:crop_x2]

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(binary_crop, cmap='gray', alpha=0.8, origin='lower')
    plt.imshow(kernel_crop, cmap='Blues', alpha=0.3, origin='lower')
    plt.imshow(overlap_crop, cmap='Reds', alpha=0.6, origin='lower')
    plt.scatter([x - crop_x1], [y - crop_y1], c='lime', marker='x', s=100, label='GT')
    plt.title('Kernel Overlap Visualization')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_heatmaps_with_top_locs(
    final_heatmap, resolution=0.1, label=None, binary_map=None, num_top=5, output_path=None
):
    fig, ax = plt.subplots(figsize=(6, 6))
    if binary_map is not None:
        ax.imshow(binary_map, cmap='gray', alpha=0.6, origin='lower')

    im = ax.imshow(final_heatmap, cmap='viridis', origin='lower', alpha=0.6)

    # Convert label to pixel coordinates
    if label is not None:
        label_pix = (label[0] / resolution, label[1] / resolution)
        ax.plot(label_pix[0], label_pix[1], 'o', color='green', markersize=6,
                label='GT Location', alpha=0.6)

    # Top-k points
    flat_indices = np.argpartition(final_heatmap.ravel(), -num_top)[-num_top:]
    top_coords = np.column_stack(np.unravel_index(flat_indices, final_heatmap.shape))
    for y, x in top_coords:
        ax.plot(x, y, 'ro', markersize=4, alpha=0.7)

    ax.set_title(f'Estimated heatmap and top-{num_top} locations')
    ax.legend(loc='best')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

        