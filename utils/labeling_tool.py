import argparse
import os
import sys
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from utils.file_io import *
from utils.dataset_utils import pano2persp
from utils.camera_utils import extract_yaw_from_pose
from utils.geometry_utils import get_R_from_orientations, apply_transformation_to_points, apply_transformation_to_segments

'''
This file contains the following tools:
1) tool for creating pose labels for the observations
2) tool for stretching the raw panoramas from iPhone, so they align better with observation data from ARKit sessions
3) tool for "rolling" the panoramas such that the panorama aligns with the ARKit observation session reference frame
'''
class LabelingTool:
    """Interactive tool for labeling observation poses on floor plans.
    
    Allows users to drag, rotate, and scale observed plane segments to align
    with floor plan maps for ground truth labeling.
    """
    def __init__(self):
        """Initialize the labeling tool."""
        self.label_path = None

        self.x = 0
        self.y = 0
        self.theta = 0
        self.dragging = False
        self.last_mouse_pos = None

        self.fig, self.ax = None, None
        self.floor_plan = None
        self.planes = None
        self.pcd_planes = None

    def update_plot(self):
        """Update the plot visualization."""
        self.ax.clear()
        for seg in self.floor_plan:
            self.ax.plot(*zip(*seg), color='black')
        for seg in self.planes:
            self.ax.plot(*zip(*seg), color='orange')

        self.ax.plot(self.x, self.y, 'bo', label='Labeled Position')
        self.ax.axis('equal')
        self.ax.grid(True)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title(f'Showing label\n x = {self.x}, y = {self.y}, θ = {np.rad2deg(self.theta)} (degrees)')
        self.ax.legend()
        self.fig.canvas.draw()

    def scale_planes(self, event, scale=1, update_plot=True):
        """Scale planes around the current label position.
        
        Args:
            event: Matplotlib event (unused, for button callback compatibility).
            scale: Scaling factor. Defaults to 1.
            update_plot: If True, refresh the plot after scaling. Defaults to True.
        """
       # Translate to origin
        T_t1 = np.array([
            [1, 0, -self.x],
            [0, 1, -self.y],
            [0, 0,  1]
        ])

        # Scale
        T_s = np.array([
            [scale, 0,     0],
            [0,     scale, 0],
            [0,     0,     1]
        ])

        # Translate back
        T_t2 = np.array([
            [1, 0, self.x],
            [0, 1, self.y],
            [0, 0, 1]
        ])

        # Final transformation
        T = T_t2 @ T_s @ T_t1
        self.planes = apply_transformation_to_segments(self.planes, T)
        point = np.array([self.x, self.y])
        self.x, self.y = apply_transformation_to_points(point.reshape((1, -1)), T)[0]
        
        if update_plot:
            self.update_plot()

    def rotate_planes(self, event, delta=5, update_plot=True):
        """Rotate planes around the current label position.
        
        Args:
            event: Matplotlib event (unused, for button callback compatibility).
            delta: Rotation angle in degrees. Defaults to 5.
            update_plot: If True, refresh the plot after rotation. Defaults to True.
        """
        self.theta += np.radians(delta)  # Rotate by x degrees
        self.theta %= 2 * np.pi

        # Compute rotation transformation
        T_t = np.eye(3)
        T_t[0, 2] = -self.x
        T_t[1, 2] = -self.y

        R, _ = get_R_from_orientations(theta=self.theta)  
        T_R = np.eye(3)
        T_R[:2, :2] = R
        T_R[:2, 2] = T_t[:2, 2] * -1

        T = np.matmul(T_R, T_t)
        self.planes = apply_transformation_to_segments(self.planes, T)
        self.T = T @ self.T

        if update_plot:
            self.update_plot()

    def on_mouse_press(self, event):
        """Handle mouse press event for dragging."""
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            self.dragging = True
            self.last_mouse_pos = (event.xdata, event.ydata)

    def on_mouse_release(self, event):
        """Handle mouse release event for stopping dragging."""
        if event.button == 1:
            self.dragging = False

    def on_mouse_move(self, event):
        """Handle mouse movement while dragging."""
        if self.dragging and event.xdata is not None and event.ydata is not None:
            dx = event.xdata - self.last_mouse_pos[0]
            dy = event.ydata - self.last_mouse_pos[1]
            self.last_mouse_pos = (event.xdata, event.ydata)

            self.x += dx
            self.y += dy

            # Apply translation transformation
            T = np.eye(3)
            T[0, 2] = dx
            T[1, 2] = dy
            self.planes = apply_transformation_to_segments(self.planes, T)

            self.update_plot()


    def load_planes(self, planes):
        """Load plane segments and apply current transformation.
        
        Args:
            planes: Array of plane segments to load.
        """
        self.planes = apply_transformation_to_segments(planes, self.T)
        

    def load_label(self, label_path=None, label=None):
        """Load ground truth label and apply transformation to planes.
        
        Args:
            label_path: Path to label file (x, y, theta format).
            label: Label as tuple (x, y, theta) or list [x, y, theta].
        """
        assert label is not None or label_path is not None, 'Provide either the label or the label path'

        self.label_path = label_path
        if label is None:
            self.x, self.y, self.theta = load_label(label_path)
        else:
            self.x, self.y, self.theta = label

        
        # Apply transformations using label data
        T_t = np.eye(3)
        T_t[0, 2] = self.x
        T_t[1, 2] = self.y

        R, _ = get_R_from_orientations(theta=self.theta)
        T_R = np.eye(3)
        T_R[:2, :2] = R

        self.T = np.matmul(T_t, T_R)
        self.planes = apply_transformation_to_segments(self.planes, self.T)
    
    def apply_transformation(self, T):
        """Apply a transformation matrix to the current label and planes.
        
        Args:
            T: 3x3 homogeneous transformation matrix.
        """
        self.T = np.matmul(T, self.T) # update current transform
        obs_point = np.array([self.x, self.y])
        obs_point = apply_transformation_to_points(obs_point.reshape((1, -1)), T)[0]
        self.x, self.y = obs_point
        rotation_matrix = T[:2, :2]
        angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        self.theta += angle
        self.theta %= 2 * np.pi

        self.planes = apply_transformation_to_segments(self.planes, T)


    def view_label(self, output_path=None):
        """Visualize the current label on the floor plan.
        
        Args:
            output_path: If provided, save figure to this path instead of displaying.
        """
        # Read label data from the output file
        if self.x is None:
            self.load_label(self.label_path)

        # Visualization
        fig, ax = plt.subplots()
        for seg in self.floor_plan:
            ax.plot(*zip(*seg), color='black')

        for seg in self.planes:
            ax.plot(*zip(*seg), color='orange')

        ax.plot(self.x, self.y, 'bo', label='Labeled Position')
        ax.axis('equal')
        ax.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Label at {self.label_path}\n x = {self.x}, y = {self.y}, θ = {np.rad2deg(self.theta)}°')
        ax.legend()
        if output_path is None:
            plt.show()
        else:
            plt.savefig(output_path)
            plt.close()

    def save_data(self, event):
        """Save the current label to the output file."""
        with open(self.label_path, 'w') as f:
            f.write(f'{self.x},{self.y},{self.theta}\n')

        print(f'Label saved to {self.label_path}')

    def start_labeling(self):
        """Start the interactive labeling interface."""
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        self.update_plot()

        # Enlarge
        ax_scale = plt.axes([0.28, 0.05, 0.1, 0.075])
        btn_scale = Button(ax_scale, 'Scale up')
        btn_scale.on_clicked(lambda event: self.scale_planes(event, scale=1.1))

        # Rotate Clockwise Button (1 degrees)
        ax_rotate_cw_1 = plt.axes([0.40, 0.05, 0.1, 0.075])
        btn_rotate_cw_1 = Button(ax_rotate_cw_1, 'Rotate -1°')
        btn_rotate_cw_1.on_clicked(lambda event: self.rotate_planes(event, delta=-1))

        # Rotate Counter Clockwise Button (1 degrees)
        ax_rotate_ccw_1 = plt.axes([0.52, 0.05, 0.1, 0.075])
        btn_rotate_ccw_1 = Button(ax_rotate_ccw_1, 'Rotate 1°')
        btn_rotate_ccw_1.on_clicked(lambda event: self.rotate_planes(event, delta=1))

        # Rotate Clockwise Button (20 degrees)
        ax_rotate_20 = plt.axes([0.64, 0.05, 0.1, 0.075])
        btn_rotate_20 = Button(ax_rotate_20, 'Rotate 20°')
        btn_rotate_20.on_clicked(lambda event: self.rotate_planes(event, delta=20))

        # Rotate 180 Button (180 degrees)
        ax_rotate_180 = plt.axes([0.76, 0.05, 0.1, 0.075])
        btn_rotate_180 = Button(ax_rotate_180, 'Rotate 180°')
        btn_rotate_180.on_clicked(lambda event: self.rotate_planes(event, delta=180))

        # Save Button
        ax_save = plt.axes([0.88, 0.05, 0.1, 0.075])
        btn_save = Button(ax_save, 'Save Data')
        btn_save.on_clicked(self.save_data)

        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        plt.show()


class PanoStretchingTool:
    """Interactive tool for stretching panorama images to align with ARKit frames."""
    def __init__(self, data_dir):
        """Initialize the panorama stretching tool.
        
        Args:
            data_dir: Root directory containing observation sessions.
        """
        self.obs_paths = sorted(glob(os.path.join(data_dir, '**/Session*'), recursive=True))
        # Skip the ones that already have stretched pano
        self.obs_paths = [obs_path for obs_path in self.obs_paths if not os.path.exists(os.path.join(obs_path, 'pano_stretched.png'))]

        self.data_idx = 0
        self.frame_idx = 0
        self.pano_image = None
        self.arkit_images = None
        self.arkit_poses = None
        self.labels = []
        self.current_line = None
        self.scale = 5  # Downscaling factor

        self.fig, self.axs = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.subplots_adjust(bottom=0.25)

        self.skip_ax = self.fig.add_axes([0.05, 0.05, 0.1, 0.075])
        self.accept_ax = self.fig.add_axes([0.17, 0.05, 0.1, 0.075])
        self.finish_ax = self.fig.add_axes([0.29, 0.05, 0.1, 0.075])
        self.skip_btn = Button(self.skip_ax, 'Skip')
        self.accept_btn = Button(self.accept_ax, 'Accept')
        self.finish_btn = Button(self.finish_ax, 'Finish')

        self.skip_btn.on_clicked(self.skip_frame)
        self.accept_btn.on_clicked(self.accept_label)
        self.finish_btn.on_clicked(self.finish_obs)

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.pending_x = None

    def load_obs(self):
        """Load observation data for current session."""
        obs_path = self.obs_paths[self.data_idx]
        data = load_obs(obs_path, data_keys=['images', 'poses', 'pano'])
        self.arkit_images = data['images']
        self.arkit_poses = data['poses']
        self.pano_image = data['pano']
        self.labels = []

    def display_frame(self):
        """Display current ARKit frame and panorama for alignment."""
        self.axs[0].clear()
        self.axs[1].clear()

        img = self.arkit_images[self.frame_idx]
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        small_img = cv2.resize(img, (img.shape[1] // self.scale, img.shape[0] // self.scale))
        small_pano = cv2.resize(self.pano_image, (self.pano_image.shape[1] // self.scale, self.pano_image.shape[0] // self.scale))

        self.axs[0].imshow(small_img)
        self.axs[0].axvline(small_img.shape[1] // 2, color='red', linestyle='--')
        self.axs[0].set_title(f'ARKit Frame {self.frame_idx}')

        self.axs[1].imshow(small_pano)
        self.axs[1].set_title('Click on center of ARKit view')

        for ax in self.axs:
            ax.axis('off')

        self.axs[0].set_position([0.05, 0.2, 0.2, 0.7])
        self.axs[1].set_position([0.3, 0.2, 0.65, 0.7])

        self.fig.canvas.draw_idle()

    def onclick(self, event):
        """Handle mouse click on panorama to mark alignment point."""
        if event.inaxes != self.axs[1]:
            return
        x = int(event.xdata)
        self.pending_x = x * self.scale
        if self.current_line:
            self.current_line.remove()
        self.current_line = self.axs[1].axvline(x=x, color='blue', linestyle='--')
        self.fig.canvas.draw_idle()

    def accept_label(self, event):
        """Accept the current alignment point and move to next frame."""
        if self.pending_x is None:
            print('⚠️ No point selected to accept.')
            return
        pose = self.arkit_poses[self.frame_idx]
        self.labels.append((self.pending_x, pose))
        print(f'✔️ Accepted x={self.pending_x} for frame {self.frame_idx}')
        self.pending_x = None
        self.next_frame(None)

    def skip_frame(self, event):
        """Skip the current frame without labeling."""
        print(f'⚠️ Skipping frame {self.frame_idx}')
        self.next_frame(None)

    def next_frame(self, event):
        """Move to the next frame in the observation."""
        if self.frame_idx < len(self.arkit_images) - 1:
            self.frame_idx += 1
            self.current_line = None
            self.pending_x = None
            self.display_frame()
        else:
            print('Reached last frame in this observation.')

    def finish_obs(self, event):
        """Finish current observation, refine labels, and save stretched panorama."""
        print('✅ Calibration finished. Refining x-coordinates...')
        self.labels = sorted(self.labels, key=lambda label: label[0])
        refined = self.refine_labels()
        print('Before')
        print([data[0] for data in self.labels])
        print('After')
        print([data[0] for data in refined])

        self.visualize_stretch([data[0] for data in self.labels], [data[0] for data in refined])

        self.data_idx += 1
        if self.data_idx < len(self.obs_paths):
            self.frame_idx = 0
            self.load_obs()
            self.display_frame()
        else:
            print('🏁 All observations labeled.')

    def refine_labels(self):
        """Refine alignment labels using yaw angles from poses.
        
        Uses pose yaw differences to correct alignment x-coordinates.
        
        Returns:
            list: List of refined (x, pose) tuples.
        """
        corrected = [self.labels[0]]
        scale = 6000 / 180.0

        for i in range(1, len(self.labels)):
            x_prev, pose_prev = corrected[-1]
            x_curr, pose_curr = self.labels[i]

            yaw_prev = np.rad2deg(extract_yaw_from_pose(pose_prev))
            yaw_curr = np.rad2deg(extract_yaw_from_pose(pose_curr))
            dyaw = (yaw_curr - yaw_prev + 180) % 360 - 180

            x_new = x_prev + dyaw * scale * -1   # Multiply by -1 because +x direction correspond to -yaw
            corrected.append((int(round(x_new)), pose_curr))

        return corrected
    

    def visualize_stretch(self, x_before, x_after):
        """Create and save stretched panorama based on alignment labels.
        
        Args:
            x_before: List of original x-coordinates.
            x_after: List of refined x-coordinates.
        """
        H, W = self.pano_image.shape[:2]
        new_W = int(x_after[-1]) + (W - int(x_before[-1]))
        stretched_pano = np.zeros((H, new_W, 3), dtype=self.pano_image.dtype)

        # Fill before the first segment
        if x_before[0] > 0 and x_after[0] > 0:
            stretched_pano[:, :int(x_after[0])] = cv2.resize(
                self.pano_image[:, :int(x_before[0])],
                (int(x_after[0]), H))

        # Fill stretched segments
        for i in range(len(x_before) - 1):
            src_start, src_end = int(x_before[i]), int(x_before[i + 1])
            dst_start, dst_end = int(x_after[i]), int(x_after[i + 1])
            segment = self.pano_image[:, src_start:src_end]
            segment_resized = cv2.resize(segment, (dst_end - dst_start, H))
            stretched_pano[:, dst_start:dst_end] = segment_resized

        # Fill remaining part after last segment
        if x_before[-1] < W - 1:
            src_remaining = self.pano_image[:, int(x_before[-1]):]
            dst_remaining_start = int(x_after[-1])
            remaining_width = new_W - dst_remaining_start

            if src_remaining.shape[1] > remaining_width:
                src_remaining = cv2.resize(src_remaining, (remaining_width, H))
            stretched_pano[:, dst_remaining_start:dst_remaining_start + src_remaining.shape[1]] = src_remaining

        # Save data
        obs_path = self.obs_paths[self.data_idx]
        pano_save_path = os.path.join(obs_path, 'pano_stretched.png')
        cv2.imwrite(pano_save_path, cv2.cvtColor(stretched_pano, cv2.COLOR_RGB2BGR))
        print(f'New pano saved to {pano_save_path}')

        # For visualizations:
        if False:
            img_before = self.pano_image.copy()
            for x in x_before:
                cv2.line(img_before, (int(x), 0), (int(x), H - 1), (0, 255, 0), 2)
            img_after = stretched_pano.copy()
            for x in x_after:
                cv2.line(img_after, (int(x), 0), (int(x), H - 1), (0, 0, 255), 2)

            cv2.imwrite('./results/pano_with_lines_before.png', cv2.cvtColor(img_before, cv2.COLOR_RGB2BGR))
            cv2.imwrite('./results/pano_with_lines_after.png', cv2.cvtColor(img_after, cv2.COLOR_RGB2BGR))

    def start(self):
        """Start the panorama stretching tool interface."""
        self.load_obs()
        self.display_frame()
        plt.show()


class PanoAngularAlignmentTool: 
    """
    Interactive tool for adjusting and aligning panorama images with perspective frames (from ARKit observation session)
    in a dataset. The tool allows the user to visually align frames with their corresponding
    panoramas by applying yaw offsets and saving the corrected panorama.
    The corrected panorama should align with the floor plan reference frame, where the center of the panorama corrspond to +y direction in FP reference frame.

    Workflow:
        - Loads observation data containing images, poses, and equirectangular panoramas.
        - Displays a perspective frame next to the corresponding extracted panorama view.
        - Provides interactive buttons to shift the panorama alignment (large or fine adjustments).
        - Allows stepping through frames and finishing alignment for each observation.
        - Saves horizontally shifted panorama images (`pano_equirectangular_shifted.png`).

    Args:
        data_dir (str): Path to the dataset directory containing observation sessions.

    Attributes:
        obs_paths (list[str]): Paths to all sessions with panoramas to align.
        data_idx (int): Current observation index.
        frame_idx (int): Current frame index within the observation.
        pano_image (np.ndarray): Loaded equirectangular panorama.
        images (list[np.ndarray]): Frame images from the observation.
        poses (list[np.ndarray]): Camera poses associated with frames.
        delta_theta (float): Current accumulated yaw adjustment in radians.
        data_theta (float): Ground truth yaw label from the dataset, in radians.
        fig, axs: Matplotlib figure and axes for visualization.
        Buttons: Interactive matplotlib Button widgets for shifting and navigation.

    Methods:
        load_obs(): Loads the current observation data.
        display_frame(): Shows the current frame and its aligned panorama view.
        shift_left_large(event): Apply a large left shift (-20°).
        shift_right(event): Apply a medium right shift (+1°).
        shift_right_small(event): Apply a fine right shift (+0.1°).
        next_frame(event): Move to the next frame in the current observation.
        prev_frame(): Move to the previous frame in the current observation.
        change_obs(direction): Move to the next or previous observation.
        shift_pano(): Apply the accumulated yaw offset to the panorama and save.
        finish_obs(event): Save the aligned panorama and move to the next observation.
        start_labeling(): Launch the interactive alignment interface.
    """
    def __init__(self, data_dir):
        """Initialize the panorama angular alignment tool.
        
        Args:
            data_dir: Root directory containing observation sessions.
        """
        self.obs_paths = glob(os.path.join(data_dir, '**/Session*'), recursive=True)
        # Skip the ones that already have shifted pano
        self.obs_paths = [obs_path for obs_path in self.obs_paths if os.path.exists(os.path.join(obs_path, 'pano_equirectangular.png'))]
        self.obs_paths = [obs_path for obs_path in self.obs_paths if not os.path.exists(os.path.join(obs_path, 'pano_equirectangular_shifted.png'))]
        
        self.fov_deg = 56
        self.fov = np.radians(56)
        self.data_idx = 0
        self.frame_idx = 0
        self.pano_image = None
        self.images = None
        self.poses = None
        self.delta_theta = 0.0  # user-adjusted yaw offset
        self.data_theta = 0.0

        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.subplots_adjust(bottom=0.25)

        # Shift ← and → buttons
        self.button_left_ax = self.fig.add_axes([0.05, 0.05, 0.1, 0.075])
        self.button_right_ax = self.fig.add_axes([0.17, 0.05, 0.1, 0.075])
        self.button_right_small_ax = self.fig.add_axes([0.29, 0.05, 0.1, 0.075])
        self.button_next_frame_ax = self.fig.add_axes([0.50, 0.05, 0.1, 0.075])
        self.button_finish_ax = self.fig.add_axes([0.62, 0.05, 0.1, 0.075])

        # Buttons
        self.button_left = Button(self.button_left_ax, 'Shift left')
        self.button_right = Button(self.button_right_ax, 'Shift right')
        self.button_right_small = Button(self.button_right_small_ax, '→ 0.1°')
        self.button_next_frame = Button(self.button_next_frame_ax, 'Next Frame')
        self.button_finish = Button(self.button_finish_ax, 'Finish Obs')

        # Button callbacks
        self.button_left.on_clicked(self.shift_left_large)
        self.button_right.on_clicked(self.shift_right)
        self.button_right_small.on_clicked(self.shift_right_small)
        self.button_next_frame.on_clicked(self.next_frame)
        self.button_finish.on_clicked(self.finish_obs)

    def load_obs(self):
        """Load observation data for panorama angular alignment."""
        obs_path = self.obs_paths[self.data_idx]
        data = load_obs(obs_path, data_keys=['images', 'poses', 'label', 'pano_equirectangular'])
        self.images = data['images']
        self.poses = data['poses']
        self.data_theta = data['label'][2]
        self.pano_image = data['pano_equirectangular']

    def shift_left_large(self, event):
        """Apply large left shift (-20°) to panorama alignment."""
        self.delta_theta -= np.radians(20.0)
        self.display_frame()

    def shift_right(self, event):
        """Apply medium right shift (+1°) to panorama alignment."""
        self.delta_theta += np.radians(1.0)
        self.display_frame()


    def display_frame(self):
        """Display current frame and aligned panorama view for angular alignment."""
        self.axs[0].clear()
        self.axs[1].clear()

        img = self.images[self.frame_idx]
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        pose = self.poses[self.frame_idx]

        # Extract yaw from pose (assume rotation matrix)
        yaw = extract_yaw_from_pose(pose)
        aligned_yaw = yaw + self.data_theta + self.delta_theta
        
        # Sample perspective image from the panorama at aligned_yaw
        pano_angle = -aligned_yaw-np.radians(90)
        persp_img = pano2persp(self.pano_image, self.fov, pano_angle, 0, 0, (640, 480))

        self.axs[0].imshow(img)
        self.axs[0].axvline(img.shape[1] // 2, color='red')
        self.axs[0].set_title(f'Original Frame {self.frame_idx}')

        self.axs[1].imshow(persp_img)
        self.axs[1].axvline(persp_img.shape[1] // 2, color='red')
        self.axs[1].set_title(f'Pano View (yaw={aligned_yaw:.1f}°)')

        for ax in self.axs:
            ax.axis('off')
        self.fig.canvas.draw_idle()

    def start_labeling(self):
        """Start the interactive panorama alignment interface."""
        self.load_obs()
        self.display_frame()
        plt.show()

    def change_obs(self, direction='forward'):
        """Move to next or previous observation.
        
        Args:
            direction: 'forward' to move to next, 'backward' to move to previous.
        """
        if direction == 'forward':
            self.data_idx = min(self.data_idx + 1, len(self.obs_paths) - 1)
        elif direction == 'backward':
            self.data_idx = max(self.data_idx - 1, 0)
        self.frame_idx = 0
        self.load_obs()
        self.display_frame()

    def prev_frame(self):
        """Move to the previous frame in the observation."""
        if self.frame_idx > 0:
            self.frame_idx -= 1
            self.display_frame()

    def next_frame(self, event):
        """Move to the next frame in the panorama angular alignment."""
        if self.frame_idx < len(self.images) - 1:
            self.frame_idx += 1
            self.display_frame()

    def shift_right_small(self, event):
        """Apply fine right shift (+0.1°) to panorama alignment."""
        self.delta_theta += np.radians(0.1)
        self.display_frame()

    def shift_pano(self):
        """Shift the panorama horizontally by the accumulated delta_theta.
        
        Applies a circular horizontal shift (wrap-around) and resets delta_theta.
        Saves the shifted panorama to disk.
        """
        if self.pano_image is None:
            print('⚠️ No pano image loaded.')
            return

        h, w = self.pano_image.shape[:2]

        # Convert degrees to pixel offset (assuming equirectangular panorama spans 360°)
        pixels_per_radians = w / (2*np.pi)
        dx = int(round(self.delta_theta * pixels_per_radians))

        # Perform horizontal circular shift
        new_pano_image = np.roll(self.pano_image, shift=dx, axis=1)

        new_pano_path = os.path.join(self.obs_paths[self.data_idx], 'pano_equirectangular_shifted.png')
        cv2.imwrite(new_pano_path, cv2.cvtColor(new_pano_image, cv2.COLOR_RGB2BGR))
        print(f'Shifted Pano saved at {new_pano_path}, Angle shifted = {np.rad2deg(self.delta_theta)} degrees')

        # Reset delta theta
        self.delta_theta = 0.0

    def finish_obs(self, event):
        """Finish current panorama alignment and move to next observation."""
        self.shift_pano()
        self.change_obs('forward')