# Reference Frame Analysis: align_pcds_by_scale

## What `align_pcds_by_scale` expects

### PCDs
- **Frame:** World frame with **Y-up** (vertical axis = Y).
- **Evidence from `detect_ground_plane` (ground_plane.py):**
  - Uses `points[:, 1]` for vertical extent.
  - `vertical_axis = np.array([0, 1, 0])` (Y is up).
  - Lower parts of the cloud have smaller Y.

### Poses
- **Frame:** Same world frame as the PCDs.
- **Use:** `pose[:3, 3]` is the camera center in world coordinates.
- **Ground scaling:** `default_y = -cam_height` (-1.5 m) = ground plane level in this frame.

### `create_pcd_from_cropped_depth` (used in overlap alignment)
- **Depth unprojection:** Standard OpenCV `(x, y, z)` → then applies `(x, -y, -z)`.
- **Interpretation:** `(x, -y, -z)` converts OpenCV camera (X right, Y down, Z forward) to **ARKit camera** (X right, Y up, Z backward).
- **Pose:** Uses `pcd.transform(pose)` → pose is expected to map this ARKit-like camera frame to world.
- **Conclusion:** This function was written for **ARKit conventions**, not Aria.

---

## Current state (observation_module make_pcd)

### PCD generation
1. Depth unprojected as `(x, y, z)` in **OpenCV** camera frame (X right, Y down, Z forward).
2. `pcd.transform(OPENCV_TO_ARIA_CAM)` → **Aria camera** frame.
3. `pcd.transform(pose)` or `pcd.transform(pose_anchor)` → **Aria world** frame.

### Result
- **PCDs:** In **Aria world** frame (Y-up, consistent with gravity).

### Poses
- **When `use_anchor=False`:** Raw Aria poses (camera-to-world).
- **When `use_anchor=True`:** `pose_anchor` = same rotation, translation `t_i = pose[:3,3] - [gt_x, gt_y, z_ref]`.
- **Frame:** Same world frame as PCDs (Aria world or anchor-relative).

---

## Mismatch

### Ground-based alignment
- `detect_ground_plane` and ground scaling assume Y-up.
- Aria world is Y-up.
- PCDs and poses are in Aria world.
→ Ground-based alignment should be consistent.

### Overlap-based alignment
- `create_pcd_from_cropped_depth` builds points in **ARKit camera** `(x, -y, -z)` then applies the pose.
- The pose is **Aria** camera-to-world.
- Aria camera ≠ ARKit camera → wrong transformation when creating overlap PCDs.
- **Result:** Overlap scale alignment can be incorrect for Aria data.

---

## Summary

| Component              | Expected frame              | Current frame                    |
|------------------------|-----------------------------|----------------------------------|
| PCDs (main pipeline)   | World, Y-up                 | Aria world, Y-up ✓              |
| Poses (main pipeline)  | World, Y-up                 | Aria world ✓                    |
| `create_pcd_from_cropped_depth` | ARKit camera → world | Uses Aria pose on ARKit-like points ✗ |

The overlap path inside `align_pcds_by_scale` is likely wrong because `create_pcd_from_cropped_depth` assumes ARKit-style points and an ARKit-style pose, while the pipeline uses Aria poses.
