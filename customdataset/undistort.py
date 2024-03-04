from convert_video_utils import get_undistortion, undistort
from colmap2poses import load_colmap_data, get_poses
import os
import numpy as np
import cv2

print("Undistorting each frame...")

source_path = "02_Flames"
output = "output"

frames_path = os.path.join(source_path)
cameras = os.listdir(frames_path)

mapx, mapy, roi, undistorted_ref = get_undistortion(source_path)
output_path = os.path.join(source_path, output)
os.makedirs(output_path, exist_ok=True)

for camera in cameras:
    if "camera" not in camera:
        continue
    print("Undistorting frames from camera " + camera + "...")
    frames = os.listdir(os.path.join(frames_path, camera, "images"))

    os.makedirs(os.path.join(output_path, camera), exist_ok=True)
    os.makedirs(os.path.join(output_path, camera, "images"), exist_ok=True)

    for frame in frames:
        undistorted_frame = undistort(cv2.imread(os.path.join(frames_path, camera, "images", frame)), mapx, mapy, roi, undistorted_ref)
        cv2.imwrite(os.path.join(output_path, camera, "images", frame), undistorted_frame)

print("Saving poses_bounds.npy")

poses, pts3d, perm = load_colmap_data(source_path)
save_arr = get_poses(poses, pts3d, perm)

np.save(os.path.join(output_path, 'poses_bounds.npy'), save_arr)

print("Done!")