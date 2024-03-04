import cv2
import numpy as np
from colmap2poses import load_camera
import os

def get_undistortion(source_path):

    distorted_path = os.path.join(source_path, "distorted")

    cam = load_camera(distorted_path)

    # Camera parameters
    fx, fy, cx, cy = cam.params[0:4]
    k1, k2, p1, p2 = cam.params[4:8]

    undistorted_ref_path = os.listdir(os.path.join(source_path, "images"))[0]
    undistorted_ref_path = os.path.join(source_path, "images", undistorted_ref_path)
    undistorted_ref = cv2.imread(undistorted_ref_path)

    image_ref_path = os.listdir(os.path.join(source_path, "input"))[0]
    image_ref_path = os.path.join(source_path, "input", image_ref_path)
    image_ref = cv2.imread(image_ref_path)

    # Prepare the camera matrix
    K = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]])

    # Distortion coefficients
    D = np.array([k1, k2, p1, p2])

    # Calculate the optimal new camera matrix
    h, w = image_ref.shape[:2]
    
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))

    # Undistort the image using the new camera matrix
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, new_camera_matrix, (w, h), 5)

    return mapx, mapy, roi, undistorted_ref

def undistort(image, mapx, mapy, roi, ref):
    undistorted_image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    
    # Crop the image based on the ROI (Region of Interest)
    x, y, w, h = roi

    # Hard coded for now, TODO: Check how to replicate colmap undistort exactly
    y_pad_1 = 0
    y_pad_2 = 0
    x_pad_1 = 0
    x_pad_2 = 0

    undistorted_image = undistorted_image[y+y_pad_1:y+h+y_pad_2, x-x_pad_1:x+w-x_pad_2]
    resize = cv2.resize(undistorted_image, (ref.shape[1], ref.shape[0]), interpolation = cv2.INTER_AREA)

    return resize

if __name__ == "__main__":
    dataset = "02_Flames"
    mapx, mapy, roi, undistorted_ref = get_undistortion(dataset)
    undistorted_frame = undistort(cv2.imread(dataset +"/input/camera_0001.png"), mapx, mapy, roi, undistorted_ref)

    cv2.imwrite("undistorted.png", undistorted_frame)
    cv2.imwrite("undistorted_ref.png", cv2.imread(dataset +"/images/camera_0001.png"))