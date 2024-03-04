import os

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import minimize

class DepthEstimator:
    def __init__(self, scene, dataset, load_model=False, skip_init=False, **kwargs):
        self.scene = scene
        self.device = kwargs['device']

        # Load depth estimates if they already exist
        stored_depths = dict()
        dir_name = kwargs['depths_path']
        
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        elif os.path.exists(dir_name):
            for file_name in tqdm(os.listdir(dir_name)):
                if file_name.endswith('.npy'):
                    stored_depths[file_name[:-4]] = np.load(os.path.join(dir_name, file_name), allow_pickle=True)

        # Load the model if not all images have been processed
        if len(stored_depths) < len(scene.train_camera) or load_model:
            self.load_model(kwargs['depth_model'])

        if skip_init: return

        test_cams = scene.getTestCameras()
        train_cams = scene.getTrainCameras()

        test_cams_new = []
        train_cams_new = []
        
        for camera in tqdm(test_cams):
            depth = stored_depths.get("test_"+camera.image_name+ str(hash(str(camera.camera_center[0])))[:10])
            if depth is not None:
                camera.estimated_depth = depth
            else:
                depth = self.estimate(camera, dataset)
                camera.estimated_depth = depth
                np.save(os.path.join(dir_name, "test_"+camera.image_name +str(hash(str(camera.camera_center[0])))[:10] +'.npy'), depth)
            test_cams_new.append(camera)
        
        for camera in tqdm(train_cams):
            depth = stored_depths.get("train_"+camera.image_name + str(hash(str(camera.camera_center[0])))[:10])
            if depth is not None:
                camera.estimated_depth = depth
            else:
                depth = self.estimate(camera, dataset)
                camera.estimated_depth = depth
                np.save(os.path.join(dir_name, "train_"+camera.image_name + str(hash(str(camera.camera_center[0])))[:10]+ '.npy'), depth)
            train_cams_new.append(camera)
        
        self.scene.setTestCameras(test_cams_new)
        self.scene.setTrainCameras(train_cams_new)
    
    def get_scene(self):
        return self.scene

    def load_model(self, depth_model="depth_anything"):
        if depth_model == "midas":
            self.depth_model = MidasModel(self.device)
        elif depth_model == "zoe":
            self.depth_model = ZoeDepthModel(self.device)
        elif depth_model == "depth_anything":
            self.depth_model = DepthAnythingModel(self.device)
        else:
            raise ValueError("Unknown depth model type")

    def estimate(self, camera, dataset):
        "Returns a SfM scale-matched dense depth map for a chosen camera"

        # Estimate sparse and dense depth (and reprojection error) maps
        x_2d, y_2d, z = self._estimate_sparse(camera, dataset)
        D_dense = self._estimate_dense(camera) ##

        # MiDaS depth maps are in disparity space
        if self.depth_model.name == ["midas"]:
            D_dense = self._match_scale_disparity(D_dense, x_2d, y_2d, z)
        elif self.depth_model.name in ["zoe", "depth_anything"]:
            D_dense = self._match_scale(D_dense, x_2d, y_2d, z)

        return D_dense

    def _estimate_dense(self, camera):
        "Use a monocular depth estimation model to estimate depth"
        if self.depth_model is None:
            raise ValueError("No depth model loaded")
        return self.depth_model.predict(camera)

    def _estimate_sparse(self, camera, dataset):
        "Returns a depth map estimated from COLMAP data"
        # Get the visible 3D points for the chosen camera
        #ids = torch.as_tensor(camera.visible_point_ids).to(self.device)
        #ids = range(len(self.scene.points.points.shape[0]))
        #xyz_world, _, errors = dataset.pcd.get_points(ids)
        xyz_world = torch.as_tensor(self.scene.points.points).to(self.device)
        
        # Transform world points to camera points 
        #view_mat = camera.world_view_transform.to(self.device).float()
        R = torch.as_tensor(camera.R).to(self.device).float()
        t = torch.as_tensor(camera.T).to(self.device).float()
        xyz_cam = torch.matmul(R, xyz_world.t()) + t[:3, np.newaxis]
        xyz_cam = xyz_cam.t()
        xyz_cam = xyz_cam.cpu().numpy()
        
        # Normalize view coordinates
        z = xyz_cam[:,2]
        x = xyz_cam[:,0] / z
        y = xyz_cam[:,1] / z
        
        # Convert to image coordinates
        image_width, image_height = camera.image_width, camera.image_height
        f_x = camera.FoVx
        f_y = camera.FoVy
        c_x = image_width / 2
        c_y = image_height / 2
        x_2d = np.round(x * f_x + c_x).astype(np.int32)
        y_2d = np.round(y * f_y + c_y).astype(np.int32)

        return x_2d, y_2d, z

    def _match_scale_disparity(self, D_disparity, D_sparse, E_sparse):
        "Matches the scale in the provided disparity map to that in the sparse map"
        i, j = D_sparse.row, D_sparse.col
        z_dense_inv = torch.as_tensor(D_disparity[i,j].data).to(self.device)
        z_sparse_inv = 1. / torch.as_tensor(D_sparse.data).to(self.device)
        e_sparse = torch.as_tensor(E_sparse.data).to(self.device)

        def func(args):
            s, t = args[0], args[1]
            z_dense_inv_adj = s * z_dense_inv + t
            return  (z_sparse_inv - z_dense_inv_adj).abs().mean().cpu().numpy()
            #return (1/e_sparse * (z_sparse_inv - z_dense_inv_adj)).abs().mean().cpu().numpy()
        res = minimize(func, x0=[-0.5,3], method='Nelder-Mead')

        s, t = res.x
        D_dense = 1. / (s * D_disparity + t)
        return D_dense

    def _match_scale(self, D_dense, x_2d, y_2d, z):
        "Matches the scale in the provided metric depth map to that in the sparse map"

        z_dense = torch.as_tensor(D_dense).to(self.device)
        z_dense_selection = z_dense[y_2d, x_2d] # This or x_2d, y_2d?
        z = torch.as_tensor(z).to(self.device)
        
        def func(args):
            s, t = args[0], args[1]
            z_dense_adj = s * z_dense_selection + t
            return (z - z_dense_adj).abs().mean().cpu().numpy()
        #res = minimize(func, x0=[-0.5,3], method='Nelder-Mead')

        #final_error = func(res.x)
        #s, t = res.x
        #s = np.clip(s, 0.6, 2.1)
        #t = np.clip(t, -20, 20)
        s = 1.54
        t= 13.39

        D_dense = s * D_dense + t
        return D_dense


class ZoeDepthModel:
    def __init__(self, device):
        self.name = "zoe"
        self.model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_N", pretrained=True)
        self.model.to(device)

    def predict(self, camera):
        img = (camera.original_image.cpu().numpy()*255).astype(np.uint8)
        img = img.transpose(1, 2, 0)

        # Monkey patch interpolation function (see https://github.com/isl-org/ZoeDepth/pull/60#issuecomment-1894272730)
        # TODO: Use forked ZoeDepth repo with merged PR
        original_interpolate = F.interpolate
        def patched_interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
            if size is not None:
                size = tuple(int(s) for s in size)
            return original_interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor)

        F.interpolate = patched_interpolate
        depth = self.model.infer_pil(img)
        F.interpolate = original_interpolate

        return depth


class DepthAnythingModel:
    def __init__(self, device):
        from Depth_Anything.depth_anything.dpt import DepthAnything
        from Depth_Anything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
        from torchvision.transforms import Compose

        self.model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vitl14")
        self.transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
                ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
            ])

    def predict(self, camera):
        img = camera.get_original_image().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = img.transpose(1, 2, 0)
        
        img = self.transform({'image': img})['image']
        img = torch.from_numpy(img).unsqueeze(0)
        
        width, height = camera.width, camera.height
        depth = self.model(img)
        depth = F.interpolate(depth.unsqueeze(1), size=(height, width), mode="bicubic", align_corners=False).squeeze(1)
        depth = depth.cpu().numpy()
        return depth


class MidasModel:
    def __init__(self, device):
        self.name = "midas"
        self.device = device
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self.model.to(device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform

    def predict(self, camera):
        img = camera.get_original_image().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        input_batch = self.transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        output = prediction.cpu().numpy()
        return output