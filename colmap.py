import os
import shutil
import subprocess
colmap = "C:/Users/Pablo/Downloads/COLMAP-3.9.1-windows-cuda/COLMAP-3.9.1-windows-cuda/COLMAP.bat"

def main(workdir, datatype):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Clean up previous runs
    shutil.rmtree(os.path.join(workdir, "sparse_"), ignore_errors=True)
    shutil.rmtree(os.path.join(workdir, "image_colmap"), ignore_errors=True)
    
    # Run datatype2colmap.py
    subprocess.run(["python", os.path.join("scripts", f"{datatype}2colmap.py"), workdir])
    
    # Clean up colmap directory
    shutil.rmtree(os.path.join(workdir, "colmap"), ignore_errors=True)
    shutil.rmtree(os.path.join(workdir, "colmap", "sparse", "0"), ignore_errors=True)
        
    # Copy necessary files
    shutil.copytree(os.path.join(workdir, "image_colmap"), os.path.join(workdir, "colmap", "images"))
    shutil.copytree(os.path.join(workdir, "sparse_"), os.path.join(workdir, "colmap", "sparse_custom"))
    
    # Extract features
    subprocess.run([colmap, "feature_extractor", "--database_path", os.path.join(workdir, "colmap", "database.db"), "--image_path", os.path.join(workdir, "colmap", "images"), "--SiftExtraction.max_image_size", "4096", "--SiftExtraction.max_num_features", "16384", "--SiftExtraction.estimate_affine_shape", "1", "--SiftExtraction.domain_size_pooling", "1"])
    
    # Update database
    subprocess.run(["python", "database.py", "--database_path", os.path.join(workdir, "colmap", "database.db"), "--txt_path", os.path.join(workdir, "colmap", "sparse_custom", "cameras.txt")])
    
    # Match features
    subprocess.run([colmap, "exhaustive_matcher", "--database_path", os.path.join(workdir, "colmap", "database.db")])
    
    # Triangulate points
    os.makedirs(os.path.join(workdir, "colmap", "sparse", "0"))
    subprocess.run([colmap, "point_triangulator", "--database_path", os.path.join(workdir, "colmap", "database.db"), "--image_path", os.path.join(workdir, "colmap", "images"), "--input_path", os.path.join(workdir, "colmap", "sparse_custom"), "--output_path", os.path.join(workdir, "colmap", "sparse", "0"), "--clear_points", "1"])
    
    # Dense reconstruction
    os.makedirs(os.path.join(workdir, "colmap", "dense", "workspace"))
    subprocess.run([colmap, "image_undistorter", "--image_path", os.path.join(workdir, "colmap", "images"), "--input_path", os.path.join(workdir, "colmap", "sparse", "0"), "--output_path", os.path.join(workdir, "colmap", "dense", "workspace")])
    subprocess.run([colmap, "patch_match_stereo", "--workspace_path", os.path.join(workdir, "colmap", "dense", "workspace")])
    subprocess.run([colmap, "stereo_fusion", "--workspace_path", os.path.join(workdir, "colmap", "dense", "workspace"), "--output_path", os.path.join(workdir, "colmap", "dense", "workspace", "fused.ply")])

if __name__ == "__main__":
    workdir = "D:/capturas/bici2/output"
    datatype = "llff"
    main(workdir, datatype)