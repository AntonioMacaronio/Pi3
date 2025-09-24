import torch
import time
import viser
import tyro
import numpy as np
from scipy.spatial.transform import Rotation as R
from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor
from pi3.utils.geometry import depth_edge
import viser.transforms as vtf
import math
import cv2

def main(ckpt: str = "model.safetensors", video_path: str = "video.mp4"):
    server = viser.ViserServer()
    
    # Debug CUDA setup
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Number of GPUs:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name())
    
    # Clear any CUDA environment variables that might cause issues
    import os
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print("CUDA_VISIBLE_DEVICES:", os.environ['CUDA_VISIBLE_DEVICES'])
    
    # Force CPU for now to avoid the CUDA issue
    device = torch.device("cpu")
    print(f"Using device: {device}")
    # load our pi3 model
    print("Loading model...")
    if ckpt is not None:
        model = Pi3().to(device).eval()
        if ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(ckpt)
        else:
            weight = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(weight)

    # load our video
    imgs = load_images_as_tensor(video_path, interval=1).to(device) # (N, 3, H, W)
    
    print("Running model inference...")
    # Use float32 for CPU, or check CUDA capability if available
    if device.type == 'cuda' and torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32 if device.type == 'cpu' else torch.float16
        
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.amp.autocast('cuda', dtype=dtype):
                res = model(imgs[None]) # Add batch dimension
        else:
            res = model(imgs[None]) # Add batch dimension

    # The output of Pi3 is a dictionary with the following keys:
    # {
    #     'points': (B, N, H, W, 3),
    #     'local_points': (B, N, H, W, 3), # per-view local point maps
    #     'conf': (B, N, H, W, 1),
    #     'camera_poses': (B, N, 4x4), # cam2world transformation matrices in 4x4 openCV format
    # }
    # here, B = 1 because we are only processing one video
    
    # Extract data from results
    points = res['points'][0]  # (N, H, W, 3) - global 3D points 
    local_points = res['local_points'][0]  # (N, H, W, 3) - local point maps
    conf = res['conf'][0]  # (N, H, W, 1) - confidence scores
    camera_poses = res['camera_poses'][0]  # (N, 4, 4) - camera poses
    num_frames = camera_poses.shape[0]
    
    # Move to CPU and numpy for visualization
    points_np = points.detach().cpu().numpy()
    local_points_np = local_points.detach().cpu().numpy()
    conf_np = conf.detach().cpu().numpy()
    camera_poses_np = camera_poses.detach().cpu().numpy()
    imgs_np = imgs.detach().cpu().numpy()
    
    print(f"Loaded {num_frames} frames for visualization")
    
    # GUI controls
    reset_camera = server.gui.add_button(
            label="Reset Up Direction",
            icon=viser.Icon.ARROW_BIG_UP_LINES,
            color="gray",
            hint="Set the up direction of the camera orbit controls to the camera's current up direction.",
        )
    @reset_camera.on_click
    def _reset_camera_cb(_) -> None:
        for client in server.get_clients().values():
            client.camera.up_direction = vtf.SO3(client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])
        
    conf_threshold = server.add_gui_slider(
        "Confidence Threshold",
        min=0.01,
        max=0.99,
        step=0.01,
        initial_value=0.1,
    )
    
    step_size = server.add_gui_slider(
        "Step Size",
        min=1,
        max=max(1, num_frames // 2),
        step=1,
        initial_value=1,
    )
    
    show_cameras = server.add_gui_checkbox("Show Cameras", initial_value=True)
    show_points = server.add_gui_checkbox("Show Point Clouds", initial_value=True)
    
    # State tracking
    current_conf_threshold = conf_threshold.value
    current_step_size = step_size.value
    current_show_cameras = show_cameras.value
    current_show_points = show_points.value
    
    def update_visualization():
        # Clear existing objects
        server.scene.reset()
        
        # Get frame indices based on step size
        frame_indices = list(range(0, num_frames, current_step_size))
        
        for i, cam_idx in enumerate(frame_indices):
            # Color for this camera/points (cycle through colors)
            colors = [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green  
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
                [1.0, 0.0, 1.0],  # Magenta
                [0.0, 1.0, 1.0],  # Cyan
            ]
            color = colors[i % len(colors)]
            
            # Add camera visualization
            if current_show_cameras:
                # Camera pose is cam2world transformation
                pose = camera_poses_np[cam_idx]  # 4x4 matrix
                
                # Extract rotation and translation
                rotation_matrix = pose[:3, :3]
                position = pose[:3, 3]
                
                # Convert rotation matrix to quaternion (wxyz format)
                rotation_obj = R.from_matrix(rotation_matrix)
                wxyz = rotation_obj.as_quat()  # scipy returns xyzw, need to reorder to wxyz
                wxyz = np.array([wxyz[3], wxyz[0], wxyz[1], wxyz[2]])  # convert xyzw -> wxyz
                
                # Create camera frame
                # server.scene.add_frame(
                #     name=f"/camera_{cam_idx}",
                #     wxyz=wxyz,
                #     position=position,
                #     axes_length=0.1,
                #     axes_radius=0.005,
                # )
                server.scene.add_camera_frustum(
                    name=f"/camera_{cam_idx}",
                    fov=math.radians(120),
                    aspect=1.33,
                    scale=0.1,
                    wxyz=wxyz,
                    position=position,
                    color="blue",
                    image=cv2.cvtColor(imgs_np[cam_idx], cv2.COLOR_BGR2RGB),
                )
            
            # Add point cloud visualization  
            if current_show_points:
                # Apply confidence mask
                conf_mask = torch.sigmoid(torch.from_numpy(conf_np[cam_idx, ..., 0])) > current_conf_threshold
                
                # Apply edge filtering
                non_edge = ~depth_edge(torch.from_numpy(local_points_np[cam_idx, ..., 2]), rtol=0.03)
                
                # Combine masks
                mask = torch.logical_and(conf_mask, non_edge).numpy()
                
                if np.any(mask):
                    # Get masked points and colors
                    masked_points = points_np[cam_idx][mask]  # Shape: (N_valid, 3)
                    
                    # Get corresponding image colors
                    img_colors = imgs_np[cam_idx].transpose(1, 2, 0)[mask]  # (N_valid, 3)
                    img_colors = (img_colors * 255).astype(np.uint8)
                    
                    # Add to scene
                    server.scene.add_point_cloud(
                        name=f"/points_{cam_idx}",
                        points=masked_points,
                        colors=img_colors,
                        point_size=0.002,
                    )
    
    # Initial visualization
    update_visualization()
    
    # Main loop
    while True:
        # Check if GUI values changed
        time.sleep(0.02)
        if (conf_threshold.value != current_conf_threshold or
            step_size.value != current_step_size or
            show_cameras.value != current_show_cameras or
            show_points.value != current_show_points):
            
            # Update state
            current_conf_threshold = conf_threshold.value
            current_step_size = step_size.value
            current_show_cameras = show_cameras.value
            current_show_points = show_points.value
            
            # Update visualization
            update_visualization()
        
        server.update()



if __name__ == "__main__":
    tyro.cli(main)