import open3d as o3d
import numpy as np
import os
import glob
import sys
import time
from plyfile import PlyData

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# CityScapes Palette (Normalized to 0.0 - 1.0 for Open3D)
PALETTE = np.zeros((30, 3))
PALETTE[1] = [128/255, 64/255, 128/255]   # Road
PALETTE[2] = [232/255, 35/255, 244/255]   # Sidewalk
PALETTE[3] = [70/255, 70/255, 70/255]     # Building
PALETTE[4] = [156/255, 102/255, 102/255]  # Wall
PALETTE[5] = [153/255, 153/255, 190/255]  # Fence
PALETTE[6] = [153/255, 153/255, 153/255]  # Pole
PALETTE[7] = [30/255, 170/255, 250/255]   # Traffic Light
PALETTE[8] = [0/255, 220/255, 220/255]    # Traffic Sign
PALETTE[9] = [35/255, 142/255, 107/255]   # Vegetation
PALETTE[10] = [152/255, 251/255, 152/255] # Terrain
PALETTE[11] = [180/255, 130/255, 70/255]  # Sky
PALETTE[12] = [60/255, 20/255, 220/255]   # Pedestrian
PALETTE[14] = [142/255, 0/255, 0/255]     # Car
PALETTE[15] = [70/255, 0/255, 0/255] # Truck
PALETTE[16] = [100/255, 60/255, 0/255] # Bus

def get_latest_session_dir(base_dir=DATA_DIR):
    if not os.path.exists(base_dir):
        return None
    subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    return max(subfolders, key=os.path.getmtime) if subfolders else None

def load_frame_data(path):
    """Reads the binary PLY, flips Y, and maps semantic tags to colors."""
    plydata = PlyData.read(path)
    v = plydata['vertex']
    
    # Extract XYZ and apply the Left-to-Right handed coordinate flip (Y * -1)
    points = np.vstack((v['x'], -v['y'], v['z'])).T
    
    # Safely extract property names native to the plyfile library
    prop_names = [p.name for p in v.properties]
    
    if 'ObjTag' in prop_names:
        tags = v['ObjTag'].astype(np.int32)
        # Cap tags at 29 to prevent index errors on the palette array
        safe_tags = np.clip(tags, 0, 29)
        colors = PALETTE[safe_tags]
    else:
        # Fallback to white if no semantic tags are found
        colors = np.full((len(points), 3), [1.0, 1.0, 1.0])
        
    return points, colors

def main(session_dir=None):
    if session_dir is None:
        session_dir = get_latest_session_dir()
        if session_dir is None:
            print(f"Error: No data sessions found in {DATA_DIR}")
            sys.exit(1)
            
    print(f"Loading session: {session_dir}")
    
    lidar_dir = os.path.join(session_dir, "_out_lidar")
    ply_paths = sorted(glob.glob(os.path.join(lidar_dir, "*.ply")))
    
    if not ply_paths:
        print("No .ply files found in the LiDAR directory.")
        sys.exit(1)

    print("Playback controls: [SPACE] Pause/Resume | [ESC] Exit")

    playback_state = {"paused": False, "running": True}

    def toggle_pause(vis):
        playback_state["paused"] = not playback_state["paused"]
        return False

    def exit_playback(vis):
        playback_state["running"] = False
        return False

    # Original KeyCallback initialization
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="LiDAR Semantic Playback", width=800, height=600)
    
    vis.register_key_callback(32, toggle_pause)
    vis.register_key_callback(256, exit_playback)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.0, 0.0, 0.0])
    opt.point_size = 2.0

    # Initialize the first frame
    points, colors = load_frame_data(ply_paths[0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

    frame_index = 1

    # Original execution loop
    while playback_state["running"]:
        if not playback_state["paused"] and frame_index < len(ply_paths):
            points, colors = load_frame_data(ply_paths[frame_index])
            
            # Update the point cloud arrays in-place
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
                
            vis.update_geometry(pcd)
            frame_index += 1
            
        elif frame_index >= len(ply_paths):
            playback_state["paused"] = True
            
        if not vis.poll_events():
            break 
            
        vis.update_renderer()
        time.sleep(0.05) 
        
    vis.destroy_window()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()