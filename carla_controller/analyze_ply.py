import os
import sys
import glob
import argparse
import numpy as np
from plyfile import PlyData

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Official CARLA Semantic Tag Mapping
CARLA_CLASSES = {
    0: "Unlabeled",
    1: "Road",
    2: "Sidewalk",
    3: "Building",
    4: "Wall",
    5: "Fence",
    6: "Pole",
    7: "TrafficLight",
    8: "TrafficSign",
    9: "Vegetation",
    10: "Terrain",
    11: "Sky",
    12: "Pedestrian",
    13: "Rider",
    14: "Car",
    15: "Truck",
    16: "Bus",
    17: "Train",
    18: "Motorcycle",
    19: "Bicycle",
    20: "Static",
    21: "Dynamic",
    22: "Other",
    23: "Water",
    24: "RoadLine",
    25: "Ground",
    26: "Bridge",
    27: "RailTrack",
    28: "GuardRail",
}

def get_latest_session_dir(base_dir=DATA_DIR):
    if not os.path.exists(base_dir):
        return None
    subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    return max(subfolders, key=os.path.getmtime) if subfolders else None

def resolve_ply_path(session_arg, file_arg):
    """Resolves flexible session and file arguments into a full .ply path."""
    # 1. If file_arg is already a valid direct path to a file, use it directly
    if file_arg and os.path.isfile(file_arg):
        return os.path.abspath(file_arg)

    # 2. Resolve the session directory
    if session_arg:
        if os.path.isdir(session_arg):
            session_dir = os.path.abspath(session_arg)
        else:
            session_dir = os.path.join(DATA_DIR, session_arg)
    else:
        session_dir = get_latest_session_dir()

    if not session_dir or not os.path.exists(session_dir):
        print(f"Error: Could not locate session directory '{session_arg or DATA_DIR}'.")
        sys.exit(1)

    lidar_dir = os.path.join(session_dir, "_out_lidar")
    if not os.path.exists(lidar_dir):
        print(f"Error: Missing '_out_lidar' directory in {session_dir}")
        sys.exit(1)

    ply_paths = sorted(glob.glob(os.path.join(lidar_dir, "*.ply")))
    if not ply_paths:
        print(f"Error: No .ply files found in {lidar_dir}")
        sys.exit(1)

    # 3. If no file specified, default to the first frame in the session
    if not file_arg:
        return ply_paths[0]

    # 4. Check for exact filename match (with or without .ply extension)
    candidate_name = file_arg if file_arg.endswith(".ply") else f"{file_arg}.ply"
    direct_match = os.path.join(lidar_dir, candidate_name)
    if os.path.isfile(direct_match):
        return direct_match

    # 5. If an integer is passed (e.g., "15"), match zero-padded frame name or list index
    if file_arg.isdigit():
        num = int(file_arg)
        for p in ply_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem.isdigit() and int(stem) == num:
                return p
        if 0 <= num < len(ply_paths):
            return ply_paths[num]

    print(f"Error: Could not find PLY file matching '{file_arg}' in {lidar_dir}")
    sys.exit(1)

def analyze_ply(ply_path):
    print(f"Analyzing file : {ply_path}")
    
    plydata = PlyData.read(ply_path)
    v = plydata['vertex']
    total_points = len(v)
    print(f"Total points   : {total_points}\n")

    prop_names = [p.name for p in v.properties]
    if 'ObjTag' not in prop_names or 'ObjIdx' not in prop_names:
        print(f"Error: File is missing 'ObjTag' or 'ObjIdx'. Found properties: {prop_names}")
        sys.exit(1)

    tags = np.asarray(v['ObjTag'], dtype=np.uint32)
    ids = np.asarray(v['ObjIdx'], dtype=np.uint32)

    unique_tags, tag_counts = np.unique(tags, return_counts=True)

    print(f"{'Tag ID':<8} | {'Class Name':<15} | {'Points':<8} | {'Unique ObjIdx Values'}")
    print("-" * 80)

    for tag, count in zip(unique_tags, tag_counts):
        class_name = CARLA_CLASSES.get(int(tag), "Unknown")
        
        # Extract all unique ObjIdx values where ObjTag == tag
        mask = (tags == tag)
        unique_ids = np.unique(ids[mask])
        ids_str = ", ".join(str(i) for i in unique_ids)
        
        print(f"{tag:<8} | {class_name:<15} | {count:<8} | [{ids_str}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ObjTag and ObjIdx in CARLA LiDAR PLY files.")
    parser.add_argument(
        "-s", "--session", 
        type=str, 
        default=None, 
        help="Session folder name inside 'data/' or path to session dir (defaults to latest session)."
    )
    parser.add_argument(
        "-f", "--file", 
        type=str, 
        default=None, 
        help="Frame number (e.g. 100), filename (000100.ply), or direct file path (defaults to first frame)."
    )
    
    args = parser.parse_args()
    target_ply = resolve_ply_path(args.session, args.file)
    analyze_ply(target_ply)