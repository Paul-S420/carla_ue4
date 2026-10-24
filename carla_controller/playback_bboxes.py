import cv2
import json
import os
import glob
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

CITYSCAPES_PALETTE = {
    1: ("Road", (128, 64, 128)),
    2: ("Sidewalk", (232, 35, 244)),
    3: ("Building", (70, 70, 70)),
    4: ("Wall", (156, 102, 102)),
    5: ("Fence", (153, 153, 190)),
    6: ("Pole", (153, 153, 153)),
    7: ("Traffic Light", (30, 170, 250)),
    8: ("Traffic Sign", (0, 220, 220)),
    9: ("Vegetation", (35, 142, 107)),
    10: ("Terrain", (152, 251, 152)),
    11: ("Sky", (180, 130, 70)),
    12: ("Pedestrian", (60, 20, 220)),
    13: ("Rider", (0, 0, 255)),
    14: ("Car", (142, 0, 0)),
    15: ("Truck", (70, 0, 0)),
    16: ("Bus", (100, 60, 0)),
    17: ("Train", (100, 80, 0)),
    18: ("Motorcycle", (230, 0, 0)),
    19: ("Bicycle", (32, 11, 119)),
    20: ("Static", (160, 190, 110)),
    21: ("Dynamic", (50, 120, 170)),
    22: ("Other", (80, 90, 55)),
    23: ("Water", (150, 60, 45)),
    24: ("RoadLine", (50, 234, 157)),
    25: ("Ground", (81, 0, 81)),
    26: ("Bridge", (100, 100, 150)),
    27: ("RailTrack", (140, 150, 230)),
    28: ("GuardRail", (180, 165, 180)),
}
target_classes={
    # 1, # road
    # 2,
    # 3,
    # 4,
    # 5,
    # 6, # pole
    7,
    8,
    # 9, # vegetation
    10,
    # 11, # sky
    12, # pedestrian
    13, # rider
    14, # car
    15, # truck
    16, # bus
    17,
    18,
    19, # bicycle
    # 20,
    # 21,
    # 22,
    # 23,
    # 24,
    # 25,
    # 26,
    # 27,
    # 28,
}

# conversion to BGR Format
CITYSCAPES_PALETTE_BGR = {}
for key, val in CITYSCAPES_PALETTE.items():
    CITYSCAPES_PALETTE_BGR[key] = (val[0], (val[1][2], val[1][1], val[1][0]))

def get_latest_session_dir(base_dir=DATA_DIR):
    if not os.path.exists(base_dir):
        return None
    subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    if not subfolders:
        return None
    return max(subfolders, key=os.path.getmtime)


def find_image_directory(session_dir):
    # candidates = ["_out_rgb", "_out_semantic_segmentation", "_out_instance_segmentation"] # show rgb images
    candidates = ["_out_semantic_segmentation", "_out_instance_segmentation", "_out_rgb"]  # show semandtic segmentation
    # candidates = ["_out_instance_segmentation","_out_rgb", "_out_semantic_segmentation"] # show instance segmentation
    for candidate in candidates:
        full_path = os.path.join(session_dir, candidate)
        if os.path.exists(full_path):
            images = glob.glob(os.path.join(full_path, "*.png"))
            if images:
                return full_path
    return None


def draw_bounding_box(img, box_data):
    if isinstance(box_data, dict):
        coords = box_data.get("bbox") # or box_data.get("coords")
        if not coords:
            print("Format error")
            return
        class_id = box_data.get("class_id", box_data.get("class", 0))
        if class_id not in target_classes: return
    else:
        return

    x_min, y_min, x_max, y_max = [int(v) for v in coords]
    label, color = CITYSCAPES_PALETTE.get(class_id, (f"Class {class_id}", (0, 255, 0)))

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

    text = f"{label} ({class_id})" if label else f"Class {class_id}"
    text += f"  ID: {box_data.get("id")}"
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    text_y = max(y_min - 4, h + 4)
    cv2.rectangle(img, (x_min, text_y - h - 2), (x_min + w + 2, text_y + 2), color, -1)
    cv2.putText(img, text, (x_min + 1, text_y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def main(session_dir=None):
    if session_dir is None:
        session_dir = get_latest_session_dir()
        if session_dir is None:
            print(f"Error: No data sessions found in {DATA_DIR}")
            sys.exit(1)

    print(f"Loading session: {session_dir}")

    image_dir = find_image_directory(session_dir)
    if not image_dir:
        print("Error: No image directory (_out_rgb, _out_semantic_segmentation, etc.) found with PNG files.")
        sys.exit(1)

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not image_paths:
        print(f"Error: No PNG images found in {image_dir}")
        sys.exit(1)

    bbox_dir = os.path.join(session_dir, "_out_bbox")
    has_bboxes = os.path.exists(bbox_dir)

    if has_bboxes:
        print(f"Detected bounding box directory: {bbox_dir}")
    else:
        print("No bounding box directory found. Replaying frames without annotations.")

    print("Playback controls: [SPACE] Pause/Resume | [ESC] Exit | [A / D] Step backward / forward (when paused)")

    idx = 0
    paused = False
    window_name = f"Playback - {os.path.basename(session_dir)}"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while idx < len(image_paths):
        img_path = image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            idx += 1
            continue

        frame_id = os.path.splitext(os.path.basename(img_path))[0]

        if has_bboxes:
            json_path = os.path.join(bbox_dir, f"{frame_id}.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r") as f:
                        boxes = json.load(f)
                    for box in boxes["boxes"]:
                        draw_bounding_box(img, box)
                except Exception as e:
                    print(f"Warning: Failed reading {json_path}: {e}")

        status_text = f"Frame: {frame_id} ({idx + 1}/{len(image_paths)})"
        if paused:
            status_text += " [PAUSED]"
        cv2.putText(img, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, img)

        delay = 0 if paused else 50
        key = cv2.waitKey(delay) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            paused = not paused
        elif paused and key in (ord("d"), ord("D"), 83):  # Step forward
            if idx < len(image_paths) - 1:
                idx += 1
        elif paused and key in (ord("a"), ord("A"), 81):  # Step backward
            if idx > 0:
                idx -= 1
        elif not paused:
            idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()