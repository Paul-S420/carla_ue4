import numpy as np

def get_mask_bounding_boxes(ins_image):
    """
    Extracts 2D bounding boxes purely from the CARLA instance segmentation mask.
    Red channel = Semantic Tag (Class)
    Green & Blue channels = Unique Instance ID
    """
    boxes = []
    
    # 1. Convert raw BGRA flat buffer to 3D matrix
    array = np.frombuffer(ins_image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (ins_image.height, ins_image.width, 4))
    
    b_channel = array[:, :, 0].astype(np.uint32)
    g_channel = array[:, :, 1].astype(np.uint32)
    r_channel = array[:, :, 2].astype(np.uint32) 
    
    # 2. Combine into a unique 24-bit identifier per object.
    # The lowest 8 bits store the Semantic Tag (Red).
    # The upper 16 bits store the unique Instance ID (Green + Blue).
    unique_objects_matrix = r_channel + (g_channel << 8) + (b_channel << 16)
    
    # 3. Find every unique object rendered on the screen
    unique_vals = np.unique(unique_objects_matrix)
    
    target_classes = {
        1,
        # 6, # pole
        # 7, # traffic light
        8, # traffic sign
        12,# pedestrian
        14,# car
        15,# truck
        16,# bus
        18,# motorcycle
        19 # bicycle
        }
    
    
    for val in unique_vals:
        # Decode the Semantic Tag (the lowest 8 bits)
        semantic_tag = val & 0xFF 
        
        if semantic_tag in target_classes:
            # Decode the unique instance ID (shift out the semantic tag bits)
            instance_id = val >> 8 
            
            # Create a boolean mask for this specific object
            mask = (unique_objects_matrix == val)
            
            # Find the min and max bounds where the mask is True
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Filter out tiny noise artifacts (boxes smaller than 15 pixels area)
            if (rmax - rmin) * (cmax - cmin) > 15:
                boxes.append({
                    "id": int(instance_id),
                    "class": int(semantic_tag),
                    "bbox": [int(cmin), int(rmin), int(cmax), int(rmax)]
                })
                
    return boxes