import numpy as np
import cv2

def get_mask_centroid(mask):
    """
    Calculates the geometric center (centroid) of a segmentation mask.
    """
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        # Avoid division by zero if the mask is empty
        return None
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return (cx, cy)

def calculate_euclidean_distance(p1, p2):
    """
    Calculates the pixel distance between two points using the Euclidean distance formula.
    Formula: $d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$
    """
    if p1 is None or p2 is None:
        return 0.0
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def get_eye_points(mask):
    """
    Extracts feature points (simulated eye positions) from an animal mask.
    Note: In advanced implementations, this should be replaced by a 
    dedicated Keypoint Detection model.
    """
    y, x = np.where(mask > 0)
    if len(x) == 0: 
        return None, None
    
    # Simple logic: Use the 25th and 75th percentiles of the mask's 
    # horizontal span as simulated eye positions.
    left_eye = (int(np.percentile(x, 25)), int(np.median(y)))
    right_eye = (int(np.percentile(x, 75)), int(np.median(y)))
    return left_eye, right_eye