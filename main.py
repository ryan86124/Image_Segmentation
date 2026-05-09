import cv2
import numpy as np
import os
from ultralytics import YOLO

class AnimalMetrology:
    def __init__(self):
        """
        Initialize the detector by loading the YOLO segmentation model.
        The model path is retrieved from environment variables for flexibility.
        """
        # Load YOLO segmentation model (Default: yolov8n-seg.pt)
        self.model = YOLO(os.getenv("MODEL_PATH", "yolov8n-seg.pt"))
        
        # Define target COCO animal classes
        # Indices: 15:cat, 16:dog, 17:horse, 18:sheep, 19:cow, 20:elephant, 21:bear, 22:zebra, 23:giraffe
        self.animal_classes = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    def get_centroid(self, mask):
        """
        Calculate the geometric center (centroid) of a binary mask.
        Uses OpenCV moments for calculation.
        """
        M = cv2.moments(mask)
        if M["m00"] == 0: 
            return (0, 0)
        # Centroid formula: Cx = M10/M00, Cy = M01/M00
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    def run_inference(self, img_path):
        """
        Run inference on the image and measure distances between detected animals.
        
        Args:
            img_path (str): Path to the input image file.
            
        Returns:
            tuple: (animal_data_list, inter_individual_distance) or None if criteria not met.
        """
        # Run inference with a configurable confidence threshold
        result = self.model(img_path, conf=float(os.getenv("CONF_THRESHOLD", 0.5)))[0]
        
        # 1. Filter result: Keep only detections belonging to self.animal_classes
        animals = [r for r in result if int(r.boxes.cls) in self.animal_classes]
        
        # Skip processing if fewer than 2 animals are detected
        if len(animals) < 2:
            print(f"Skip {img_path}: Fewer than 2 animals detected.")
            return None

        # 2. Simulated Eye Detection Logic
        # Note: This uses horizontal extrema of the mask as a proxy for eye positions.
        # For higher precision, consider using a Pose (Keypoint) model.
        data = []
        for i, anim in enumerate(animals):
            # Extract binary mask data as a NumPy array
            mask = anim.masks.data[0].cpu().numpy()
            
            # Find coordinates where the mask is active
            y, x = np.where(mask > 0)
            
            # Simulate eyes: Use leftmost/rightmost points for X, and mean for Y
            left_eye = (int(np.min(x)), int(np.mean(y)))
            right_eye = (int(np.max(x)), int(np.mean(y)))
            
            # Calculate intra-individual distance (distance between eyes of the same animal)
            # Calculated using L2 Norm (Euclidean Distance)
            dist = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            
            data.append({
                "id": i,
                "left_eye": left_eye,
                "right_eye": right_eye,
                "intra_dist": round(float(dist), 2)
            })

        # 3. Inter-individual Distance Measurement
        # Measures the pixel distance between the 'right eye' of the first two detected animals
        inter_dist = np.linalg.norm(np.array(data[0]['right_eye']) - np.array(data[1]['right_eye']))
        
        return data, round(float(inter_dist), 2)

# Main Execution block
if __name__ == "__main__":
    detector = AnimalMetrology()
    input_path = "data/input/animals.jpg"
    output_csv = "result/metrology_results.csv"
    
    # 確保輸出目錄存在
    os.makedirs("result", exist_ok=True)

    print(f"Processing image: {input_path}...")
    result = detector.run_inference(input_path)

    if result:
        animal_info, inter_dist = result
        print(f"Measurement successful! Distance between right eyes of two animals: {inter_dist} px")

        # write result to csv
        with open(output_csv, "w") as f:
            f.write("animal_id,intra_eye_dist_px,inter_individual_dist_px\n")
            for anim in animal_info:
                f.write(f"{anim['id']},{anim['intra_dist']},{inter_dist if anim['id'] == 0 else ''}\n")
        
        print(f"Data saved to: {output_csv}")
    else:
        print("Measurement failed: Fewer than 2 animals detected.")