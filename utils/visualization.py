import cv2

def draw_measurements(image, animal_data, inter_dist):
    """
    Draws detection boxes, eye connection lines, and measurement values on the image.
    """
    annotated_img = image.copy()
    
    for data in animal_data:
        p1, p2 = data['left_eye'], data['right_eye']
        
        # Draw eye points (keypoints)
        cv2.circle(annotated_img, p1, 5, (0, 255, 0), -1)
        cv2.circle(annotated_img, p2, 5, (0, 255, 0), -1)
        
        # Draw the connection line for intra-individual eye distance
        cv2.line(annotated_img, p1, p2, (255, 0, 0), 2)
        cv2.putText(annotated_img, f"Dist: {data['intra_dist']:.1f}px", 
                    (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw the distance between right eyes across different individuals (if 2+ animals are detected)
    if len(animal_data) >= 2:
        r1 = animal_data[0]['right_eye']
        r2 = animal_data[1]['right_eye']
        
        # Draw the inter-individual connection line
        cv2.line(annotated_img, r1, r2, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Calculate midpoint for text placement
        mid_point = ((r1[0]+r2[0])//2, (r1[1]+r2[1])//2)
        cv2.putText(annotated_img, f"Inter-R: {inter_dist:.1f}px", 
                    mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    return annotated_img