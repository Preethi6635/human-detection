from ultralytics import YOLO
import cv2
import numpy as np

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use 'yolov8s.pt' for better accuracy if needed

# Open video file or webcam stream
video_path = r"C:\Users\91901\Downloads\1044583837-preview.mp4"  # Replace with your video path, or use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Loop over frames from the video stream
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection on the current frame
    results = model(frame)

    # Draw bounding boxes and shade humans
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Convert bounding box tensor to numpy
        confidences = result.boxes.conf.cpu().numpy()  # Convert confidence tensor to numpy
        classes = result.boxes.cls.cpu().numpy()  # Convert class tensor to numpy
        
        for i, cls in enumerate(classes):
            if int(cls) == 0:  # Class label 0 is for humans
                # Extract bounding box coordinates and confidence score
                x1, y1, x2, y2 = boxes[i]
                conf = confidences[i]
                
                # Create a green shaded rectangle with transparency
                overlay = frame.copy()
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), -1)  # Filled rectangle
                
                # Add transparency to the shaded area (0.4 is the transparency level)
                alpha = 0.4
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # Draw the label on top of the shaded human
                label = f"Human {conf:.2f}"  # Format the confidence score as a float
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow('Human Detection with Shading', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()