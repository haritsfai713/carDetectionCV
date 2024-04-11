import cv2
import torch

# Load YOLOv8 model and weights
model = torch.hub.load('ultralytics/yolov8', 'yolov8n', pretrained=True)

# Open video capture
cap = cv2.VideoCapture('videoplayback.mp4')

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Count cars
    car_count = 0
    for box, conf, cls in results.xyxy[0]:
        if cls == 2 and conf > 0.5:  # Filter for car class with confidence above 0.5
            car_count += 1
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(frame, f"Car: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display count and frame
    cv2.putText(frame, f"Car Count: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Car Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
