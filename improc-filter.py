import cv2

# Load cascade classifier
car_cascade = cv2.CascadeClassifier('cars.xml')

# Open video capture
cap = cv2.VideoCapture('samplevideo4.mp4')

# Define ROIs (adjust coordinates based on your camera view)
roi1 = (100, 300, 300, 400)  # Approaching lane ROI
roi2 = (400, 100, 300, 200)  # Receding lane ROI

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Detect cars in each ROI
    cars1 = car_cascade.detectMultiScale(frame[roi1[1]:roi1[1]+roi1[3], roi1[0]:roi1[0]+roi1[2]], 1.1, 4)
    cars2 = car_cascade.detectMultiScale(frame[roi2[1]:roi2[1]+roi2[3], roi2[0]:roi2[0]+roi2[2]], 1.1, 4)

    # Filter for approaching cars (adjust filtering criteria as needed)
    approaching_cars = [car for car in cars1 if car[2] > 50 and car[3] > 50 and car[1] > 200]  # Example criteria

    # Count approaching cars
    car_count = len(approaching_cars)

    # Draw bounding boxes and display count
    for (x, y, w, h) in approaching_cars:
        x1, y1 = x + roi1[0], y + roi1[1]  # Adjust coordinates for ROI
        cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
    cv2.putText(frame, f"Approaching Cars: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Car Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
