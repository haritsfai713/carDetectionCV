# Bard version
import cv2

# Load the pre-trained car cascade classifier
car_cascade = cv2.CascadeClassifier('cars.xml')

# Open video capture
cap = cv2.VideoCapture('videoplayback.mp4')

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars using the cascade classifier
    cars = car_cascade.detectMultiScale(gray, 1.1, 4)

    # Count cars
    car_count = len(cars)

    # Draw bounding boxes around detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Car', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display count and frame
    cv2.putText(frame, f"Car Count: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Car Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# import cv2

# # Load the pre-trained car detection classifier (you may need to change the path)
# car_cascade = cv2.CascadeClassifier('cars.xml')

# # Initialize variables
# car_count = 0

# # Open a video capture object (you may need to change the index or path)
# cap = cv2.VideoCapture('samplevideo4.mp4')  # 0 for default camera, or provide a video file path

# roi_coord = (640, 0, 1280, 720)  # Approaching lane ROI

# while True:
#     # Read a frame from the camera
#     ret, frame = cap.read()

#     roi = frame[roi_coord[0]:roi_coord[1]+roi_coord[3], roi_coord[0]:roi_coord[0]+roi_coord[2]]

#     # Convert the frame to grayscale (optional, but can improve performance)
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

#     # Detect cars in the frame
#     cars = car_cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Update the car count
#     car_count = len(cars)

#     # Draw rectangles around the cars
#     for (x, y, w, h) in cars:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

#     # Display the frame with the car count
#     cv2.rectangle(frame,(roi_coord[0], roi_coord[1]),(roi_coord[0]+roi_coord[2],roi_coord[1]+roi_coord[3]),(0,0,255),2)
#     cv2.putText(frame, f'Cars: {car_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.imshow('Car Detection', frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture object and close all windows
# cap.release()
# cv2.destroyAllWindows()


