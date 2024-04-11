import cv2

# Load the pre-trained car cascade classifier
car_cascade = cv2.CascadeClassifier('cars.xml')

# Load the image
img = cv2.imread('133971.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect cars in the image
cars = car_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangles around the detected cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the image with detected cars
cv2.imshow('Car Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
