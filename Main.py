# Install library yang dibutuhkan
!pip install torch torchvision
!pip install ultralytics
!git clone https://github.com/ultralytics/yolov5
from PIL import Image
from numpy import asarray
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ultralytics import settings
dir = "/content/mbil bus.jpg"
image = Image.open(dir)
image = image.convert('RGB')
plt.axis('off')
plt.imshow(image)
detector = YOLO('yolov8n.pt')
results = detector('mbil bus.jpg')
print(results)
# Function to draw boxes only for cars
def draw_yolo_boxes_for_cars(result_list):
    # Load the image
    data = plt.imread('mbil bus.jpg')  # Directly using 'a.jpg' as the filename

    # Plot the image
    plt.imshow(data)

    # Get the context for drawing boxes
    ax = plt.gca()

    # Initialize object count
    car_count = 0

    # Plot each detection result
    for result in result_list:
        # YOLO returns a `boxes` attribute which contains all bounding boxes and their labels
        for box in result.boxes:
            # Get the label index for the detected object
            label_index = int(box.cls[0])

            # Convert the label index to its string representation (e.g., "car", "bus", etc.)
            label = detector.model.names[label_index]

            # Check if the detected object is a car
            if label == 'car':
                # Get bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0]

                # Calculate width and height
                width = x2 - x1
                height = y2 - y1

                # Create a rectangle for the bounding box
                rect = plt.Rectangle((x1, y1), width, height, fill=False, color='Red')

                # Draw the bounding box
                ax.add_patch(rect)

                # Increment the car count
                car_count += 1

    # Show the plot
    plt.show()

    # Return the total number of detected cars
    return car_count

# Call the function to draw boxes for cars
car_count = draw_yolo_boxes_for_cars(results)
# To get the number of detected objects:
num_detections = len(results[0].boxes)

# Print the total number of detected objects
print(f"Total objek mobil yang terdeteksi: {num_detections}")
