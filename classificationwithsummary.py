from roboflow import Roboflow
import cv2
import matplotlib.pyplot as plt
import os
from collections import Counter
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Access variables
api = os.getenv("API")
pro = os.getenv("PROJECT")
mod = os.getenv("MODEL_VERSION")
# 1️⃣ Load Roboflow model
# Note: Keep your API key private in production!
rf = Roboflow(api_key=api)
project = rf.workspace().project(pro)
model = project.version(int(mod)).model

# Define your image path
image_path = "test1.png"

if not os.path.exists(image_path):
    print(f"Error: {image_path} not found!")
else:
    # 2️⃣ Run inference
    # Note: confidence=80 and overlap=80 are quite high;
    # adjust if you miss objects or get duplicates.
    result = model.predict(
        image_path,
        confidence=80,
        overlap=95
    ).json()

    # 3️⃣ Load image for drawing
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # List to store all detected class names for counting
    detected_classes = []

    # 4️⃣ Process predictions and draw boxes
    for pred in result["predictions"]:
        # Extract data
        x, y = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]
        cls = pred["class"]
        conf = pred["confidence"]

        # Add to our counting list
        detected_classes.append(cls)

        # Calculate bounding box corners
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # Draw the rectangle
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Create label with class and confidence
        label = f"{cls}: {conf:.2f}"
        cv2.putText(img_rgb, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 5️⃣ Calculate and Print Counts
    class_counts = Counter(detected_classes)

    print("\n" + "=" * 30)
    print("      DETECTION SUMMARY")
    print("=" * 30)
    if not class_counts:
        print("No objects detected.")
    for item, count in class_counts.items():
        print(f"{item.upper():<15} : {count}")
    print("=" * 30 + "\n")

    # 6️⃣ Display result
    plt.figure(figsize=(15, 10))
    plt.imshow(img_rgb)
    plt.title(f"Total Detections: {len(detected_classes)}")
    plt.axis('off')
    plt.show()