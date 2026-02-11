from roboflow import Roboflow
import cv2
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Access variables
api = os.getenv("API")
pro = os.getenv("PROJECT")
mod = os.getenv("MODEL_VERSION")
# 1️⃣ Load Roboflow model
# PRO TIP: Keep your API key secret!
rf = Roboflow(api_key=api)
project = rf.workspace().project(pro)
model = project.version(int(mod)).model

# Define your image path once to avoid mismatches
image_path = "test1.png"

if not os.path.exists(image_path):
    print(f"Error: {image_path} not found!")
else:
    # 2️⃣ Run inference
    result = model.predict(
        image_path,
        confidence=85,
        overlap=95
    ).json()

    # 3️⃣ Load image for drawing
    img = cv2.imread(image_path)
    # Convert BGR (OpenCV default) to RGB for Matplotlib display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 4️⃣ Draw boxes
    for pred in result["predictions"]:
        # Roboflow provides center coordinates
        x, y = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]
        cls = pred["class"]
        conf = pred["confidence"]

        # Calculate top-left and bottom-right corners
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # Draw rectangle (Green)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add Label
        label = f"{cls}: {conf:.2f}"
        cv2.putText(img_rgb, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 5️⃣ Display result
    plt.figure(figsize=(15, 15))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()