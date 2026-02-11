from roboflow import Roboflow
import cv2
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Access variables
api = os.getenv("API")
pro = os.getenv("PROJECT")
mod = os.getenv("MODEL_VERSION")

# 1️⃣ Load Roboflow model
rf = Roboflow(api_key=api)
project = rf.workspace().project(pro)
model = project.version(int(mod)).model

# 2️⃣ Run inference on local image
result = model.predict(
    "test.png",    # path to your image
    confidence=40,
    overlap=30
).json()

# 3️⃣ Print raw predictions (IMPORTANT)
print(result)

# 4️⃣ Draw boxes manually (optional)
img = cv2.imread("test.png")
for pred in result["predictions"]:
    x, y = int(pred["x"]), int(pred["y"])
    w, h = int(pred["width"]), int(pred["height"])
    cls = pred["class"]
    conf = pred["confidence"]

    x1, y1 = int(x - w/2), int(y - h/2)
    x2, y2 = int(x + w/2), int(y + h/2)

    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(img, f"{cls} {conf:.2f}", (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()