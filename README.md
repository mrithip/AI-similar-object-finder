# Similar Object Identifier (Screws and nuts)

AI-based system for detecting screws and nuts in images and finding similar components based on a selected reference object.

## Features

- **Object Detection**: Detects screws and nuts using YOLOv8
- **Classification**: Identifies Hex Nut, Phillips Screw, and Torx Screw
- **Similarity Matching**: Find visually similar components based on size
- **Interactive UI**: Streamlit interface for easy use

## Prerequisites

```bash
pip install roboflow opencv-python python-dotenv matplotlib streamlit streamlit-drawable-canvas inference
```

## Setup

1. Create a `.env` file with your Roboflow credentials:
   ```
   API=your_roboflow_api_key
   PROJECT=your_project_name
   MODEL_VERSION=your_model_version
   WORKSPACE=your_workspace_name
   ```

2. Add test images (`test.png`, `test1.png`) to the project directory

## Usage

### Basic Detection
```bash
python classification.py
```

### Detection with Summary Count
```bash
python classificationwithsummary.py
```

### Interactive Similarity Finder
```bash
streamlit run similarimg.py
```

## How It Works

1. Upload an image
2. The YOLOv8 model detects all screws and nuts
3. Draw a box around a reference object
4. The system finds similar objects based on:
   - **Length**: max(width, height)
   - **Thickness**: min(width, height)
   - Objects within 8% tolerance are marked as similar

## File Overview

| File | Description |
|------|-------------|
| `classification.py` | Basic detection with OpenCV |
| `classificationwithsummary.py` | Detection with count summary |
| `classificatiowithoutsummary.py` | Simple detection output |
| `similarimg.py` | Interactive similarity finder (Streamlit) |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API` | Roboflow API key |
| `PROJECT` | Roboflow project name |
| `MODEL` | Model version number |
| `MODEL_ID` | Full model identifier |

## Notes

- Keep your `.env` file secure and never commit it to version control
- Detection confidence threshold: 55%
- Similarity tolerance: 8%
