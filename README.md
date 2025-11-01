# AI-In-Robotics (YOLOv8 Object Detection)

This project uses the **YOLOv8** model (by [Ultralytics](https://github.com/ultralytics/ultralytics)) for object detection.  
The model was trained using the `yolov8m` variant.  
This README provides a full setup guide for running the project on **both CPU and GPU** environments.

The project repository is quite large (even without including the dataset), so it is provided via the GitHub link below.  
The primary development branch with all the features included is `main`.

The whole dataset is also included in the main branch.

Link to Github Project: 

https://github.com/tashil99/AI-In-Robotics

---

## Requirements

- **Python** ≥ 3.8 (recommended: 3.10 or newer)  
- **pip** ≥ 21.0  
- **Git** (for cloning)  
- **CUDA Toolkit** (only if using GPU)

---

## 1. Clone the Repository

```
https://github.com/tashil99/AI-In-Robotics.git
```
## 2. Create and Activate a Virtual Environment
### For Windows (PowerShell)

```
python -m venv venv
venv\Scripts\activate
```
### For Linux/macOS
```
python3 -m venv venv
source venv/bin/activate
```
## 3. Install Dependencies
### Option A - CPU Setup

If your system does not have an NVIDIA GPU or CUDA:
```
pip install torch torchvision torchaudio
pip install ultralytics numpy
```
### Option B - GPU Setup (CUDA 12.1)

If you have an NVIDIA GPU and CUDA installed, use the CUDA-compatible build of PyTorch:

```
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics numpy
```

Note: If installing via terminal does not work, some IDEs (like PyCharm) allow you to install dependencies by hovering over the missing package or via the interpreter settings. 
You can use that approach to complete the installation.
## 4. Verify Installation

Check YOLOv8 installation in terminal:

```
yolo
```
If installed correctly, you’ll see YOLOv8 CLI options something like this below:
```
usage: yolo [-h] {detect,segment,pose,train,val,predict,export,track}
```
If this command fails in PyCharm, install YOLO again within PyCharm’s interpreter:
```
pip install ultralytics
```
## 5. Running the Model

After all dependencies are installed, open and run the file `object-detection.py`.  
When prompted, select an image to process.  

The model will perform object detection on **one of the six classes** defined in `data.yaml`:  
`chair`, `desk`, `pen`, `laptop`, `mouse`, or `printer`.  

Detected objects will be highlighted with **bounding boxes** and the corresponding **confidence scores** will be displayed.

---