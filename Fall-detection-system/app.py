from ultralytics import YOLO
import gradio as gr
import cv2
import numpy as np

# Load the model
model = YOLO("yolo11s-pose.pt")  # You can change to your custom trained model later

def is_falling(keypoints):
    """Advanced fall detection using body keypoints"""
    if len(keypoints) < 17:
        return False
    
    head_y = keypoints[0][1].item()
    shoulder_y = (keypoints[11][1].item() + keypoints[12][1].item()) / 2
    ankle_y = (keypoints[15][1].item() + keypoints[16][1].item()) / 2
    
    height_diff = abs(head_y - ankle_y)
    shoulder_height = abs(shoulder_y - ankle_y)
    
    # Tuned thresholds for lying down position
    if height_diff < 90 and shoulder_height < 130:
        return True
    return False

def detect_fall(image):
    results = model.predict(image, conf=0.5, verbose=False)
    annotated = results[0].plot()
    
    fall_detected = False
    for result in results:
        if result.keypoints is not None:
            kpts = result.keypoints.data[0]
            if is_falling(kpts):
                fall_detected = True
                cv2.putText(annotated, "🚨 FALL DETECTED!", (50, 100),
                          cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)
                cv2.putText(annotated, "Call for help immediately!", (50, 160),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    status = "🚨 Fall Detected! Call for help" if fall_detected else "✅ No Fall Detected"
    return annotated, status

# Gradio Web Interface
interface = gr.Interface(
    fn=detect_fall,
    inputs=gr.Image(type="pil", label="Upload Image or Use Webcam"),
    outputs=[
        gr.Image(type="pil", label="Detection Result"),
        gr.Textbox(label="Status")
    ],
    title="🛡️ Advanced Fall Detection System",
    description="Real-time fall detection using YOLO11 Pose Estimation. Useful for elderly care, hospitals, and smart homes.",
    examples=[
        ["examples/fall_example.jpg"],
        ["examples/normal_example.jpg"]
    ],
    live=True,
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch(share=False)  