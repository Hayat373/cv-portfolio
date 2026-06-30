from ultralytics import YOLO
import gradio as gr
import cv2
import numpy as np

# Load model
model = YOLO("yolo11s-pose.pt")

def fall_detection(image):
    results = model.predict(image, conf=0.5, verbose=False)
    annotated = results[0].plot()
    
    fall_detected = False
    for result in results:
        if result.keypoints is not None:
            kpts = result.keypoints.data[0]
            if len(kpts) > 15:
                # Advanced fall detection using pose
                head_y = kpts[0][1].item()
                ankle_y = kpts[15][1].item()
                if abs(head_y - ankle_y) < 80:   # Person is horizontal
                    fall_detected = True
                    cv2.putText(annotated, "🚨 FALL DETECTED!", (50, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    
    status = "🚨 Fall Detected!" if fall_detected else "✅ No Fall"
    return annotated, status

# Gradio Interface
interface = gr.Interface(
    fn=fall_detection,
    inputs=gr.Image(type="pil", label="Upload Image or Live Camera"),
    outputs=[
        gr.Image(type="pil", label="Detection Result"),
        gr.Textbox(label="Status")
    ],
    title="🛡️ Advanced Fall Detection System",
    description="Real-time fall detection using YOLO11 Pose. Useful for elderly care & safety.",
    examples=[
        ["examples/fall_example1.jpg"],
        ["examples/normal_example1.jpg"]
    ],
    live=True
)

if __name__ == "__main__":
    interface.launch(share=True)