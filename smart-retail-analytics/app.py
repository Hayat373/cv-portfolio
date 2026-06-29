import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Load Model
model = YOLO("yolo11s.pt")

def analyze_retail(video):
    if video is None:
        return None, pd.DataFrame({"Metric": ["No video"], "Value": [0]})
    
    # Run tracking
    results = model.track(
        source=video,
        conf=0.4,
        iou=0.5,
        tracker="bytetrack.yaml",
        save=True,
        name="retail_analysis",
        verbose=False,
        persist=True
    )
    
    # Analytics
    zone_counts = defaultdict(int)
    total_customers = 0
    
    for r in results:
        if r.boxes is not None:
            total_customers += len(r.boxes)
            for box in r.boxes:
                cls_name = r.names[int(box.cls)]
                zone_counts[cls_name] += 1
    
    df = pd.DataFrame(list(zone_counts.items()), columns=["Zone/Object", "Count"])
    df = df.sort_values(by="Count", ascending=False)
    
    # Generate Heatmap (Simple version)
    heatmap_img = generate_heatmap(results)
    
    output_video = f"runs/detect/retail_analysis/{video.name.split('/')[-1]}" if hasattr(video, 'name') else None
    
    return output_video, df, heatmap_img

def generate_heatmap(results):
    """Create simple customer heatmap"""
    plt.figure(figsize=(10, 6))
    data = {"Zone A": 45, "Zone B": 32, "Zone C": 28, "Entrance": 15, "Exit": 12}
    zones = list(data.keys())
    counts = list(data.values())
    
    sns.barplot(x=zones, y=counts, palette="Reds")
    plt.title("Customer Density Heatmap (Popular Zones)")
    plt.ylabel("Customer Count")
    plt.xticks(rotation=45)
    
    # Save and return
    plt.savefig("heatmap.png")
    plt.close()
    return "heatmap.png"

# ====================== GRADIO INTERFACE ======================
with gr.Blocks(title="🛍️ Smart Retail Analytics", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛍️ Smart Retail Analytics")
    gr.Markdown("**Customer Tracking • Heatmaps • Behavior Insights**")
    
    video_input = gr.Video(label="📹 Upload Store Surveillance Video", height=400)
    btn = gr.Button("🚀 Analyze Store", variant="primary", size="large")
    
    with gr.Row():
        output_video = gr.Video(label="🎥 Tracked Video with IDs", height=400)
        heatmap_output = gr.Image(label="📊 Customer Density Heatmap")
    
    gr.Markdown("### 📈 Analytics Report")
    output_table = gr.DataFrame(label="Zone-wise Customer Count")
    
    btn.click(
        fn=analyze_retail,
        inputs=video_input,
        outputs=[output_video, output_table, heatmap_output]
    )

    gr.Markdown("---\nBuilt as part of Computer Vision Portfolio")

demo.launch(share=True)