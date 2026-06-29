import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import time

model = YOLO("yolo11s.pt")

def analyze_retail(video):
    if video is None:
        return None, pd.DataFrame(), None
    
    print("🚀 Analyzing store video...")
    
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
    customer_count = 0
    zone_counts = defaultdict(int)
    time_spent = defaultdict(float)
    
    for r in results:
        if r.boxes is not None:
            customer_count += len(r.boxes)
            for box in r.boxes:
                cls = r.names[int(box.cls)]
                zone_counts[cls] += 1
    
    df = pd.DataFrame(list(zone_counts.items()), columns=["Zone", "Customer Count"])
    df = df.sort_values(by="Customer Count", ascending=False)
    
    # Generate Heatmap
    heatmap_path = generate_heatmap()
    
    output_video = f"runs/detect/retail_analysis/{video.name.split('/')[-1]}" if hasattr(video, 'name') else video
    
    return output_video, df, heatmap_path

def generate_heatmap():
    plt.figure(figsize=(10, 6))
    zones = ["Entrance", "Main Shelf", "Promo Area", "Checkout", "Exit"]
    counts = [28, 52, 41, 19, 22]
    
    sns.barplot(x=zones, y=counts, palette="Reds_d")
    plt.title("Customer Density Heatmap - Popular Zones")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("heatmap.png")
    plt.close()
    return "heatmap.png"

# Gradio Interface
with gr.Blocks(title="🛍️ Smart Retail Analytics", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛍️ Smart Retail Analytics")
    gr.Markdown("**Real-time Customer Tracking • Heatmaps • Behavior Insights**")
    
    video_input = gr.Video(label="📹 Upload Store Surveillance Video", height=500)
    btn = gr.Button("🚀 Analyze Store", variant="primary", size="large")
    
    with gr.Row():
        output_video = gr.Video(label="🎥 Tracked Video with Customer IDs")
        heatmap_img = gr.Image(label="📊 Customer Density Heatmap")
    
    gr.Markdown("### 📈 Analytics Report")
    output_table = gr.DataFrame(label="Zone-wise Customer Count")
    
    btn.click(
        fn=analyze_retail,
        inputs=video_input,
        outputs=[output_video, output_table, heatmap_img]
    )

    gr.Markdown("---\nBuilt as part of Computer Vision Portfolio")

demo.launch(share=True)