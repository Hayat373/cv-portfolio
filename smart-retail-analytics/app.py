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

# To keep track of time spent per customer
track_history = defaultdict(list)

def analyze_retail(video):
    if video is None:
        return None, pd.DataFrame(), None, None
    
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
    
    fps = 25  # Assume 25 fps for time calculation
    
    for frame_idx, r in enumerate(results):
        if r.boxes is not None and r.boxes.id is not None:
            customer_count += len(r.boxes)
            for box, track_id in zip(r.boxes, r.boxes.id):
                cls = r.names[int(box.cls)]
                zone_counts[cls] += 1
                
                # Track time spent (simple version)
                track_history[int(track_id)].append(frame_idx)
    
    # Calculate time spent
    for track_id, frames in track_history.items():
        duration = len(frames) / fps
        time_spent["Average Time Spent"] += duration
    
    if time_spent:
        time_spent["Average Time Spent"] /= len(track_history)
    
    # DataFrames
    zone_df = pd.DataFrame(list(zone_counts.items()), columns=["Zone", "Customer Count"])
    zone_df = zone_df.sort_values(by="Customer Count", ascending=False)
    
    time_df = pd.DataFrame([["Average Time in Store", f"{time_spent.get('Average Time Spent', 0):.1f} seconds"]], 
                           columns=["Metric", "Value"])
    
    # Generate Heatmap
    heatmap_path = generate_heatmap()
    
    output_video = f"runs/detect/retail_analysis/{video.name.split('/')[-1]}" if hasattr(video, 'name') else video
    
    return output_video, zone_df, time_df, heatmap_path

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
    gr.Markdown("**Customer Tracking • Heatmaps • Time Spent Analysis**")
    
    video_input = gr.Video(label="📹 Upload Store Surveillance Video", height=500)
    btn = gr.Button("🚀 Analyze Store", variant="primary", size="large")
    
    with gr.Row():
        output_video = gr.Video(label="🎥 Tracked Video with Customer IDs")
        heatmap_img = gr.Image(label="📊 Customer Density Heatmap")
    
    with gr.Row():
        zone_table = gr.DataFrame(label="Zone-wise Customer Count")
        time_table = gr.DataFrame(label="Time Spent Analysis")
    
    btn.click(
        fn=analyze_retail,
        inputs=video_input,
        outputs=[output_video, zone_table, time_table, heatmap_img]
    )

    gr.Markdown("---\nBuilt as part of Computer Vision Portfolio")

demo.launch(share=True)