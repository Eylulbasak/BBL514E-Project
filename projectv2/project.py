#  Traffic Vehicle Density Estimation from Recorded Videos Using Object Detection

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import torch
import pandas as pd
from ultralytics import YOLO
from tracker import *
import numpy as np
import time
import csv
import json
import os
from datetime import datetime

CONFIG_FILE = "config.json"

config = {"csv_path": ""}

def load_config():
    global config
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            config["csv_path"] = data.get("csv_path", "")

def save_config():
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"csv_path": config["csv_path"]}, f, ensure_ascii=False, indent=4)

#gui 
root = tk.Tk()
root.title("Traffic Density Analysis System")
root.geometry("800x600")

model = YOLO('yolo11x.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
tracker = Tracker()

with open("coco.txt", "r") as f:
    class_list = f.read().split("\n")

def point_in_polygon(x, y, polygon):
    num = len(polygon)
    j = num - 1
    odd_nodes = False
    for i in range(num):
        if (polygon[i][1] < y and polygon[j][1] >= y) or (polygon[j][1] < y and polygon[i][1] >= y):
            if polygon[i][0] + (y - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) * (polygon[j][0] - polygon[i][0]) < x:
                odd_nodes = not odd_nodes
        j = i
    return odd_nodes

def select_region(video_file):
    points = []

    def draw(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))

    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return []

    cv2.namedWindow("Select Region")
    cv2.setMouseCallback("Select Region", draw)

    while True:
        temp = frame.copy()

        for i, pt in enumerate(points):
            cv2.circle(temp, pt, 5, (0, 255, 0), -1)
            cv2.putText(temp, f"{i+1}", pt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if len(points) == 4:
            cv2.polylines(temp, [np.array(points)], isClosed=True, color=(0, 255, 255), thickness=2)
            cv2.putText(temp, "Region complete. Press Q to confirm.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 6)
            cv2.putText(temp, "Region complete. Press Q to confirm.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(temp, "Click 4 points to draw region", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 6)
            cv2.putText(temp, "Click 4 points to draw region", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Select Region", temp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and len(points) == 4:
            break

    cv2.destroyAllWindows()
    return points

def log_to_csv(location, vehicle_id):
    csv_file = config["csv_path"]
    if not csv_file:
        return
    fieldnames = ["location", "date", "vehicle_id"]
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_data = {
        "location": location,
        "date": now_str,
        "vehicle_id": vehicle_id
    }
    try:
        file_exists = os.path.exists(csv_file)
        with open(csv_file, 'a', newline='', encoding='utf-8') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
    except Exception as e:
        print("CSV saved error:", e)

def process_video(video_path, region):
    cap = cv2.VideoCapture(video_path)
    vehicle_count = 0
    counted_ids = set()
    base_name = os.path.basename(video_path)
    location = os.path.splitext(base_name)[0]
    
    exit_flag = False
    show_only_region = True  #displays vehicles within a specific region
    
    vehicle_positions = {}  #position log for each ID
    counting_threshold = 5  #minimum number of frames for counting a vehicle
    
    #metrics for traffic density 
    fps = cap.get(cv2.CAP_PROP_FPS)
    flow_window = int(fps * 30)  #for 30 second
    flow_history = []  #number of vehicles that passed in the last 30 seconds
    current_vehicles_in_region = 0  #number of vehicles in the region
    vehicle_waiting_times = {}  #waiting time for each vehicle
    vehicle_entry_times = {}  #time of entry of each vehicle into the zone
    
    # Debug:polygon points
    print(f"Selected region points: {region}")

    def handle_mouse(event, x, y, flags, param):
        nonlocal exit_flag, show_only_region
        if event == cv2.EVENT_LBUTTONDOWN:
            if x > frame.shape[1] - 110 and y < 40:
                exit_flag = True
            elif x < 150 and y < 40:
                show_only_region = not show_only_region

    window_name = "Traffic Counter"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, handle_mouse)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement to better detect dark vehicles
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        #merging the original frame and improved version of the frame
        alpha = 0.7
        processed_frame = cv2.addWeighted(frame, alpha, enhanced_frame, 1-alpha, 0)

        results = model.predict(processed_frame, device=device, conf=0.25, iou=0.45)
        detections = results[0].boxes.data.cpu().numpy()
        px = pd.DataFrame(detections).astype("float")

        boxes = []
        for _, row in px.iterrows():
            x1, y1, x2, y2, confidence, cls_id = row
            label = class_list[int(cls_id)]
            
            #accept only these three vehicle classes and check lower confidence score
            if label in ['car', 'bus', 'truck'] and confidence > 0.25:
                #minimum size control
                width = x2 - x1
                height = y2 - y1
                min_size = 25
                
                if width >= min_size and height >= min_size:
                    boxes.append([int(x1), int(y1), int(x2), int(y2)])

        tracked = tracker.update(boxes)
        
        #update the vehicles positions
        current_vehicles_in_region = 0
        current_time = frame_count / fps  #time (second)
        
        for x1, y1, x2, y2, obj_id in tracked:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            in_region = point_in_polygon(cx, cy, region)
            
            if in_region:
                current_vehicles_in_region += 1
                #record the time of entry when the vehicle enters the zone for the first time
                if obj_id not in vehicle_entry_times:
                    vehicle_entry_times[obj_id] = current_time
                #update waiting time
                vehicle_waiting_times[obj_id] = current_time - vehicle_entry_times[obj_id]
            
            if show_only_region and not in_region:
                continue
            
            if obj_id not in vehicle_positions:
                vehicle_positions[obj_id] = []
            
            vehicle_positions[obj_id].append({
                'x': cx, 'y': cy, 'in_region': in_region, 'frame': frame_count
            })
            
            #last 10 frame
            if len(vehicle_positions[obj_id]) > 10:
                vehicle_positions[obj_id] = vehicle_positions[obj_id][-10:]
            
            #Counting methos: If the vehicle has been in the zone for a sufficient time and has not yet been counted
            if in_region and obj_id not in counted_ids:
                #check vehicle position log
                recent_positions = vehicle_positions[obj_id][-counting_threshold:]
                in_region_frames = sum(1 for pos in recent_positions if pos['in_region'])
                
                if in_region_frames >= counting_threshold:
                    print(f"Vehicle {obj_id} counted at position ({cx}, {cy}) after {in_region_frames} frames")
                    counted_ids.add(obj_id)
                    vehicle_count += 1
                    log_to_csv(location, obj_id)
                    
                    #Add it to flow history
                    flow_history.append(frame_count)
                    #last 10 second
                    flow_history = [f for f in flow_history if frame_count - f <= flow_window]
                    
                    #show counted vehicles with a red box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"COUNTED: {obj_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    #show vehicles that are still waiting to be counted with a yellow box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f"WAITING: {obj_id} ({in_region_frames}/{counting_threshold})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif in_region and obj_id in counted_ids:
                #show already counted vehicles with a blue box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ALREADY COUNTED: {obj_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                #show vehicles outside the region with a green box (only if show_only_region=False)
                if not show_only_region:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {obj_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        #calculate traffic metrics
        flow_rate = len(flow_history)  #number of vehicles that passed in the last 30 seconds
        avg_waiting_time = sum(vehicle_waiting_times.values()) / len(vehicle_waiting_times) if vehicle_waiting_times else 0
        
        #density score
        density_score = current_vehicles_in_region
        
        #congestion level
        if density_score >= 8 or flow_rate >= 15:
            congestion_level = "DENSE"
            congestion_color = (0, 0, 255)  #RED
        elif density_score >= 5 or flow_rate >= 10:
            congestion_level = "MEDIUM"
            congestion_color = (0, 165, 255)  #ORANGE
        elif density_score >= 3 or flow_rate >= 5:
            congestion_level = "SLIGHT"
            congestion_color = (0, 255, 255)  #YELLOW
        else:
            congestion_level = "FLOW"
            congestion_color = (0, 255, 0)  #GREEN

        if region:
            pts = np.array(region, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        #UI Improvements: Text with background
        text = f"Vehicle Count: {vehicle_count}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        sub_img = frame[20:20+text_height+20, 20:20+text_width+20]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
        frame[20:20+text_height+20, 20:20+text_width+20] = res
        cv2.putText(frame, text, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #traffic metrics
        metrics_y = 120
        cv2.putText(frame, f"Flow Rate (30s): {flow_rate} vehicles", (30, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Density: {density_score} vehicles in region", (30, metrics_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Avg Wait: {avg_waiting_time:.1f} seconds", (30, metrics_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        #traffic congestion level
        cv2.putText(frame, f"TRAFFIC: {congestion_level}", (30, metrics_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, congestion_color, 2)

        #Debug info
        debug_text = f"Tracked IDs: {len(tracked)}, Counted: {len(counted_ids)}, Frame: {frame_count}"
        cv2.putText(frame, debug_text, (30, metrics_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        #Toggle Button
        toggle_text = "REGION ONLY" if show_only_region else "SHOW ALL"
        cv2.rectangle(frame, (10, 10), (140, 40), (0, 255, 0), -1)
        cv2.putText(frame, toggle_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        #exit button
        cv2.rectangle(frame, (frame.shape[1] - 110, 10), (frame.shape[1] - 10, 40), (0, 0, 255), -1)
        cv2.putText(frame, "EXIT", (frame.shape[1] - 100, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or exit_flag: #exit on ESC or button
            break

    cap.release()
    cv2.destroyAllWindows()

def select_video():
    if not config["csv_path"]:
        messagebox.showwarning("Warning", "Set CSV path first")
        return
    file_path = filedialog.askopenfilename(title="Select video",
                                           filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
    if file_path:
        region = select_region(file_path)
        if not region:
            messagebox.showwarning("Warning", "No valid region selected")
            return
        process_video(file_path, region)

def open_log_settings():
    win = tk.Toplevel(root)
    win.title("CSV Path")
    win.geometry("400x120")
    var = tk.StringVar(value=config["csv_path"])

    def choose():
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV Files", "*.csv")])
        if path:
            var.set(path)

    def save():
        config["csv_path"] = var.get()
        save_config()
        win.destroy()

    tk.Label(win, text="Current Path:").pack(pady=5)
    tk.Label(win, textvariable=var).pack()
    tk.Button(win, text="Choose Path", command=choose).pack(pady=5)
    tk.Button(win, text="Save", command=save).pack()

# GUI buttons
label_info = tk.Label(root, text="1) Set CSV path\n2) Select a video and draw a region to count vehicles")
label_info.pack(pady=10)

btn_path = tk.Button(root, text="Set CSV Path", command=open_log_settings, width=20)
btn_path.pack(pady=5)

btn_select = tk.Button(root, text="Select Video", command=select_video, width=20)
btn_select.pack(pady=5)

root.protocol("WM_DELETE_WINDOW", root.destroy)

if __name__ == "__main__":
    load_config()
    root.mainloop()