import cv2
import numpy as np
import os
import glob
import tifffile
import time
from ultralytics import YOLO

INPUT_DIR = "output"             # Aligned TIFF folder
OUTPUT_DIR = "predictions_rpi3"   # Results folder
MODEL_PATH = "thermalnano_ncnn_model" # NCNN wrapper model folder
CONF_THRESHOLD = 0.10

# Model loading
print(f"Loading NCNN wrapper model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH, task="segment")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Process all TIFF files in the input directory
tiff_files = glob.glob(os.path.join(INPUT_DIR, "*.tiff"))
print(f"Found {len(tiff_files)} TIFF files.")

times = []

for i, filepath in enumerate(tiff_files):
    filename = os.path.basename(filepath)
    
    try:
        img_chw = tifffile.imread(filepath) 
        img_hwc = np.transpose(img_chw, (1, 2, 0))
        img_input = np.ascontiguousarray(img_hwc)
        
        start_time = time.time()

        results = model(img_input, conf=CONF_THRESHOLD, verbose=False)
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        times.append(elapsed_ms)
        
        result = results[0]
        num_det = len(result.boxes)
        print(f"[{i+1}/{len(tiff_files)}] {filename} -> {elapsed_ms:.1f} ms | Oggetti: {num_det}")
        
        if num_det > 0:
            rgb_pure = img_input[:, :, :3]
            bgr_vis = cv2.cvtColor(rgb_pure, cv2.COLOR_RGB2BGR)
            annotated_rgb = result.plot(img=bgr_vis)
            therm_channel = img_input[:, :, 3]
            norm_therm = cv2.normalize(therm_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            therm_color = cv2.applyColorMap(norm_therm, cv2.COLORMAP_JET)
            annotated_therm = result.plot(img=therm_color)
            combined_img = np.hstack((annotated_rgb, annotated_therm))
            
            save_path = os.path.join(OUTPUT_DIR, filename.replace(".tiff", "_result.jpg"))
            cv2.imwrite(save_path, combined_img)

    except Exception as e:
        print(f"[ERRORE] {filename}: {e}")

# Stats
if times:
    avg_time = sum(times) / len(times)
    print(f" RASPBERRY PI PERFORMANCE STATS\n")
    print(f" Total images: {len(times)}")
    print(f" Average Predict Time: {avg_time:.2f} ms")
    print(f" Estimated FPS:         {1000/avg_time:.2f} fps")