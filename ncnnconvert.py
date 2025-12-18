from ultralytics import YOLO

model = YOLO("thermalnano.pt")  # model to convert

# Export to NCNN specifying the correct task
model.export(format="ncnn", task="segment", half=True, imgsz=1024) 