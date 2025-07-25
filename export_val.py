from ultralytics import YOLO

model = YOLO("MP-YOLO.pt")

# Export the model to NCNN format
model.export(format="ncnn",imgsz=640,data="/home/zhang/Project/MP-YOLO/yaml/Dataset/02.anti_uav410.yaml") 

ncnn_model = YOLO("MP-YOLO_ncnn_model",task="detect")  

results = ncnn_model(source="/home/zhang/Dataset/anti_uav410/images/val/20190925_130434_1_1",imgsz=640)  