from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("/home/zj/Project/MP-YOLO/MP-YOLO-8.3.7/MP-YOLO/ultralytics/cfg/models/rt-detr/rtdetr-l.yaml")

# Display model information (optional)
model.info()
print(model)
# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="yaml/02.anti_uav410.yaml", epochs=50, imgsz=640,seed = 42,batch = 8)

# Run inference with the RT-DETR-l model on the 'bus.jpg' image
# results = model("path/to/bus.jpg")