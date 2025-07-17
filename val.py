from ultralytics import YOLO

# Load a model

# model = YOLO("yolo11n.pt")
# /home/zj/Project/yolov11/ultralytics/runs/detect/yolo11nMultilittle_biasconv4_biasc3k2_startblock5_batch8/weights/best.pt
# model = YOLO("yaml/yolo11n.yaml")
# yaml_name = "yolov11n-p2_EMM"
yaml_name = "val"
model = YOLO("/home/zj/Project/MP-YOLO/MP-YOLO-8.3.7/MP-YOLO/MP-YOLO.pt")
print(model)
# Train the model/home/zj/Dataset/anti_uav410/test.txt/home/zj/Project/yolov11/ultralytics/yaml/yolo11n_useAllV1.yaml
train_results = model.val(
    name = yaml_name,
    data="yaml/02.anti_uav410.yaml",  # path to dataset YAML
    # data="yaml/01.SIRST-UAVB.yaml",
    # data="yaml/03.tiny-TUD.yaml",
    # data="yaml/00.IRSTD-1k.yaml",
    save = False,
    # conf=0.5,
)
# print(train_results)
# print(train_results.box.map)  # map50-95
# print(train_results.box.map50)  # map50
# print(train_results.box.map75)  # map75
# print(train_results.box.maps) # a list contains map50-95 of each category
# Evaluate model performance on the validation set
# metrics = model.val()

# # Perform object detection on an image
# results = model("path/to/image.jpg")
# results[0].show()

# Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model
