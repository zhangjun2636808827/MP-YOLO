from ultralytics import YOLO

# Load a model
yaml_name = "val"
model = YOLO("/home/zj/Project/MP-YOLO/MP-YOLO-8.3.7/MP-YOLO/runs/Comparative/00.L-MP-YOLOn/weights/epoch70.pt")
print(model)

train_results = model.val(
    name = yaml_name,
    # data="yaml/Dataset/02.anti_uav410.yaml",
    # data="yaml/Dataset/01.SIRST-UAVB.yaml",
    data="yaml/Dataset/00.IRSTD-1k.yaml",
    # data="yaml/Dataset/03.TUD.yaml",
    batch = 8,
    workers = 4,#线程
    amp = True,#加速
    save = False,
)
print(train_results.results_dict)