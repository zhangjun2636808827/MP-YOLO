from ultralytics import YOLO

# Load a model
yaml_name = "MP-YOLO"
model = YOLO("yaml/" + yaml_name + ".yaml")

print(model)

train_results = model.train(
    name = "00."+yaml_name,
    project = "runs/Comparative",  # project name
    data="yaml/Dataset/00.IRSTD-1k.yaml",  # path to dataset YAML
    epochs=700,  # number of training epochs
    imgsz=640,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch= 16,  # batch size
    amp = True,#加速
    visualize = True,#可以保存每一层的特征图。 
    profile = True,#可以记录每一层的计算时间。
    lr0=0.01, 
    lrf=0.01,
    workers = 4,#线程
    patience = 70,#耐心参数（100轮无提升）
    cache = False,#是否缓存数据集。ADDDDDDDDDDDJDKDWAW
    optimizer = "auto",#优化器类型，auto 表示自动选择。
    verbose = True,#是否打印详细信息。
    seed = 38,# 随机种子。
    deterministic=True, #是否使用确定性算法。c
    save_period = 1,

)

# Evaluate model performance on the validation set
metrics = model.val()

