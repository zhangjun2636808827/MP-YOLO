from ultralytics import YOLO

# Load a model

# model = YOLO("yolo11n.pt")
# /home/zj/Project/yolov11/ultralytics/runs/detect/yolo11nMultilittle_biasconv4_biasc3k2_startblock5_batch8/weights/best.pt
# model = YOLO("yaml/yolo11n.yaml")
model = YOLO("yaml/yolov11n-p2.yaml")
print(model)
# Train the model/home/zj/Dataset/anti_uav410/test.txt/home/zj/Project/yolov11/ultralytics/yaml/yolo11n_useAllV1.yaml
train_results = model.train(
    name = "02.yolov11n-p2_batch8",
    data="yaml/02.anti_uav410.yaml",  # path to dataset YAML
    epochs=50,  # number of training epochs
    imgsz=640,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch= 8,  # batch size
    amp = True,#加速
    visualize = True,#可以保存每一层的特征图。 
    profile = True,#可以记录每一层的计算时间。
    lr0=0.01, 
    lrf=0.01,
    workers = 4,#线程
    patience = 10,#耐心参数（100轮无提升）
    cache = False,#是否缓存数据集。
    pretrained = False,#是否使用预训练模型。
    optimizer = "auto",#优化器类型，auto 表示自动选择。
    verbose = True,#是否打印详细信息。
    seed = 42,# 随机种子。
    deterministic=True, #是否使用确定性算法。c
    # single_cls=True,
    
    # augment = True,task=detect, mode=train, model=yaml/yolo11sam1.yaml, data=coco128.yaml, epochs=300, time=None, patience=100, batch=4, imgsz=640, save=True, save_period=-1, cache=False, device=None, workers=0, project=None, name=train30, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=E:\ProjectSpace\02.Code\03.Pycharm\17.yoloUltralytics\ultralytics\runs\detect\train30

    # flipud=0.5,  # 垂直翻转的概率  
    # fliplr=0.5,  # 水平翻转的概率  
    # mosaic=1.0,  # 是否启用 Mosaic 数据增强  
    # mixup=0.3,   # 是否启用 MixUp 数据增强  
    # degrees=30.0,  # 图片随机旋转角度范围，+/-设置最大角度  
    # translate=0.1, # 平移的比例 [0-1]  
    # scale=0.5,     # 缩放范围  
    # shear=10.0,    # 剪切的角度，+/-值  
    # perspective=0.1, # 视角变换比例 /
)

# Evaluate model performance on the validation set
metrics = model.val()

# # Perform object detection on an image
# results = model("path/to/image.jpg")
# results[0].show()

# Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model
