from ultralytics import YOLO  
import os  

# Load a pretrained YOLO model  
model = YOLO("/home/zj/Project/MP-YOLO/MP-YOLO-8.3.7/MP-YOLO/runs/detect/02.yolo11n-p1_MPBlock_MPCBlock_EMBlock3/weights/best.pt")  

# 指定包含图像路径的TXT文件路径  
txt_file_path = "/home/zj/Dataset/anti_uav410/val.txt"  # 替换为你的实际路径  
source_base_path = "/home/zj/Dataset/anti_uav410/images/val/"  # 图像源路径的基础部分  

# 读取TXT文件，并对每个图像路径进行检测  
with open(txt_file_path, 'r') as file:  
    for line in file:  
        image_path = line.strip()  # 去除行首尾空格和换行符  
        if not os.path.exists(image_path):  
            print(f"警告: 找不到图像文件 {image_path}")  
            continue  

        # 生成保存路径  
        relative_path = image_path.replace(source_base_path, "")  # 获取相对路径  
        # directory_path = os.path.dirname(relative_path)  # 获取目录部分  
        # print("relative_path=",directory_path)
        save_path = os.path.join("mpyolo_ani_uav410", relative_path)  # 只保留目录结构  
        print("save_path=",save_path)
        # 确保保存的目录存在  
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  

        # Run inference on the image  
        results = model(image_path)  
        for result in results:
            result.save(filename=save_path,conf=0.0)  # save to disk

print("检测完成。")