import os  
import json  
from PIL import Image  
   
# 设置数据集路径  
output_dir = "/home/zj/Dataset/anti_uav410"   # 修改为 YOLO 格式的数据集路径；  
dataset_path = "/home/zj/Dataset/anti_uav410"  # 修改为你想输出的 COCO 格式数据集路径  
images_path = os.path.join(dataset_path, "images")  
labels_path = os.path.join(dataset_path, "labels")  
   
# 类别映射  
categories = [  
    {"id": 0, "name": "uav"},  
    # {"id": 1, "name": "bird"},  
    # 添加更多类别  
]  
   
# YOLO 格式转 COCO 格式的函数  
def convert_yolo_to_coco(x_center, y_center, width, height, img_width, img_height):  
    x_min = (x_center - width / 2) * img_width  
    y_min = (y_center - height / 2) * img_height  
    width = width * img_width  
    height = height * img_height  
    return [x_min, y_min, width, height]  

# 初始化 COCO 数据结构  
def init_coco_format():  
    return {  
        "images": [],  
        "annotations": [],  
        "categories": categories  
    }  

# 处理每个数据集分区  
for split in ['train', 'test', 'val']:  
    coco_format = init_coco_format()  
    annotation_id = 1  

    split_path = os.path.join(images_path, split)  

    # 遍历子文件夹及其内容  
    for root, dirs, files in os.walk(split_path):  
        for img_name in files:  
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  
                img_path = os.path.join(root, img_name)  
                label_name = img_name.replace("jpg", "txt").replace("jpeg", "txt").replace("png", "txt")  
                label_path = os.path.join(labels_path, split,os.path.basename(root), label_name)  

                img = Image.open(img_path)  
                img_width, img_height = img.size  
                print("dirs=",label_path)
                image_info = {  
                    "file_name": os.path.basename(root)+"/"+img_name,  
                    "id": len(coco_format["images"]),  
                    "width": img_width,  
                    "height": img_height  
                }  
                coco_format["images"].append(image_info)  

                if os.path.exists(label_path):  
                    with open(label_path, "r") as file:  
                        # 检查文件是否为空  
                        content = file.read()  # 读取文件内容
                        
                        if content.strip():  # 如果内容非空，继续处理  
                            
                            for line in content.splitlines():  
                                print("content=",line)
                                category_id, x_center, y_center, width, height = map(float, line.split())  
                               
                                bbox = convert_yolo_to_coco(x_center, y_center, width, height, img_width, img_height)  
                                annotation = {  
                                    "id": annotation_id,  
                                    "image_id": image_info["id"],  
                                    "category_id": int(category_id),  
                                    "bbox": bbox,  
                                    "area": bbox[2] * bbox[3],  
                                    "iscrowd": 0  
                                }  
                                coco_format["annotations"].append(annotation)  
                                annotation_id += 1  

    # 为每个分区保存 JSON 文件  
    with open(os.path.join(output_dir, f"{split}_coco_format.json"), "w") as json_file:  
        json.dump(coco_format, json_file, indent=4)