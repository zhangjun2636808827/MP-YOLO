#!/bin/bash  

# 激活所需的虚拟环境（可选）  
conda activate yolov11  
# 或者如果你有其他需要初始化的环境，请在这里添加  

# 执行训练的命令  Project/MP-YOLO/mmdetection/
nohup python train_Antiuav410.py > nohup.log 2>&1 &