#!/bin/bash  

# 激活所需的虚拟环境（可选）  
conda activate yolov11  
# 或者如果你有其他需要初始化的环境，请在这里添加  

# 检查传入的参数，并根据参数执行不同的脚本  
if [ "$1" == "run1" ]; then  
    # 执行第一个训练脚本  
    echo "Running script for run1..."  
    nohup python train_Antiuav410_run1.py > nohup_run1.log 2>&1 &  
elif [ "$1" == "run2" ]; then  
    # 执行第二个训练脚本  
    echo "Running script for run2..."  
    nohup python train_Antiuav410_run2.py > nohup_run2.log 2>&1 &  
else  
    echo "Usage: ./train.sh {run1|run2}"  
    exit 1  
fi