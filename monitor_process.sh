#!/bin/bash  

# 要监控的进程名称部分  
PROCESS_NAME="train_TUD.py"  
# 要运行的下一个脚本命令  
NEXT_COMMAND="nohup python train_TUD.py > tud.log 2>&1 &"  

# 无限循环，监控进程  
while true; do  
    # 检查指定的进程是否存在  
    if pgrep -f "$PROCESS_NAME" > /dev/null; then  
        echo "$PROCESS_NAME 进程正在运行..."  
        sleep 5  # 每隔5秒检查一次  
    else  
        echo "$PROCESS_NAME 进程已结束."  
        echo "运行下一个脚本: $NEXT_COMMAND"  
        eval $NEXT_COMMAND  # 执行下一个脚本  
        break  # 退出监控循环  
    fi  
done