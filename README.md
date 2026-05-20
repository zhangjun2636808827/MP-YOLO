# MP-YOLO

## Installation
`conda` virtual environment is recommended. 
```
conda create -n mpyolo python=3.9
conda activate mpyolo
pip install -r requirements.txt
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

```
## Dataset
1. [IRSTD-1K](https://github.com/RuiZhang97/ISNet)  
2. [SIRST-UAVB](https://github.com/JN-Yang/PConv-SDloss-Data)  
3. [Anti-UAV410](https://github.com/HwangBo94/Anti-UAV410#)  
4. [TUD](https://pan.baidu.com/s/1YQTDwKCqQHZi4w5GgU7aCA?pwd=in6j)  

## Acknowledgement

The code is based on [ultralytics == 8.3.7](https://github.com/ultralytics/ultralytics). Thanks for their excellent work!


## Citation

If our code or models help your work, please cite our paper:
```BibTeX
@article{zhang2026mp,
  title={MP-YOLO: a multi-padding strategy framework for enhanced infrared UAV detection},
  author={Zhang, Jun and Liu, Chun},
  journal={Journal of Electronic Imaging},
  volume={35},
  number={2},
  pages={023033--023033},
  year={2026},
  publisher={Society of Photo-Optical Instrumentation Engineers}
}

```
