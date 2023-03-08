车牌识别,

更新，将yolo v5部分添加进入该项目，yolo v5使用的6.0的版本，权重直接使用其权重，形成完整的车辆检测与车牌检测识别的完整项目。

## 要求

Python 3.8 或更晚与所有[要求.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)依赖项安装，包括.要安装运行：`torch>=1.7`

```
$ pip install -r requirements.txt
```

## 教程

车牌检测数据集来源于中科大的CCPD2019开源数据集，可在DetectDatase类中，修改路径，换为自己数据再次训练。

车牌识别，使用的是生成的虚假数据。如需优化，可根据需要添加额外真实数据集，或者更改数据增强方式，以及修改模型。

## 环境

任何最新验证环境中运行（所有依赖项（包括[CUDA](https://developer.nvidia.com/cuda)/ CUDNN、Python和[PyTorch](https://pytorch.org/)预装）：



## 推理

要对test_image文件夹下的示例图像进行推理，

```
$ python kenshutsu.py
```

## 关于我们

## 联系