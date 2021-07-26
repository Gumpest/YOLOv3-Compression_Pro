# YOLOv3-Compression_Pro
Prune, Quantization and Slight Model

# YOLOv3-Compression



| Model                           | Parameters | mAP@0.5 | FPS@CPU                |
| ------------------------------- | ---------- | ------- | ---------------------- |
| YOLOv3-tiny                     | 8.67M      | 99.30%  | 16.7  （Fuse BN 18.1） |
| YOLOv3-tiny + prune@0.4         | 3.20M      | 98.70%  | 33.63                  |
| YOLOv3-tiny + MobilenetV3-small | 2.86M      | 97.10%  | 35.36                  |
|                                 |            |         |                        |



### 检测演示效果

![000139_00](https://user-images.githubusercontent.com/56063339/126035552-ff168821-e6f7-493e-bf67-0f56a6331fc5.png)


![wurenji-001](https://user-images.githubusercontent.com/56063339/126035556-024364d7-613a-4044-9852-3c9a2ef9c69a.png)



### 模型压缩

## Prune

流程：

未稀疏与稀疏后的γ分布

![image-20210517160316575](https://user-images.githubusercontent.com/56063339/126035606-f59ab2f3-7aaa-4143-bdc0-c390810e16f7.png)



1. γ稀疏化训练 -->  稀疏后的模型

```shell
python3 train.py --data data/uavcut.data --cfg cfg/uavcut.cfg --epochs 30 -sr
```

2. Slimming 剪枝  -->  剪枝后的`.cfg`文件与权重文件

```shell
python Tiny_yolo_prune.py
```

3. Retrain提升mAP

```shell
python3 train.py --data data/uavcut.data --cfg cfg/prune_0.4_uavcut.cfg --epochs 30 --weights weights/yolov3_tiny_uavcut_pruning_0.4percent.pt
```

## Substitute YOLO Backbone with Slight Model

2.1 Mobilenetv3-small替换darknet53

- 替换至11个bottleneck，改写cfg文件

- 添加depthwise算子

- 添加seblock算子 + SeBlock类

- 修改darknet中的forward函数

```shell
python3 train.py --data data/uavcut.data --cfg cfg/yolov3-tiny-mobilenet-small.cfg --epochs 30 --weights weights/yolov3_tiny_uavcut_mobilenetsmall.pt
```

