数据集下载地址[百度网盘(提取码:ippg)](https://pan.baidu.com/s/1odgmskNaLG-jwwanlbsNHA)，下载后解压到`wrist_images`文件夹下。
在[YOLO](https://pjreddie.com/darknet/yolo/)下载一个weights文件，放在跟目录下，命名为yolov3.weights。
运行
~~~
python train.py
~~~
进行训练。

训练完成后，运行
~~~
python yolo_inference_miou.py --image
~~~
进行模型评价。测试结果：平均IOU为0.66，最大IOU为0.95。数据集只有较少的图片，扩大数据集的数量可得到更好的结果。