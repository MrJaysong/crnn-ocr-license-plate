# 基于 CRNN 架构的轻量级车牌识别OCR：从数据准备到模型部署全流程解析(包含源码+数据集(25万张)+训练模型)

## 1、简介
车牌识别（License Plate Recognition, LPR）是智能交通、安防监控和车辆管理中的核心技术之一。本文介绍一个基于 CRNN（Convolutional Recurrent Neural Network） 架构开发的完整开源车牌识别系统，涵盖从原始标注数据处理、训练集构建、模型训练、验证评估到 ONNX 模型导出与推理的全流程。

CRNN OCR具有以下特点：

- 端到端识别：直接输入裁剪后的车牌图像，输出识别字符串；

- 支持中文字符：覆盖中国车牌常用省份简称、字母、数字及特殊字符（如“警”、“学”、“港”、“挂”等）；

- 模块化设计：清晰分离数据预处理、模型定义、训练逻辑与推理流程；

- 工业友好：支持 PyTorch 训练 + ONNX 导出 + ONNX Runtime 高效推理；

- 灵活配置：提供 small / medium / large 三种模型规模，适应不同算力场景。

## 2、数据预处理
分别处理 CBLPRD-330k 和 CRPD 两大公开车牌数据集；

制作步骤：

- 读取原始标注文件（含车牌坐标或直接车牌号）；

- 裁剪/缩放车牌区域为统一尺寸（默认 168×48）；

- 按车牌号命名图像文件（如 京A12345.jpg）；

- 自动过滤非法字符（不在 plateName 中的车牌）；

- 支持中文路径读写（使用 cv2.imdecode + np.fromfile）。

 输出：纯车牌图像文件夹，文件名即标签。

执行脚本
```
python cblprd_to_ocr.py
```
实现CBLPR数据集转成CRNN OCR 训练的数据集

执行脚本
```
python crpd_to_ocr.py
```
实现CRPD数据集转成CRNN OCR 训练的数据集

标签文件
- 将上述图像文件夹转换为训练所需的 标签文件（.txt）；
- 格式：图像路径 字符索引1 字符索引2 ...
- 示例：
```
datasets/train/京A12345.jpg 1 52 43 44 45 46 47
```
自动删除含非法字符的图片，保证数据纯净。

## 3、训练示例
```
python train.py
```
<img width="1899" height="888" alt="image" src="https://github.com/user-attachments/assets/70d9cd7e-aea2-4225-9333-cb7cf0482986" />
<img width="1899" height="908" alt="image" src="https://github.com/user-attachments/assets/55e0afcc-7ecf-4629-9770-bf8e029592d9" />

## 4、模型导出
```
python export.py
```
## 5、模型评估
```
python val.py
```
## 6、性能表现
```
(myvenv) nvr@lan:~/work/crnn/crnn_plate_ocr$ python val.py --model_path=run/train/small_epochs100/best.pth --image_path=images
使用PyTorch模型: run/train/small_epochs100/best.pth
OK: 真实:藏QFE6081 识别:藏QFE6081
OK: 真实:鄂PDT5402 识别:鄂PDT5402
OK: 真实:黑GDE5694 识别:黑GDE5694
OK: 真实:藏TFS6082 识别:藏TFS6082
OK: 真实:藏ZFR3218 识别:藏ZFR3218
OK: 真实:甘HFK7578 识别:甘HFK7578
OK: 真实:甘ND04171 识别:甘ND04171
OK: 真实:鄂YFP8993 识别:鄂YFP8993
OK: 真实:鄂SDJ2358 识别:鄂SDJ2358
OK: 真实:藏PFD3364 识别:藏PFD3364
OK: 真实:甘QD31615 识别:甘QD31615
OK: 真实:鄂HF31708 识别:鄂HF31708
OK: 真实:川ADJ6810 识别:川ADJ6810
OK: 真实:川A0JH81 识别:川A0JH81
OK: 真实:京C61493F 识别:京C61493F
OK: 真实:京C38798D 识别:京C38798D
OK: 真实:川A0K70D 识别:川A0K70D
OK: 真实:琼BU2232 识别:琼BU2232
OK: 真实:藏A2VGD0 识别:藏A2VGD0
OK: 真实:陕F5JM3D 识别:陕F5JM3D
OK: 真实:川HFN8482 识别:川HFN8482
OK: 真实:京BQWK5Q 识别:京BQWK5Q
Total: 22, Success: 22, Failed: 0, Acc: 1.0000
```
## 7、数据集
<img width="595" height="107" alt="image" src="https://github.com/user-attachments/assets/6c66c85d-bd02-44a9-9751-199c4244705c" />

统计数量

- test_lprnet ：总 105 张
- val_lprnet：总 690 张
- train_lprnet：总 254703 张

test_lprnet 部分测试集
<img width="972" height="783" alt="image" src="https://github.com/user-attachments/assets/1307ea42-3246-4b38-84a0-40fc7d34e84a" />

val_lprnet 部分验证集
<img width="963" height="736" alt="image" src="https://github.com/user-attachments/assets/d3cb2bd5-ef39-4d05-b82c-093c20cfadf0" />

train_lprnet 部分训练集
<img width="952" height="736" alt="image" src="https://github.com/user-attachments/assets/bffbdc3b-5011-4b1e-adb9-091782717e97" />

## 详细介绍请看文章
https://blog.csdn.net/u011425939/article/details/157723361?spm=1011.2415.3001.5331
