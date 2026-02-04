# 手写数字 0/1 识别（MNIST + 摄像头实时推理）

本项目使用 TensorFlow/Keras 训练一个二分类模型（数字 0 与 1），并通过 OpenCV 打开摄像头进行实时识别。模型基于 MNIST 数据集的 0/1 样本训练，配合改进的图像预处理（自适应阈值、膨胀等）以适应纸面拍摄场景。

## 环境要求
- Python 3.13（建议）
- 已安装摄像头设备
- Windows（当前项目目录结构基于 Windows）

## 依赖安装
在项目目录下执行：
```bash
pip install tensorflow numpy opencv-python matplotlib
```

说明：
- 若遇到 TensorFlow 安装或版本兼容问题，建议使用 Python 3.13 并通过 `py -3.13 -m pip install tensorflow` 安装。
- OpenCV 用于摄像头访问与图像预处理。

## 运行
- 直接运行脚本：
```bash
python 1.py
```
- 或使用批处理文件：
```bash
run.bat
```

程序将先训练模型并显示训练曲线，关闭图表窗口后进入摄像头实时识别模式。

## 使用指南
1. 准备一张白纸，用较粗的黑色笔写数字 0 或 1。
2. 保持光线充足，尽量避免阴影。
3. 将纸张置于取景框中心，观察左上角“Processed”小图，确保数字清晰、笔画完整。
4. 识别结果与概率显示在画面左上方；按下 `q` 键退出。

