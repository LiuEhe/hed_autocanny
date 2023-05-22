## 项目描述

本项目基于HED (Holistically-Nested Edge Detection) 模型和Canny边缘检测算法，通过比较输入图像的灰度图像与HED边缘检测结果之间的均方误差（MSE），自动选择最佳的Canny边缘检测阈值参数。项目使用Python和OpenCV实现。

## 项目运行效果截图
![答案截图](output.jpg)

## 功能

- 对输入的RGB图像进行HED边缘检测
- 通过比较输入的灰度图像与HED边缘检测结果之间的均方误差（MSE），自动选择最佳的Canny边缘检测阈值参数
- 使用最佳阈值参数计算最终的Canny边缘检测结果
- 绘制MSE与sigma值关系图

## 依赖

- Python
- OpenCV
- NumPy
- scikit-image

## 使用

1. 克隆项目到本地
2. 安装所需依赖库
3. 将所需图片放入`img`文件夹
4. 在`hed_autocanny.py`中修改`IMG`变量为所需图片的路径
5. 运行`hed_autocanny.py`文件，查看并分析结果

## 注意

- 确保已安装所有依赖库
- 输入图像应为RGB图像


