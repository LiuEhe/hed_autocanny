import cv2
from tqdm import tqdm
import numpy as np
from skimage.metrics import mean_squared_error 
from skimage.filters import threshold_otsu

#############################################################################
############# 如果在Jupyter Notebook环境中，此段代码只能运行一次 #################
# 定义一个名为CropLayer的类
class CropLayer(object):
    # 构造函数，用于初始化裁剪的起始和结束位置
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # 该层接收两个输入。
    # 我们需要将第一个输入blob裁剪成与第二个输入blob相同的形状（保持批大小和通道数不变）
    def getMemoryShapes(self, inputs):
        # 获取输入形状和目标形状
        inputShape, targetShape = inputs[0], inputs[1]
        # 获取批大小和通道数
        batchSize, numChannels = inputShape[0], inputShape[1]
        # 获取目标形状的高度和宽度
        height, width = targetShape[2], targetShape[3]
        # 计算裁剪的起始位置
        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        # 计算裁剪的结束位置
        self.yend = self.ystart + height
        self.xend = self.xstart + width
        # 返回裁剪后的形状
        return [[batchSize, numChannels, height, width]]

    # 前向传播函数，进行裁剪操作
    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]

# 使用cv2.dnn_registerLayer函数将自定义裁剪层注册到网络中
cv2.dnn_registerLayer('Crop', CropLayer)
##############################################################################



# 该函数使用预训练的 HED (Holistically-Nested Edge Detection) 模型来检测图像中的边缘。
# 通过使用 OpenCV 库加载模型并传递输入图像，该函数生成并返回二值化的边缘检测结果。
# 参数:
#   img_rgb (numpy.ndarray): 输入的 RGB 图像，图像尺寸应为 (height, width, 3)。
#   blur_window (int, 可选): 用于对输入图像和边缘检测结果进行高斯模糊的窗口大小。默认值为 5。
#   scalefactor (float, 可选): 输入图像在创建 blob 时的缩放系数。默认值为 1.0。
# 返回:
#   hed (numpy.ndarray): 二值化的边缘检测结果，图像尺寸与输入图像相同。
# 示例:
#   >>> img = cv2.imread("example.jpg")
#   >>> hed = get_hed(img)
#   >>> plt.imshow(hed)
def get_hed(img_rgb,blur_window=5, scalefactor=1.0):
    # 指定模型文件的路径
    model_path ='model/hed_pretrained_bsds.caffemodel'
    prototxt_path ='model/deploy.prototxt'
    # 使用OpenCV加载预训练模型
    net = cv2.dnn.readNetFromCaffe(prototxt_path,model_path)
    
    img_rgb=cv2.GaussianBlur(img_rgb,(blur_window,blur_window), 0)
    h,w = img_rgb.shape[:2]
    # 从图像创建blob
    blob = cv2.dnn.blobFromImage(img_rgb, scalefactor=scalefactor, size=(w,h),
                                mean=(105, 117, 123),
                                swapRB=False, crop=False)


    net.setInput(blob) # 将 blob 设置为网络的输入
    hed_output = net.forward() # 执行前向传递以获得边缘检测结果
    hed = hed_output[0, 0] # 从输出中提取边缘图
    hed = (255 * hed).astype("uint8") # 规范化边缘图以进行可视化
    hed=cv2.GaussianBlur(hed,(blur_window,blur_window), 0)
    hed=cv2.threshold(hed,0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    return hed




# 该函数通过比较输入的灰度图像与 HED 边缘检测结果之间的均方误差 (MSE)，自动选择最佳的 Canny 边缘检测阈值参数。使用最佳阈值参数计算最终的边缘检测结果。
# 参数:
#   img_gray (numpy.ndarray): 输入的灰度图像，图像尺寸应为 (height, width)。
#   hed (numpy.ndarray): 使用 HED 模型生成的边缘检测结果，图像尺寸应与输入灰度图像相同。
# 返回:
#   edge_img (numpy.ndarray): 使用最佳阈值参数生成的 Canny 边缘检测结果。
#   best_params (tuple): 包含最佳阈值参数的元组，格式为 (best_sigma, min_mse, lower, upper)。
#   plot_data (tuple): 包含用于绘制 MSE 与 sigma 值关系图的数据，格式为 (sigma_values, mse_values)。
# 示例:
# >>> img_gray = cv2.cvtColor(cv2.imread("example.jpg"), cv2.COLOR_BGR2GRAY)
# >>> hed = get_hed(cv2.imread("example.jpg"))
# >>> edge_img, best_params, plot_data = auto_canny(img_gray, hed)
# >>> plt.imshow(edge_img)
def auto_canny(img_gray,hed):
    img_gray=cv2.GaussianBlur(img_gray,(5,5), 0)

    # 定义用于测试的sigma值范围（100个sigma值，范围从0.001到1.0）
    sigma_values = np.linspace(0.001, 1.0, 100)
    best_sigma = None
    min_mse = float('inf')
    mse_values = []
    median = np.median(img_gray)

    # 循环遍历sigma值，并计算每个值的MSE（使用tqdm显示进度条）
    for sigma in tqdm(sigma_values, desc="搜索最佳sigma", unit="sigma"):
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        auto_canny = cv2.Canny(img_gray, lower, upper)
        
        mse = mean_squared_error(auto_canny, hed)
        mse_values.append(mse)  
        
        if mse < min_mse:
            min_mse = mse
            best_sigma = sigma

    # 使用最佳sigma值计算最终的边缘检测结果
    median = np.median(img_gray)
    lower = int(max(0, (1.0 - best_sigma) * median))
    upper = int(min(255, (1.0 + best_sigma) * median))
    edge_img = cv2.Canny(img_gray, lower, upper)

    return edge_img,(best_sigma,min_mse,lower,upper),(sigma_values,mse_values)