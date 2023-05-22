import cv2
import matplotlib.pyplot as plt
from util import get_hed, auto_canny


IMG = './img/pebbles.jpg'
img_rgb = cv2.imread(IMG, 1)
img_gray = cv2.imread(IMG, 0)

# 通过HED算法获取图像的边缘图
hed = get_hed(img_rgb)

# 基于hed得到图像的最佳sigma的边缘图以及相关设定
edge_img,setting,plot_xy = auto_canny(img_gray,hed)

# 获取最佳sigma, 最小的MSE, 上下限
best_sigma,min_mse,lower,upper = setting

# 获取实验过的所有sigma列表作为x轴，每一个MSE列表作为y轴
sigma_values,mse_values = plot_xy


# 绘制结果
fig, axs = plt.subplots(1, 4, figsize=(20,5))
# 在第一个子图中显示原始图像
axs[0].imshow(img_rgb)
axs[0].set_title('Original', fontsize=15)
axs[0].axis('off')
# 在第二个子图中显示原始图像
axs[1].imshow(hed,cmap='gray')
axs[1].set_title('HED', fontsize=15)
axs[1].axis('off')
# 在第二个子图中显示使用最佳sigma值得到的边缘检测结果
axs[2].imshow(edge_img, cmap='gray')
axs[2].set_title(f'Best Sigma: {best_sigma:.3f}\nMin MSE: {min_mse:.3f}\n[{lower}, {upper}]', fontsize=15)
axs[2].axis('off')
# 在第三个子图中绘制MSE与Sigma的关系曲线
axs[3].plot(sigma_values, mse_values, label='MSE')
axs[3].set_xlabel('Sigma')
axs[3].set_ylabel('MSE')
axs[3].set_title('MSE & Sigma', fontsize=15)
axs[3].legend()
# 显示子图
plt.tight_layout()
plt.show()