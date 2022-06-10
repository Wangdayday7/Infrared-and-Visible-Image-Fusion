import cv2
import numpy as np
import math

import skimage.measure

# 参数0表示读入即为灰度图
img = cv2.imread('Test_vi/103.bmp',0)
#img_fu = cv2.imread('FusionNet_Results/04.bmp', 0)
img_fu = cv2.imread('evaduation/STD (4).bmp', 0)
# 结构相似性
from skimage.metrics import structural_similarity
(score,diff) = structural_similarity(img,img_fu,full = True)
diff = (diff *255).astype("uint8")
print("SSIM={}".format(score))


# 峰值信噪比
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
peak_signal_noise_ratio = skimage.metrics.peak_signal_noise_ratio(img, img_fu, data_range=None)
print('peak_signal_noise_ratio=', peak_signal_noise_ratio)

# 均方误差

from skimage.metrics import mean_squared_error as compare_mse
mse = compare_mse(img, img_fu)
print('mean_squared_error=', mse)

# 均方根误差
rmse = math.sqrt(mse)
print('root_mean_squared_error=', rmse)

# 信息熵
entropy = skimage.measure.shannon_entropy(img_fu, base=2)
print('entropy=', entropy)

#空间频率
def spatialF(image):
    M = image.shape[0]
    N = image.shape[1]

    cf = 0
    rf = 0

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            dx = float(image[i, j - 1]) - float(image[i, j])
            rf += dx ** 2
            dy = float(image[i - 1, j]) - float(image[i, j])
            cf += dy ** 2

    RF = math.sqrt(rf / (M * N))
    CF = math.sqrt(cf / (M * N))
    SF = math.sqrt(RF ** 2 + CF ** 2)

    return SF



def getMI(im1,im2):
    #im1 = im1.astype('float')
    #im2 = im2.astype('float')

    hang, lie = im1.shape
    count = hang*lie
    N = 256

    h = np.zeros((N,N))

    for i in range(hang):
        for j in range(lie):
            h[im1[i,j],im2[i,j]] = h[im1[i,j],im2[i,j]]+1

    h = h/np.sum(h)

    im1_marg = np.sum(h,axis=0)
    im2_marg = np.sum(h, axis=1)

    H_x = 0
    H_y = 0

    for i in range(N):
        if(im1_marg[i]!=0):
            H_x = H_x + im1_marg[i]*math.log2(im1_marg[i])

    for i in range(N):
        if(im2_marg[i]!=0):
            H_x = H_x + im2_marg[i]*math.log2(im2_marg[i])

    H_xy = 0

    for i in range(N):
        for j in range(N):
            if(h[i,j]!=0):
                H_xy = H_xy + h[i,j]*math.log2(h[i,j])

    MI = H_xy-H_x-H_y

    return MI


if __name__ == '__main__':
    print('spatialF=', spatialF(img_fu))
    print('MI=', getMI(img, img_fu))

(mean,stddv) = cv2.meanStdDev(img_fu)
#输出平均值
print('mean=', mean)
#输出标准差
print('stddv', stddv)
