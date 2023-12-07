import numpy as np
import cv2
from matplotlib import pyplot as plt

def my_sobel(img):
    hs = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) #Horizontal sobel
    vs = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) #Vertical sobel
    [rows, cols] = np.shape(img)
    sobel_result = np.zeros(shape = [rows, cols])
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            Gx = (hs * img[i-1:i+2, j-1:j+2]).sum()
            Gy = (vs * img[i-1:i+2, j-1:j+2]).sum()
            sobel_result[i,j] = np.sqrt(Gx ** 2 + Gy ** 2)
            if sobel_result[i,j] > 255: #Avoid overflow
                sobel_result[i,j] = 255
            elif sobel_result[i,j] < 0:
                sobel_result[i,j] = 0
    return sobel_result

def my_LoG(img):
    LoG = np.array([[0.0, 0.0, -1.0, 0.0, 0.0],
                    [0.0, -1.0, -2.0, -1.0, 0.0],
                    [-1.0, -2.0, 16.0, -2.0, -1.0],
                    [0.0, -1.0, -2.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0, 0.0]])
    [rows, cols] = np.shape(img)
    LoG_result = np.zeros(shape = [rows, cols])
    for i in range(2, rows - 2):
        for j in range(2, cols - 2):
            LoG_result[i,j] = (LoG * img[i-2:i+3, j-2:j+3]).sum()
            if LoG_result[i,j] > 255: #Avoid overflow
                LoG_result[i,j] = 255
            elif LoG_result[i,j] < 0:
                LoG_result[i,j] = 0
    return LoG_result

def main():
    print('Convolution takes 19 seconds, please wait...')
    image1_input = cv2.imread('./HW4_test_image/image1.jpg', cv2.IMREAD_GRAYSCALE)
    image2_input = cv2.imread('./HW4_test_image/image2.jpg', cv2.IMREAD_GRAYSCALE)
    image3_input = cv2.imread('./HW4_test_image/image3.jpg', cv2.IMREAD_GRAYSCALE)
    image1_sobel = my_sobel(image1_input)
    image2_sobel = my_sobel(image2_input)
    image3_sobel = my_sobel(image3_input)
    image1_LoG = my_LoG(image1_input)
    image2_LoG = my_LoG(image2_input)
    image3_LoG = my_LoG(image3_input)
    fig1 = plt.figure(figsize=(9,9))
    fig1.canvas.manager.set_window_title('Edge Detection')
    fig1.add_subplot(3,3,1)
    plt.imshow(image1_input, cmap = 'gray')
    plt.title('Original image1')
    fig1.add_subplot(3,3,2)
    plt.imshow(image1_sobel, cmap = 'gray')
    plt.title('Sobel image1')
    fig1.add_subplot(3,3,3)
    plt.imshow(image1_LoG, cmap = 'gray')
    plt.title('LoG image1')
    fig1.add_subplot(3,3,4)
    plt.imshow(image2_input, cmap = 'gray')
    plt.title('Original image2')
    fig1.add_subplot(3,3,5)
    plt.imshow(image2_sobel, cmap = 'gray')
    plt.title('Sobel image2')
    fig1.add_subplot(3,3,6)
    plt.imshow(image2_LoG, cmap = 'gray')
    plt.title('LoG image2')
    fig1.add_subplot(3,3,7)
    plt.imshow(image3_input, cmap = 'gray')
    plt.title('Original image3')
    fig1.add_subplot(3,3,8)
    plt.imshow(image3_sobel, cmap = 'gray')
    plt.title('Sobel image3')
    fig1.add_subplot(3,3,9)
    plt.imshow(image3_LoG, cmap = 'gray')
    plt.title('LoG image3')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()