import cv2
from matplotlib import pyplot as plt
import numpy as np

#Power-Law (Gamma) Transformation
def gamma_correction(input_img, const, gamma):
    if (const > 0) & (gamma > 0.0):
        output_img = const * ((input_img / 255) ** gamma) * 255 #Divided 255 and multiplies gamma
        return np.array(output_img).astype(int) #Convert all float value in the array into integer
    else:
        return input_img

def myHistEq(input_img):                                       #Histogram(1D array, bins, interval of bins' range)
    hist, bins = np.histogram(input_img.ravel(), 256, [0,255]) #Return 1D array (count pixels), bins edges array (levels), and other outputs. 
    pdf = hist / input_img.size                                #Calculate pdf of each bins
    cdf = pdf.cumsum()                                         #Calculate cdf of each bins
    return np.around(cdf * 255).astype(np.uint8)[input_img]

def myLaplacian(input_img, kernel):
    Laplacian_operator = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    rows = len(input_img)
    cols = len(input_img[0])
    result = np.arange(rows*cols*3).reshape(rows,cols,3)
    input_img = np.transpose(input_img)
    for i in range(0, rows):
        for j in range(0, cols):
            for k in range(0, 3):
                if (i < rows - 1) & (i > 0) & (j < rows - 1) & (j > 0):
                    result[i][j][k] = sum(sum(kernel*(np.transpose([index[i-1:i+2] for index in input_img[k][j-1:j+2]]))))
                if result[i][j][k] > 255:
                    result[i][j][k] = 255
                elif result[i][j][k] < 0:
                    result[i][j][k] = 0
    return result

Laplacian_first = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
Laplacian_second = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
Laplacian_third = np.array([[1, -2, 1], [-2, 5, -2], [1, -2, 1]])
KernelBank = {
    'Laplacian_first': Laplacian_first,
    'Laplacian_second': Laplacian_second,
    'Laplacian_third': Laplacian_third
}


def main():
    print('It will take approximately 27 seconds to launch plots, please wait...')
    Jetplane_file = './HW1_test_image/blurry_moon.tif'
    Lake_file = './HW1_test_image/Lake.bmp'
    Peppers_file = './HW1_test_image/Peppers.bmp'
    Jetplane_img = cv2.imread(Jetplane_file) #Input image value [0, 255]
    Lake_img = cv2.imread(Lake_file)
    Peppers_img = cv2.imread(Peppers_file)

    #Laplacian Sharpening
    fig4 = plt.figure(figsize = (16, 9))
    fig4.canvas.manager.set_window_title('Laplacian Image Sharpening')
    rows = 3
    cols = 4
    fig4.add_subplot(rows, cols, 1)
    plt.imshow(Jetplane_img)
    plt.title('Original Jetplane')
    fig4.add_subplot(rows, cols, 2)
    Jetplane_output = myLaplacian(Jetplane_img, KernelBank["Laplacian_first"])
    plt.imshow(Jetplane_output)
    plt.title('Laplacian first-order derivative Jetplane')
    fig4.add_subplot(rows, cols, 3)
    Jetplane_output = myLaplacian(Jetplane_img, KernelBank["Laplacian_second"])
    plt.imshow(Jetplane_output)
    plt.title('Laplacian second-order derivative Jetplane')
    fig4.add_subplot(rows, cols, 4)
    Jetplane_output = myLaplacian(Jetplane_img, KernelBank["Laplacian_third"])
    plt.imshow(Jetplane_output)
    plt.title('Laplacian third-order derivative Jetplane')
    fig4.add_subplot(rows, cols, 5)
    plt.imshow(Lake_img)
    plt.title('Original Lake')
    fig4.add_subplot(rows, cols, 6)
    Lake_output = myLaplacian(Lake_img, KernelBank["Laplacian_first"])
    plt.imshow(Lake_output)
    plt.title('Laplacian first-order derivative Lake')
    fig4.add_subplot(rows, cols, 7)
    Lake_output = myLaplacian(Lake_img, KernelBank["Laplacian_second"])
    plt.imshow(Lake_output)
    plt.title('Laplacian second-order derivative Lake')
    fig4.add_subplot(rows, cols, 8)
    Lake_output = myLaplacian(Lake_img, KernelBank["Laplacian_third"])
    plt.imshow(Lake_output)
    plt.title('Laplacian third-order derivative Lake')
    fig4.add_subplot(rows, cols, 9)
    plt.imshow(Peppers_img)
    plt.title('Original Peppers')
    fig4.add_subplot(rows, cols, 10)
    Peppers_output = myLaplacian(Peppers_img, KernelBank["Laplacian_first"])
    plt.imshow(Peppers_output)
    plt.title('Laplacian first-order derivative  Peppers')
    fig4.add_subplot(rows, cols, 11)
    Peppers_output = myLaplacian(Peppers_img, KernelBank["Laplacian_second"])
    plt.imshow(Peppers_output)
    plt.title('Laplacian second-order derivative Peppers')   
    fig4.add_subplot(rows, cols, 12)
    Peppers_output = myLaplacian(Peppers_img, KernelBank["Laplacian_third"])
    plt.imshow(Peppers_output)
    plt.title('Laplacian third-order derivative Peppers')   
    plt.tight_layout()

#Histogram Equalization
    fig2 = plt.figure(figsize = (18, 9))
    fig2.canvas.manager.set_window_title('Histogram Equalization')
    rows = 3
    cols = 4
    fig2.add_subplot(rows, cols, 1)
    plt.imshow(Jetplane_img)
    plt.title('Original Jetplane')
    fig2.add_subplot(rows, cols, 2)
    plt.hist(Jetplane_img.ravel(), 256, [0, 255])
    plt.title('Histogram of Original Jetplane')
    fig2.add_subplot(rows, cols, 3)
    Jetplane_output = myHistEq(Jetplane_img)
    plt.imshow(Jetplane_output)
    plt.title('Histogram Equalization Jetplane')
    fig2.add_subplot(rows, cols, 4)
    plt.hist(Jetplane_output.ravel(), 256, [0, 255])
    plt.title('Histogram of Histogram Equalization Jetplane')
    fig2.add_subplot(rows, cols, 5)
    plt.imshow(Lake_img)
    plt.title('Original Lake')
    fig2.add_subplot(rows, cols, 6)
    plt.hist(Lake_img.ravel(), 256, [0, 255])
    plt.ylim(0, 8000)   #Make subplot easier to see
    plt.title('Histogram of Original Lake')
    fig2.add_subplot(rows, cols, 7)
    Lena_output = myHistEq(Lake_img)
    plt.imshow(Lake_output)
    plt.title('Histogram Equalization Lake')
    fig2.add_subplot(rows, cols, 8)
    plt.hist(Lake_output.ravel(), 256, [0, 255])
    plt.ylim(0, 8000)   #Make subplot easier to see
    plt.title('Histogram of Histogram Equalization Lake')
    fig2.add_subplot(rows, cols, 9)
    plt.imshow(Peppers_img)
    plt.title('Original Peppers')
    fig2.add_subplot(rows, cols, 10)
    plt.hist(Peppers_img.ravel(), 256, [0, 255])
    plt.ylim(0, 12000)   #Make subplot easier to see
    plt.title('Histogram of Original Peppers')
    fig2.add_subplot(rows, cols, 11)
    Peppers_output = myHistEq(Peppers_img)
    plt.imshow(Peppers_output)
    plt.title('Histogram Equalization Peppers')
    fig2.add_subplot(rows, cols, 12)
    plt.hist(Peppers_output.ravel(), 256, [0, 255])
    plt.ylim(0, 12000)  #Make subplot easier to see
    plt.title('Histogram of Histogram Equalization Peppers')
    plt.tight_layout()

    #Gamma Correction
    fig1 = plt.figure(figsize = (16, 9))
    fig1.canvas.manager.set_window_title('Power-Law (Gamma) Transformation')
    rows = 3
    cols = 4
    fig1.add_subplot(rows, cols, 1)
    plt.imshow(Jetplane_img)
    plt.title('Original Jetplane')
    fig1.add_subplot(rows, cols, 2)
    Jetplane_output = gamma_correction(Jetplane_img, 1, 0.4)
    plt.imshow(Jetplane_output)
    plt.title('Power-Law Jetplane' + ' (γ = 0.4)')
    fig1.add_subplot(rows, cols, 3)
    Jetplane_output = gamma_correction(Jetplane_img, 1, 1.2)
    plt.imshow(Jetplane_output)
    plt.title('Power-Law Jetplane' + ' (γ = 1.2)')
    fig1.add_subplot(rows, cols, 4)
    Jetplane_output = gamma_correction(Jetplane_img, 1, 2.5)
    plt.imshow(Jetplane_output)
    plt.title('Power-Law Jetplane' + ' (γ = 2.5)')
    fig1.add_subplot(rows, cols, 5)
    plt.imshow(Lake_img)
    plt.title('Original Lake')
    fig1.add_subplot(rows, cols, 6)
    Lake_output = gamma_correction(Lake_img, 1, 0.4)
    plt.imshow(Lake_output)
    plt.title('Power-Law Lake' + ' (γ = 0.4)')
    fig1.add_subplot(rows, cols, 7)
    Lake_output = gamma_correction(Lake_img, 1, 1.2)
    plt.imshow(Lake_output)
    plt.title('Power-Law Lake' + ' (γ = 1.2)')
    fig1.add_subplot(rows, cols, 8)
    Lake_output = gamma_correction(Lake_img, 1, 2.5)
    plt.imshow(Lake_output)
    plt.title('Power-Law Lake' + ' (γ = 2.5)')
    fig1.add_subplot(rows, cols, 9)
    plt.imshow(Peppers_img)
    plt.title('Original Peppers')
    fig1.add_subplot(rows, cols, 10)
    Peppers_output = gamma_correction(Peppers_img, 1, 0.4)
    plt.imshow(Peppers_output)
    plt.title('Power-Law Peppers' + ' (γ = 0.4)')
    fig1.add_subplot(rows, cols, 11)
    Peppers_output = gamma_correction(Peppers_img, 1, 1.2)
    plt.imshow(Peppers_output)
    plt.title('Power-Law Peppers' + ' (γ = 1.2)')
    fig1.add_subplot(rows, cols, 12)
    Peppers_output = gamma_correction(Peppers_img, 1, 2.5)
    plt.imshow(Peppers_output)
    plt.title('Power-Law Peppers' + ' (γ = 2.5)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()