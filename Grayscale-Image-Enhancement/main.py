import cv2
from matplotlib import pyplot as plt
import numpy as np
from plotting_utils import plot_images, plot_histograms
from image_enhancement import laplacian, hist_eq, gamma_correction

def main():
    Jetplane_file = './Image/Jetplane.bmp'
    Lake_file = './Image/Lake.bmp'
    Peppers_file = './Image/Peppers.bmp'

    # Gamma Correction
    images = []
    titles = []

    for filename in [Jetplane_file, Lake_file, Peppers_file]:
        img = cv2.imread(filename)
        images.append(img)
        titles.append(f'Original {filename.split("/")[-1].split(".")[0]}')

        for gamma_val in [0.4, 1.2, 2.5]:
            output = gamma_correction(img, 1, gamma_val)
            images.append(output)
            titles.append(f'Power-Law {filename.split("/")[-1].split(".")[0]} (γ = {gamma_val})')

    plot_images(images, titles, rows=3, cols=4, figsize=(16, 9), window_title='Power-Law (Gamma) Transformation')

    # Histogram Equalization
    images = []
    titles = []

    for filename in [Jetplane_file, Lake_file, Peppers_file]:
        img = cv2.imread(filename)
        images.append(img)
        titles.append(f'Original {filename.split("/")[-1].split(".")[0]}')

        output = hist_eq(img)
        images.append(output)
        titles.append(f'Histogram Equalization {filename.split("/")[-1].split(".")[0]}')

    plot_histograms(images, titles, rows=3, cols=4, figsize=(18, 9), ylims=[], window_title='Histogram Equalization')

    # Laplacian Sharpening
    images = []
    titles = []

    # Define Laplacian Kernels
    Laplacian_first = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    Laplacian_second = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    Laplacian_third = np.array([[1, -2, 1], [-2, 5, -2], [1, -2, 1]])

    KernelBank = {
        'Laplacian_first': Laplacian_first,
        'Laplacian_second': Laplacian_second,
        'Laplacian_third': Laplacian_third
    }
    
    kernels = [KernelBank["Laplacian_first"], KernelBank["Laplacian_second"], KernelBank["Laplacian_third"]]
    kernel_titles = ["Laplacian first-order derivative", "Laplacian second-order derivative", "Laplacian third-order derivative"]

    for filename in [Jetplane_file, Lake_file, Peppers_file]:
        img = cv2.imread(filename)
        images.append(img)
        titles.append(f'Original {filename.split("/")[-1].split(".")[0]}')

        for kernel, kernel_title in zip(kernels, kernel_titles):
            output = laplacian(img, kernel)
            images.append(output)
            titles.append(f'{kernel_title} {filename.split("/")[-1].split(".")[0]}')

    plot_images(images, titles, rows=3, cols=4, figsize=(16, 9), window_title='Laplacian Image Sharpening')

if __name__ == "__main__":
    main()
