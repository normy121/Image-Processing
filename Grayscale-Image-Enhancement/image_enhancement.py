import cv2
import numpy as np

def gamma_correction(input_img, const, gamma):
    if (const > 0) and (gamma > 0.0):
        output_img = const * ((input_img / 255) ** gamma) * 255
        return np.array(output_img).astype(int)
    else:
        return input_img

def hist_eq(input_img):
    def calculate_histogram(image):
        hist, bins = np.histogram(image.ravel(), 256, [0, 255])
        return hist

    def apply_histogram_equalization(image):
        hist_original = calculate_histogram(image)

        hist, bins = np.histogram(image.ravel(), 256, [0, 255])
        pdf = hist / image.size
        cdf = pdf.cumsum()
        output_img = np.around(cdf * 255).astype(np.uint8)[image]

        hist_processed = calculate_histogram(output_img)

        return output_img, hist_original, hist_processed

    return apply_histogram_equalization(input_img)[0]

def laplacian(input_img, kernel):
    result = cv2.filter2D(input_img, -1, kernel)
    result[result > 255] = 255
    result[result < 0] = 0
    return result
