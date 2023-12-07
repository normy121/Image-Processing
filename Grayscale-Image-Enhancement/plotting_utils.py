import cv2
from matplotlib import pyplot as plt

def plot_images(images, titles, rows, cols, figsize=(16, 9), window_title=''):
    fig = plt.figure(figsize=figsize)
    fig.canvas.manager.set_window_title(window_title)

    for i in range(len(images)):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(images[i])
        ax.set_title(titles[i])

    plt.tight_layout()
    plt.show()

def plot_histograms(images, titles, rows, cols, figsize, ylims, window_title):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.canvas.manager.set_window_title(window_title)

    for i, ax in enumerate(axes.flatten()):
        idx = i // 2
        if i % 2 == 0 and idx < len(images):
            # Plot image
            ax.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
            ax.set_title(titles[idx])
        elif i % 2 != 0 and idx < len(images):
            # Plot histogram
            ax.hist(images[idx].ravel(), bins=256, range=[0, 256], color='#6495ED')
            image_type = "Original" if (i+1) % 4 == 2 else "Processed"
            ax.set_title(f'Histogram of {image_type} {titles[idx].split(" ")[-1]}')
            ax.set_ylim([0, 20000])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()