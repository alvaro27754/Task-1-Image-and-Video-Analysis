# AUTHORS:
# Álvaro Velasco Sobrino
# Denis Molnar Ardelean

import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

def get_candidate_mask(img):
    """
    Creates a mask highlighting likely candidate regions based on spatial and
    color constraints using RGB and YCrCb color spaces.
    """
    h, w, _ = img.shape

    # Start with a full True mask and exclude upper 40% and rightmost 15%
    mask = np.ones((h, w), dtype=bool)
    mask[:int(0.4 * h), :] = False
    mask[:, int(0.85 * w):] = False
    
    # RGB filtering for candidate colors
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_RGB = cv2.inRange(img_RGB, np.array([50, 50, 50]), np.array([180, 120, 120])).astype(bool)
    mask = mask & mask_RGB
    
    # YCrCb filtering to select only skin-like tones
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    mask_YCrCb = cv2.inRange(img_YCrCb, np.array([0, 120, 85]), np.array([255, 200, 135])).astype(bool)
    mask = mask & mask_YCrCb
    
    # HSV filtering to remove colors that are too vibrant to be human skin
    # img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # mask_HSV = ~cv2.inRange(img_HSV, np.array([100, 0, 0]), np.array([140, 255, 255])).astype(bool)
    # mask = mask & mask_HSV
    
    mask = (mask.astype(np.uint8)) * 255
    
    # Clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Fill in the space between nearby elements
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    mask = mask.astype(bool)

    return mask

def separate_regions(img):
    """
    Enhances region separation by detecting edges, closing gaps, and masking
    the original image to isolate major structures.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Detect edges
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Close broken edges to form continuous boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Invert edges
    edges = 255 - edges

    # Apply edge mask to separate regions
    img = cv2.bitwise_and(img, img, mask=edges)

    return img

def label_connected_components(binary_img):
    """
    Custom implementation of connected component labeling (two-pass with
    collision resolution).
    Returns a label image where each connected region has a unique ID.
    
    This function was taken from the subject's GitHub repository
    """
    label_img = np.zeros_like(binary_img, dtype=np.uint16)
    collisions = {}

    for i in range(binary_img.shape[0]):
        for j in range(binary_img.shape[1]):
            if binary_img[i, j]:
                previous_labels = []
                
                if i >= 1 and label_img[i-1, j]:
                    previous_labels.append(label_img[i-1, j])
                if j >= 1 and label_img[i, j-1]:
                    previous_labels.append(label_img[i, j-1])

                if len(previous_labels) == 0:
                    label_img[i, j] = np.max(label_img) + 1
                elif len(previous_labels) == 1:
                    label_img[i, j] = min(previous_labels)
                else:
                    representative_label = min(previous_labels)
                    for label in previous_labels:
                        if label in collisions:
                            representative_label = min(representative_label, collisions[label])
                            
                    label_img[i, j] = representative_label
                    for label in previous_labels:
                        collisions[label] = representative_label

    for label in range(np.max(label_img)):
        if label in collisions:
            representative = collisions[label]
            
            if representative in collisions:
                collisions[label] = collisions[representative]

    for label, min_label_in_same_cc in collisions.items():
        label_img[label_img == label] = min_label_in_same_cc

    return label_img

def process_components(img):
    """
    Filters connected components based on their size and position.
    Removes noise and keeps components in expected zones of the image.
    """
    binary_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(bool)

    label_img = label_connected_components(binary_img)

    ft = {}     # component sizes (frequency table)
    repr = {}   # representative (first pixel) of each component

    # Count pixels per component
    for i in range(label_img.shape[0]):
        for j in range(label_img.shape[1]):
            val = label_img[i, j]

            if val not in ft:
                ft[val] = 0
                repr[val] = list((i, j))
            
            ft[val] += 1

    mask = np.ones(binary_img.shape).astype(bool)

    h = img.shape[0]
    w = img.shape[1]

    # Filter components
    for i in range(label_img.shape[0]):
        for j in range(label_img.shape[1]):
            val = label_img[i, j]

            if repr[val][0] >= int(0.5 * h) or repr[val][1] >= int(0.8 * w):
                if 60 <= ft[val] <= 200:
                    continue
            else:
                if 10 <= ft[val] <= 100:
                    continue

            mask[i, j] = False
    mask = (mask.astype(np.uint8)) * 255

    img = cv2.bitwise_and(img, img, mask=mask)

    return img

def select_representatives(img):
    """
    Finds connected components and marks their centroids as representative points.
    Returns an image with white dots at component centers and the number detected.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary_img = (img > 0).astype(np.uint8)

    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

    centroid_img = np.zeros_like(binary_img, dtype=np.uint8)

    # Draw a small circle on each centroid
    for i in range(1, num_labels):
        cx, cy = centroids[i]
        cx, cy = int(round(cx)), int(round(cy))
        cv2.circle(centroid_img, (cx, cy), 5, 255, -1)

    cnt = num_labels - 1

    return centroid_img, cnt

def load_points(name, img):
    """
    Loads ground-truth (CSV-labeled) points for a specific image.
    Returns an image with circles at the labeled coordinates.
    """
    points = []

    with open('labels_task_annotations_2025-12-09-08-55-14.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            image_name = row[3]
            if image_name == name:
                x, y = int(row[1]), int(row[2])
                points.append((x, y))

    centroid_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for cx, cy in points:
        cx, cy = int(round(cx)), int(round(cy))
        cv2.circle(centroid_img, (cx, cy), 5, 255, -1)

    cnt = len(points)

    return centroid_img, cnt

def accuracy(name, representatives):
    """
    Computes detection accuracy:
    A ground-truth point is correct if it falls inside a dilated representative region.
    """
    points = []

    with open('labels_task_annotations_2025-12-09-08-55-14.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            image_name = row[3]
            if image_name == name:
                x, y = int(row[1]), int(row[2])
                points.append((x, y))

    # Dilate representative dots so they count if close to center
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    representatives = cv2.dilate(representatives, kernel, iterations=3)

    # plt.imshow(representatives, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # Count matches
    cnt = 0
    for cx, cy in points:
        if representatives[cy, cx] == 0:
            continue

        cnt += 1
    
    return cnt / len(points)
    

def main():
    """
    Main pipeline: load images, process them through each stage,
    compare detections with ground truth, save results, compute MSE.
    """
    images = []
    names = []

    # Load all images from directory
    for file in os.listdir("images"):
        img = cv2.imread(os.path.join("images", file))

        images.append(img)
        names.append(file)

    mse = 0
    cnt = 0

    for img, name in zip(images, names):
        orig = img

        base, ext = os.path.splitext(name)

        # candidate mask
        mask = get_candidate_mask(img)
        mask = (mask.astype(np.uint8)) * 255

        img = cv2.bitwise_and(img, img, mask=mask)
        cv2.imwrite(os.path.join("results", base + "_1" + ext), img)

        # region separation
        img = separate_regions(img)
        cv2.imwrite(os.path.join("results", base + "_2" + ext), img)

        # component filtering
        img = process_components(img)
        cv2.imwrite(os.path.join("results", base + "_3" + ext), img)

        # compute detected representatives
        representatives, cnt1 = select_representatives(img)
        orig[:,:,1] = np.maximum(orig[:,:,1], representatives)

        # load ground truth
        points, cnt2 = load_points(name, img)
        orig[:,:,0] = np.maximum(orig[:,:,0], points)

        print(f'Ground truth: {cnt2}, Detected: {cnt1}, Percentage: {100 * (cnt1 / cnt2)}')

        mse += (cnt1 - cnt2) ** 2
        cnt += 1

        print(f'Accuracy={accuracy(name, representatives)}')

        cv2.imwrite(os.path.join("results", base + "_4" + ext), orig)
    
    mse /= cnt

    print(f'MSE={mse}')

if __name__ == '__main__':
    main()

