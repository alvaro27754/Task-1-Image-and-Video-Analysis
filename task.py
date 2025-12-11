import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

def getCandidateMask(img):
    h, w, _ = img.shape
    mask = np.ones((h, w), dtype=bool)
    mask[:int(0.4 * h), :] = False
    mask[:, int(0.85 * w):] = False
    
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_RGB = cv2.inRange(img_RGB, np.array([50, 50, 50]), np.array([180, 120, 120])).astype(bool)
    mask = mask & mask_RGB
    
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    mask_YCrCb = cv2.inRange(img_YCrCb, np.array([0, 120, 85]), np.array([255, 200, 135])).astype(bool)
    mask = mask & mask_YCrCb
    
    # img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # mask_HSV = ~cv2.inRange(img_HSV, np.array([100, 0, 0]), np.array([140, 255, 255])).astype(bool)
    # mask = mask & mask_HSV
    
    mask = (mask.astype(np.uint8)) * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    mask = mask.astype(bool)

    return mask

def separateRegions(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        edges = cv2.Canny(gray, threshold1=50, threshold2=150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        edges = 255 - edges

        img = cv2.bitwise_and(img, img, mask=edges)

        return img

def label_connected_components(binary_img):
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

def processComponents(img):
        binary_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(bool)

        label_img = label_connected_components(binary_img)

        ft = {}
        repr = {}
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

def selectRepresentatives(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary_img = (img > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)

    centroid_img = np.zeros_like(binary_img, dtype=np.uint8)

    for i in range(1, num_labels):
        cx, cy = centroids[i]
        cx, cy = int(round(cx)), int(round(cy))
        cv2.circle(centroid_img, (cx, cy), 5, 255, -1)

    return centroid_img

def load_points(name, img):
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

    return centroid_img

def main():
        images = []
        names = []
        for file in os.listdir("images"):
                img = cv2.imread(os.path.join("images", file))

                images.append(img)
                names.append(file)

        for img, name in zip(images, names):
                orig = img

                base, ext = os.path.splitext(name)

                mask = getCandidateMask(img)
                mask = (mask.astype(np.uint8)) * 255

                img = cv2.bitwise_and(img, img, mask=mask)
                cv2.imwrite(os.path.join("results", base + "_1" + ext), img)

                img = separateRegions(img)
                cv2.imwrite(os.path.join("results", base + "_2" + ext), img)

                img = processComponents(img)
                cv2.imwrite(os.path.join("results", base + "_3" + ext), img)

                representatives = selectRepresentatives(img)
                orig[:,:,1] = np.maximum(orig[:,:,1], representatives)

                points = load_points(name, img)
                orig[:,:,0] = np.maximum(orig[:,:,0], points)

                cv2.imwrite(os.path.join("results", base + "_4" + ext), orig)

if __name__ == '__main__':
        main()

