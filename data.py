from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import pytesseract
import cv2



def load_images_from_folder(folder):
    images = []
    image_names = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            image_names.append(filename)
    return images, image_names


def augment_images(image_names):
    for j,img in enumerate(image_names):
        image_path = f"/kaggle/input/dataset/images/{img}"  # Change this to your image path
        image = cv2.imread(image_path)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # Adjust based on text density
        dilated = cv2.dilate(clean, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b[1])
        
        merged_boxes = []
        i = 0
        
        while i < len(bounding_boxes) - 1:
            x1, y1, w1, h1 = bounding_boxes[i]
            x2, y2, w2, h2 = bounding_boxes[i + 1]
        
            # Check if two bounding boxes are close enough to be considered part of the same section
            if abs(y2 - (y1 + h1)) < 50:  # Merge if next box is within 50 pixels
                x_new = min(x1, x2)
                y_new = min(y1, y2)
                w_new = max(x1 + w1, x2 + w2) - x_new
                h_new = max(y1 + h1, y2 + h2) - y_new
        
                merged_boxes.append((x_new, y_new, w_new, h_new))
                i += 2
            else:
                i += 1
        
        output_dir = "/kaggle/working/cropped_paragraphs"
        os.makedirs(output_dir, exist_ok=True)
        
        min_area = 10000  # Minimum bounding box area (to ignore single words)
        min_height = 60   # Minimum height of a paragraph block
        min_aspect_ratio = 0.5  # To avoid very long but short-height text (headers, footers)
        
        for i, (x, y, w, h) in enumerate(merged_boxes):
            aspect_ratio = w / h 
        
            if h > min_height and w * h > min_area and aspect_ratio > min_aspect_ratio:
                cropped_paragraph = image[y:y+h, x:x+w]  # Crop paragraph
                save_path = os.path.join(output_dir, f"multi_paragraph_{j}_{i+1}.png")
                cv2.imwrite(save_path, cropped_paragraph)
        
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        half_count = len(contours) // 3
        j=1
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            
            
            aspect_ratio = w / h 
        
            if h > min_height and w * h > min_area and aspect_ratio > min_aspect_ratio:
                cropped_paragraph = image[y:y+h, x:x+w]  # Crop paragraph
                save_path = os.path.join(output_dir, f"paragraph_{j}_{i+1}.png")
                cv2.imwrite(save_path, cropped_paragraph)
                j=j+1
            if j==half_count:
                break
                # Draw bounding box on original image (for visualization)
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)