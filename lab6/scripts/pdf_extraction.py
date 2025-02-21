import os
import re
import time
import cv2 as cv
import numpy as np
from tqdm import tqdm
from pdf2image import convert_from_path


#* Function to convert PDF to images
def pdf_to_images(input_files, output_base, dpi=300, thread_count=24):
    for _, file in enumerate(tqdm(input_files, total=len(input_files), position=0, desc="Processing PDFs'")):
        file_name, _= os.path.splitext(os.path.basename(file))
        output_dir = os.path.join(output_base, file_name)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if not os.path.exists(output_txtdir):
            os.makedirs(output_txtdir)
        
        images = convert_from_path(
            file,
            dpi=dpi,
            fmt="jpeg",
            thread_count=thread_count
        )
        
        for i, image in enumerate(tqdm(images, total=len(images), position=1, leave=False, desc=f"Converting {file_name} PDF file to images")):
            image.save(f"{output_dir}/page_{i+1}.jpg", "JPEG")
            
            with open(os.path.join(output_txtdir, f"{file_name}_imagelist.txt"), "a") as f:
                f.write(f"{output_dir}/page_{i+1}.jpg\n")
            
            time.sleep(0.01)
    
    print(f"Converted all {len(input_files)} to images and saved in output folder.")


#* Function to detect table in the image
def detect_table(image_path):
    src = cv.imread(image_path, cv.IMREAD_COLOR)
    if len(src.shape) != 2:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src
    
    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    
    horizontal = np.copy(bw)
    vertical = np.copy(bw)
    
    #? Detect horizontal lines
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    
    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
    
    #? Detect vertical lines
    rows = vertical.shape[0]
    vertical_size = rows // 40
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
    
    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)
    
    edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, -2)
    
    kernel = np.ones((2, 2), np.uint8)
    edges = cv.dilate(edges, kernel)
    
    smooth = np.copy(vertical)
    smooth = cv.blur(smooth, (2, 2))
    
    (rows, cols) = np.where(edges == 0)
    vertical[rows, cols] = smooth[rows, cols]
    
    #? Combine horizontal and vertical lines
    merge = cv.add(horizontal, vertical)
    
    merge_inverted = cv.bitwise_not(merge)
    
    contours, _ = cv.findContours(merge_inverted, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    min_area = 100  # Minimum area to filter out small noise (adjust as needed)
    max_area = (merge.shape[0] * merge.shape[1]) * 0.5  # Max area to avoid outer frame
    
    for contour in contours:
        #? Get the bounding rectangle for each contour
        x, y, w, h = cv.boundingRect(contour)
        area = w * h
        
        # Filter based on area and aspect ratio
        if min_area < area < max_area:
            peri = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:  # Approximate quadrilateral
                boxes.append((x, y, x + w, y + h))
    
    src_noTable = src.copy()
    if boxes:
        #? Fill the table with white color for OCR words that are not in the table
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            src_noTable[y_min:y_max, x_min:x_max] = (255, 255, 255)
        return src, src_noTable, boxes
    else:
        return src, None, None


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = "/Users/yiwen/Downloads/DSCI560_Lab5/"
    output_base = os.path.join(BASE_DIR, "../data/raw_data/")
    output_txtdir = os.path.join(BASE_DIR, "../data/processed_data/pdf_images_txt/")
    files = [f.path for f in os.scandir(pdf_dir) if f.is_file()]
    
    
    pdf_to_images(files, output_base)