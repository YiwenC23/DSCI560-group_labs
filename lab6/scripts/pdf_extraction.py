import os
import re
import time
import cv2 as cv
import numpy as np
import pytesseract as pt
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


#* Function to preprocess the image
def img_preprocess(image_path):
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    binary = cv.adaptiveThreshold(
        gray, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY, 11, 2
    )
    
    denoised = cv.medianBlur(binary, 3)
    
    kernel = np.ones((3, 3), np.uint8)
    processed = cv.morphologyEx(denoised, cv.MORPH_CLOSE, kernel)
    
    def deskew(processed_img):
        coords = np.column_stack(np.where(processed_img == 0))
        angle = cv.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        (h, w) = processed_img.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        rotation = cv.warpAffine(processed_img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
        
        return rotation
    
    preprocessed_image = deskew(processed)
    
    return preprocessed_image


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
        
        #? Filter based on area and aspect ratio
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


#* Function to extract text from the image
def extract_text(image_path):
    custom_config = r"--oem 3 --psm 6"
    
    image, src_noTable, boxes = detect_table(image_path)
    
    def extract_text_from_box(box, image):
        x1, y1, x2, y2 = box
        roi = image[y1:y2, x1:x2]
        box_text = pt.image_to_string(roi, config=custom_config)
        return box_text
    
    if boxes:
        box_texts = []
        for _, box in enumerate(boxes):
            box_text = extract_text_from_box(box, image)
            box_texts.append(box_text)
        
        text_box = "\n".join(box_texts) + "\n"
        text_noTable = pt.image_to_string(src_noTable, config=custom_config)
        text_overall = text_box + "\n" + text_noTable
        
        return text_overall
    
    else:
        prepro_img = img_preprocess(image_path)
        text = pt.image_to_string(prepro_img, config=custom_config)
        
        return text


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = "/Users/yiwen/Downloads/DSCI560_Lab5/"
    output_base = os.path.join(BASE_DIR, "../data/raw_data/")
    output_txtdir = os.path.join(BASE_DIR, "../data/processed_data/pdf_images_txt/")
    files = [f.path for f in os.scandir(pdf_dir) if f.is_file()]
    
    
    pdf_to_images(files, output_base)