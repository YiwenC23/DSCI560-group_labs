import os
import re
import csv
import time
import cv2 as cv
import numpy as np
import requests
import pytesseract as pt
from tqdm import tqdm
from pdf2image import convert_from_path
from joblib import Parallel, delayed
from collections import Counter


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

def extract_text(image_path):
    src = cv.imread(image_path, cv.IMREAD_COLOR)
    custom_config = r"--oem 3 --psm 6"
    
    image, src_noTable, boxes = detect_table(src)
    
    def extract_text_from_box(box, image):
        x1, y1, x2, y2 = box
        roi = image[y1:y2, x1:x2]
        box_text = pt.image_to_string(roi, config=custom_config).strip()
        return box_text
    
    if boxes:
        box_texts = []
        for _, box in enumerate(boxes):
            box_text = extract_text_from_box(box, image)
            box_texts.append(box_text)
        
        text_box = "\n\n".join(box_texts) + "\n"
        text_noTable = pt.image_to_string(src_noTable, config=custom_config).strip()
        text_overall = text_box + "\n" + text_noTable
        
        return text_overall
    
    else:
        text = pt.image_to_string(src, config=custom_config, lang="eng").strip()
        
        return text


def extract_content(text):
    extracted_data = {}
    
    field_patterns = {
        "Operator": r"\bOperator\b:?.*?([A-Z]\w+\s.*[^\d+\W])",
        "Well Name": r"\bWell\b\s+\bName\b\s+\band\b\s+\bNumber\b:?.*?([A-Z]\w+.*?\d[A-Z])",
        "API": r"(\d{2}-\d{3}-\d{5})",
        "County": r"\bCounty\b:?\s*?\|?\s*(.+),?",
        "State": r"\bState\b:?\s*?\|?\s*([A-Z][A-Z]).*?",
        "Footages": r"Footages\b.*?(\d+[^\n]+)",
        "Section": r"\bSection\b:?.*?(\d+).*?",
        "Township": r"\bTownship\b.*?(\d+[^\n]+)",
        "Range": r"\bRange\b.*?(\d+[^\n]+)",
        "Latitude": r"(\d+°\s*\d+'\s*\d+\.\d+\s[NS])",
        "Longitude": r"(\d+°\s*\d+'\s*\d+\.\d+\s[EW])",
    }
    
    
    for key, pattern in field_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 1:
                value = groups[0].strip()
                if key == "County":
                    url = "https://raw.githubusercontent.com/raphaelreyna/State-County-City/master/state-county-city_data.json"
                    response = requests.get(url)
                    county_data = response.json()
                    for state, counties in county_data.items():
                        if counties and isinstance(counties, dict):
                            if value in counties:
                                extracted_data["County"] = value
                                break
                        else:
                            pass
                else: 
                    pass
                extracted_data[key] = value
            else:
                pass
    
    return extracted_data


def process_image(image_path):
    try:
        img_text = extract_text(image_path)
        extracted_data = extract_content(img_text)
        for key, value in extracted_data.items():
            if isinstance(value, tuple):
                extracted_data[key] = " ".join(value)
        return extracted_data
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {}


def select_value(values):
    
    if not values:
        return None
    
    longest_run_value = values[0]
    longest_run_length = 1
    current_value = values[0]
    current_length = 1
    
    for val in values[1:]:
        if val == current_value:
            current_length += 1
        
        else:
            if current_length > longest_run_length:
                longest_run_length = current_length
                longest_run_value = current_value
            
            current_value = val
            current_length = 1
    
    if current_length > longest_run_length:
        longest_run_length = current_length
        longest_run_value = current_value
    
    if longest_run_length > 1:
        return longest_run_value
    
    counts = Counter(values)
    max_freq = max(counts.values())
    
    if max_freq > 1:
        candidates = [v for v, cnt in counts.items() if cnt == max_freq]
        return sorted(candidates)[0]
    else:
        return sorted(values)[0]


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    txt_dir = os.path.join(BASE_DIR, "../data/processed_data/pdf_images_txt/")
    output_txtdir = os.path.join(BASE_DIR, "../data/processed_data/pdf_images_txt/")

    txt_files = [txt.path for txt in os.scandir(txt_dir) if txt.is_file() and txt.name.endswith(".txt")]

    file_results = {}
    for txt_file in tqdm(txt_files, total=len(txt_files), position=0, desc="Processing txt files"):
        file_name = os.path.basename(txt_file).split("_")[0]
        
        with open(txt_file, "r") as f:
            image_paths = [line.strip() for line in f.readlines()]
        
        keyword_list = ["Operator", "Well Name", "API", "County", "State", "Footages", "Section", "Township", "Range", "Latitude", "Longitude"]
        
        def process_image(image_path):
            img_text = extract_text(image_path)
            extracted_data = extract_content(img_text)
            for key, value in extracted_data.items():
                if isinstance(value, tuple):
                    extracted_data[key] = " ".join(value)
            return extracted_data
        
        results = Parallel(n_jobs=-1, timeout=99999)(
            delayed(process_image)(path) for path in tqdm(image_paths, position=1, leave=False, desc=f"Processing the images of {file_name} file")
        )
        
        values = {key: [] for key in keyword_list}
        for extracted_data in results:
            for key, value in extracted_data.items():
                values[key].append(value)
        
        final_result = {}
        for key in keyword_list:
            final_result[key] = select_value(values[key])
        
    csv_path = os.path.join(BASE_DIR, "../well_info.csv")
    with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow([
            file_name,
            final_result["Operator"],
            final_result["Well Name"],
            final_result["API"],
            final_result["County"],
            final_result["State"],
            final_result["Footages"],
            final_result["Section"],
            final_result["Township"],
            final_result["Range"],
            final_result["Latitude"],
            final_result["Longitude"]
        ])