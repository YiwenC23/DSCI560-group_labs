import os
import re
import json
import time
import cv2 as cv
import numpy as np
import requests
import pytesseract as pt
from tqdm import tqdm
from pdf2image import convert_from_path
from joblib import Parallel, delayed
from collections import Counter

from database import SessionLocal, WellInfo

#* Function to convert PDF to images
def pdf_to_images(input_path, output_base, dpi=300, thread_count=24):
    files = [f.path for f in os.scandir(input_path) if f.is_file() and f.name.endswith(".pdf")]
    
    for _, file in enumerate(tqdm(files, total=len(files), position=0, desc="Processing PDFs'")):
        file_name, _= os.path.splitext(os.path.basename(file))
        output_dir = os.path.join(output_base, file_name)
        txt_dir = os.path.join(output_base, "../processed_data/pdf_images_txt/")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)
        
        images = convert_from_path(
            file,
            dpi=dpi,
            fmt="jpeg",
            thread_count=thread_count
        )
        
        for i, image in enumerate(tqdm(images, total=len(images), position=1, leave=False, desc=f"Converting {file_name} PDF file to images")):
            image.save(f"{output_dir}/page_{i+1}.jpg", "JPEG")
            
            with open(os.path.join(txt_dir, f"{file_name}_imagelist.txt"), "a") as f:
                f.write(f"{output_dir}/page_{i+1}.jpg\n")
            
            time.sleep(0.01)
    
    print(f"Converted all {len(input_path)} to images and saved in output folder.")


#* Function to preprocess the image
def img_preprocess(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    binary = cv.adaptiveThreshold(
        gray, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY, 11, 2
    )
    
    denoised = cv.medianBlur(binary, 5)
    
    kernel = np.ones((3, 3), np.uint8)
    processed = cv.morphologyEx(denoised, cv.MORPH_OPEN, kernel)
    
    return processed


#* Function to detect table in the image
def detect_table(src):
    if len(src.shape) != 2:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src
    
    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, -2)
    denoise = cv.medianBlur(bw, 5)
    
    horizontal = np.copy(denoise)
    vertical = np.copy(denoise)
    
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
    
    edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, -2)
    
    kernel = np.ones((2, 2), np.uint8)
    edges = cv.dilate(edges, kernel)
    
    smooth = np.copy(vertical)
    smooth = cv.blur(smooth, (4, 4))
    
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
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            src_noTable[y_min:y_max, x_min:x_max] = (255, 255, 255)
        return src, src_noTable, boxes
    else:
        return src, None, None


#* Function to extract text from the image
def extract_text(image_path):
    src = cv.imread(image_path, cv.IMREAD_COLOR)
    custom_config = r"--oem 3 --psm 6"
    
    image, src_noTable, boxes = detect_table(src)
    
    def extract_text_from_box(box, image):
        x1, y1, x2, y2 = box
        roi = image[y1:y2, x1:x2]
        box_text = pt.image_to_string(roi, config=custom_config, lang="eng").strip()
        return box_text
    
    if boxes:
        box_texts = []
        for _, box in enumerate(boxes):
            box_text = extract_text_from_box(box, image).replace("\n", " ")
            box_texts.append(box_text)
        
        text_box ="\n\n".join(box_texts)
        text_noTable = pt.image_to_string(src_noTable, config=custom_config, lang="eng")
        text_overall = text_box + "\n" + text_noTable
        
        return text_overall
    
    else:
        prepro_img = img_preprocess(src)
        text = pt.image_to_string(prepro_img, config=custom_config, lang="eng")
        
        return text


def extract_content(text):
    extracted_data = {}
    
    field_patterns = {
        "operator": r"\bOperator\b:?.*?([A-Z]\w+\s.*[^\d+\W])",
        "well_name": r"\bWell\b\s+\bName\b\s+\band\b\s+\bNumber\b:?.*?([A-Z]\w+.*?\d[A-Z])",
        "API": r"(\d{2}-\d{3}-\d{5})",
        "county": r"\bCounty\b:?\s*?\|?\s*(?!,)(.+)",
        "state": r"\bState\b:?\s*?\|?\s*([A-Z][A-Z]).*?",
        "footages": r"Footages\b.*?(\d+[^\n]+)",
        "section": r"\bSection\b:?.*?(\d+).*?",
        "township": r"\bTownship\b.*?(\d+[^\n]+)",
        "range": r"\bRange\b.*?(\d+[^\n]+)",
        "latitude": r"(\d+°\s*\d+'\s*\d+\.\d+\s[NS])",
        "longitude": r"(\d+°\s*\d+'\s*\d+\.\d+\s[EW])",
        "date_stimulated": r"Date\s*?Stimulated\b.*?(\d{2}/\d{2}/\d{4})",
        "stimulated_formation": r"Stimulated?\s*?\BFormation\b:?\s*?\|?\s*?([A-Z][a-z]+)",
        "top": r"Top.*?:?\s*?\|?\s*?(\d+[^\n]+)",
        "bottom": r"Bottom.*?:?\s*?\|?\s*?(\d+[^\n]+)",
        "stimulation_stages": r"Stimulation?\s\bStages\b:?\|?\s*?(\d\d?)",
        "volume": r"\bVolume\b.*?:?\s*?\|?\s*?(\d+[^\n]+)",
        "volume_unites": r"\bVolume\b.*?:?\s*?\|?\s*?([A-Z][a-z]+)",
        "type_treatment": r"Type?\s*?\bTreatment\b\s*?:?\s*?\|?\s*?([A-Z][a-z]+\s[A-Z][a-z]+?)",
        "acid": r"Acid\s*?\%?\s*?:?\s*?\|?\s*?(\d+[^\n]+)",
        "lbs_proppant": r"Lbs\s*?Proppant\s*?:?\s*?\|?\s*?(\d+[^\n]+)",
        "maximum_treatment_pressure": r"Maximum\s*?Treatment\s*?Pressure\s*?:?\s*?\|?\s*?(\d+[^\n]+)",
        "maximum_treatment_rate": r"Maximum\s*?Treatment\s*?Rate\s*?:?\s*?\|?\s*?(\d+[^\n]+)",
        "details": r"Details\s*?:?\s*?\|?\s*?(.+)",
    }
    
    for key, pattern in field_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 1:
                value = groups[0].strip()
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


def longest_common_prefix(str_list):
    if not str_list:
        return ""
    prefix = str_list[0]
    
    for s in str_list[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def trim_value_by_keywords(value, keywords):
    min_index = len(value)
    
    for kw in keywords:
        idx = value.find(kw)
        if idx != -1 and idx < min_index:
            min_index = idx
    
    return value[:min_index].strip() if min_index < len(value) else value


def select_value(values, keywords=None):
    if not values:
        return None
    
    if keywords:
        processed_values = [trim_value_by_keywords(val, keywords) for val in values]
    else:
        processed_values = values
    
    common_part = longest_common_prefix(processed_values)
    if common_part:
        return common_part
    
    longest_run_value = processed_values[0]
    longest_run_length = 1
    current_value = processed_values[0]
    current_length = 1
    
    for val in processed_values[1:]:
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
    
    counts = Counter(processed_values)
    max_freq = max(counts.values())
    
    if max_freq > 1:
        candidates = [v for v, cnt in counts.items() if cnt == max_freq]
        return sorted(candidates)[0]
    else:
        return sorted(processed_values)[0]


def data_storage(data):
    db = SessionLocal()
    try:
        db.add(WellInfo(
            well_name=data["well_name"],
            operator=data["operator"],
            API=data["API"],
            county=data["county"],
            state=data["state"],
            footages=data["footages"],
            section=data["section"],
            township=data["township"],
            range=data["range"],
            latitude=data["latitude"],
            longitude=data["longitude"],
            date_stimulated=data["date_stimulated"],
            stimulated_formation=data["stimulated_formation"],
            top=data["top"],
            bottom=data["bottom"],
            stimulation_stages=data["stimulation_stages"],
            volume=data["volume"],
            volume_unites=data["volume_unites"],
            type_treatment=data["type_treatment"],
            acid=data["acid"],
            lbs_proppant=data["lbs_proppant"],
            maximum_treatment_pressure=data["maximum_treatment_pressure"],
            maximum_treatment_rate=data["maximum_treatment_rate"],
            details=data["details"]
        ))
        db.commit()
    except Exception as e:
        print(f"Error storing data: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_base = os.path.join(BASE_DIR, "../data/raw_data/")
    input_path = os.path.join(BASE_DIR, "../data/raw_data/DSCI560_Lab6/")
    pdf_to_images(input_path, output_base, dpi=300, thread_count=24)
    
    txt_dir = os.path.join(BASE_DIR, "../data/processed_data/pdf_images_txt/")
    
    txt_files = [txt.path for txt in os.scandir(txt_dir) if txt.is_file() and txt.name.endswith(".txt")]
    
    url = "https://raw.githubusercontent.com/raphaelreyna/State-County-City/master/state-county-city_data.json"
    response = requests.get(url)
    county_state_data = response.json()
    with open("county_state_data.json", "w") as f:
        json.dump(county_state_data, f)
    
    file_results = {}
    for txt_file in tqdm(txt_files, total=len(txt_files), position=0, desc="Processing txt files"):
        file_name = os.path.basename(txt_file).split("_")[0]
        
        with open(txt_file, "r") as f:
            image_paths = [line.strip() for line in f.readlines()]
        
        keyword_list = [
            "operator", "well_name", "API", "county", "state", "footages", "section", "township", "range", "latitude", "longitude",
            "date_stimulated", "stimulated_formation", "top", "bottom", "stimulation_stages", "volume", "volume_unites", "type_treatment",
            "acid", "lbs_proppant", "maximum_treatment_pressure", "maximum_treatment_rate", "details"
        ]
        
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
        
        if final_result["county"] in ["Williams", "McKenzie"]:
            final_result["state"] = "North Dakota"
        
        data_storage(final_result)