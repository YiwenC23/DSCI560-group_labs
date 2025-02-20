import os
import re
import cv2
import time
import numpy as np
from tqdm import tqdm
from pdf2image import convert_from_path


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

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = "/Users/yiwen/Downloads/DSCI560_Lab5/"
    output_base = os.path.join(BASE_DIR, "../data/raw_data/")
    output_txtdir = os.path.join(BASE_DIR, "../data/processed_data/pdf_images_txt/")
    files = [f.path for f in os.scandir(pdf_dir) if f.is_file()]
    
    
    pdf_to_images(files, output_base)