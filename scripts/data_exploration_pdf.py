import os
import re
import sys
import requests
import pdfplumber
import pytesseract


def save_pdf():
	try:
		r = requests.get(pdfURL, stream = True)
		
		with open(interview_pdf_path, "wb") as fd:
			for chunk in r.iter_content(chunk_size = 1024):
				fd.write(chunk)
		
		print("Successfully saved the PDF file!\n")
	
	except Exception as e:
		print(e)
		sys.exit()


def extract_text():
	try:
		lines = ""
		with pdfplumber.open(interview_pdf_path) as pdf:
			for page in pdf.pages:
				words = page.extract_words(extra_attrs=["size"])
				
				for i, word in enumerate(words):
					# Remove footer by skipping words with font size < 10
					if word["size"] < 10:
						continue
					
					section = word["size"] > 15
					question = 13 <= word["size"] <= 15
					answer = word["size"] == 12
					
					# Check next word's size
					next_word_answer = (i == len(words) - 1) or (words[i + 1]["size"] <= 12)
					next_word_question = (i < len(words) - 1) and (13 <= words[i + 1]["size"] <= 15)
					next_word_not_section = (i == len(words) - 1) or (words[i + 1]["size"] <= 15)
					
					# Add word and handle sentence endings for answers
					if answer and re.search(r'(?<!\d)[.!?:]$', word["text"]):
						lines += word["text"] + "\n"
					else:
						lines += word["text"] + " "
					
					# Add newline if current word belongs to section AND next word doesn't
					if section and next_word_not_section:
						lines += "\n"
					# Add newline if current word belongs to question AND next word belongs to answer
					elif question and next_word_answer:
						lines += "\n"
					# Add newline if current word belongs to answer AND next word belongs to question
					elif answer and next_word_question:
						lines += "\n"
		return lines
	
	except Exception as e:
		print(e)
		sys.exit()


if __name__ == "__main__":
	pdfURL = "https://storage.googleapis.com/kaggle-forum-message-attachments/984081/16703/DS%20interview%20quESTIONS.pdf"
	
	CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
	BASE_DIR = os.path.dirname(CURRENT_DIR)
	interview_pdf_path = os.path.join(BASE_DIR, "data/ds_interview_questions.pdf")
	interview_txt_path = os.path.join(BASE_DIR, "data/ds_interview_questions.txt")
	
	print("Saving the PDF file...")
	save_pdf()
	
	print("Extracting the text from the PDF file...\n")
	lines = extract_text()
	
	print("Saving the data into a text file...")
	with open(interview_txt_path, "w") as f:
		f.write(lines)
	
	print("Successfully saved the text file!")