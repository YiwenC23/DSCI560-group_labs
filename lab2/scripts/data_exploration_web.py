import requests
from bs4 import BeautifulSoup
import csv

url = "https://news.ycombinator.com/item?id=24460141"
response = requests.get(url)

if response.status_code == 200:
	web_data = BeautifulSoup(response.text, "html.parser")
	content_paragraphs = []
	content_divs = web_data.find_all(class_='commtext c00')
	
for div in content_divs:
    content_paragraphs.append([div.get_text()])
    paragraphs = div.find_all('p')
    for p in paragraphs:
        content_paragraphs.append([p.get_text()])
    informations = div.find_all('i')
    for i in informations:
        content_paragraphs.append([i.get_text()])

with open('../data/comment.csv', 'w', newline='', encoding='utf-8') as file:
	comment_writer = csv.writer(file)
	comment_writer.writerow(['Comments'])
	comment_writer.writerows(content_paragraphs)

print('Here are the first 5 entries of comments:')
for line in content_paragraphs[:5]:
    print(line[0])
    
print(f'We have {len(content_paragraphs)} rows/entries of comments.')

missing_data = [i for i, row in enumerate(content_paragraphs) if not row or None in row or "" in row]
if missing_data:
    print(f'Here are the rows with missing data: {missing_data}.')
else:
    print(f'We do not have missing data for this dataset.')
