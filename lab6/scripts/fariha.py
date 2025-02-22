from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import pandas as pd
import time
df=pd.read_csv('data.csv')
print("Column names:", df.columns)
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--headless")  # Run in headless mode (optional)
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--remote-debugging-port=9222")
chrome_options.add_argument("--user-data-dir=./chrome-data")
driver = webdriver.Chrome()

for _, row in df.iterrows():
    api_number = row["API"]
#    well_name = row[" Well Name"]
    if pd.notna(api_number):  # Only proceed if API number exists
        driver.get("https://www.drillingedge.com/search")
#    driver.get("https://www.drillingedge.com/search")
        search_box = driver.find_element(By.NAME, "api_no")
        search_box.send_keys(api_number)
        search_box.send_keys(Keys.RETURN)
        time.sleep(3)
        results = driver.find_elements(By.CLASS_NAME, "result-class")
        for result in results:
            print(result.text.strip())
driver.quit()
