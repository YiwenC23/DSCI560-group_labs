from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

driver = webdriver.Chrome()
for _, row in df.iterrows():
    api_number = row["API#"]
    well_name = row["Well Name"]
    driver.get("https://www.drillingedge.com/search")
    search_box = driver.find_element(By.NAME, "q")
    search_box.send_keys(f"{api_number} {well_name}")
    search_box.send_keys(Keys.RETURN)
    time.sleep(3)
    results = driver.find_elements(By.CLASS_NAME, "result-class")
    for result in results:
        print(result.text.strip())
driver.quit()
