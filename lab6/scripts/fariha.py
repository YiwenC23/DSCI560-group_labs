from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time
import logging

# Set up logging
logging.basicConfig(
    filename='well_scraping.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--remote-debugging-port=9222")
    chrome_options.add_argument("--user-data-dir=./chrome-data")
    return webdriver.Chrome(options=chrome_options)

def get_well_details(driver, api_number):
    """Extract well details from the well page"""
    details = {
        'Well Status': None,
        'Well Type': None,
        'Closest City': None,
        'Barrels Produced': None
    }
    
    try:
        # Wait for the details table to load
        wait = WebDriverWait(driver, 10)
        table = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "interest_table")))
        
        # Dictionary mapping field names to their XPath expressions
        field_xpaths = {
            'Well Status': "//td[contains(text(),'Well Status')]/following-sibling::td",
            'Well Type': "//td[contains(text(),'Well Type')]/following-sibling::td",
            'Closest City': "//td[contains(text(),'Closest City')]/following-sibling::td",
            'Barrels Produced': "//td[contains(text(),'Barrels Produced')]/following-sibling::td"
        }
        
        # Extract each field
        for field, xpath in field_xpaths.items():
            try:
                element = driver.find_element(By.XPATH, xpath)
                details[field] = element.text.strip()
            except NoSuchElementException:
                logging.warning(f"Could not find {field} for API {api_number}")
                
    except Exception as e:
        logging.error(f"Error extracting details for API {api_number}: {str(e)}")
        
    return details

def main():
    try:
        # Read the CSV file
        df = pd.read_csv('API.csv')
        logging.info(f"Loaded CSV with {len(df)} rows")
        
        # Initialize new columns if they don't exist
        for col in ['Well Status', 'Well Type', 'Closest City', 'Barrels Produced']:
            if col not in df.columns:
                df[col] = None
        
        driver = setup_driver()
        
        for index, row in df.iterrows():
            api_number = str(row["API"]).strip()
            
            if pd.isna(api_number) or api_number == "":
                logging.warning(f"Skipping row {index}: Empty or invalid API number")
                continue
                
            logging.info(f"Processing API number: {api_number}")
            
            try:
                # Navigate to search page
                driver.get("https://www.drillingedge.com/search")
                search_box = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "api_no"))
                )
                
                # Perform search
                search_box.clear()
                search_box.send_keys(api_number)
                search_box.send_keys(Keys.RETURN)
                time.sleep(3)
                
                # Try to find and click the well link
                try:
                    well_link = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((
                            By.XPATH, 
                            f"//td[contains(text(),'{api_number}')]/following-sibling::td/a"
                        ))
                    )
                    well_link.click()
                    time.sleep(3)
                    
                    # Get well details
                    details = get_well_details(driver, api_number)
                    
                    # Update DataFrame
                    for field, value in details.items():
                        df.loc[index, field] = value
                        
                except TimeoutException:
                    logging.warning(f"No well link found for API {api_number}")
                    
            except Exception as e:
                logging.error(f"Error processing row {index} with API {api_number}: {str(e)}")
                
            # Save progress after each row
            if index % 10 == 0:
                df.to_csv('Updated_API.csv', index=False)
                logging.info(f"Progress saved after processing {index} rows")
                
            time.sleep(1)
            
        # Final save
        df.to_csv('Updated_API.csv', index=False)
        logging.info("Processing completed. Data saved to Updated_API.csv")
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
