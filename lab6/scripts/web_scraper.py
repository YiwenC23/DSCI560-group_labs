import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import update
import time
import re
import string
from test_database import WellStatus, SessionLocal
from lxml import etree

def separate_and_lowercase(text):
    text = re.sub(r'_', ' ', text)  # Replace underscores with spaces
    separated_text = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)  # Split CamelCase words by inserting spaces before uppercase letters
    return separated_text.lower()

# Convert the abbreviations to the full words/phrases
def tokenize_phrase(text, ABBR_TREE):
    text = ''.join([c if c not in string.punctuation else ' ' for c in text])
    text = separate_and_lowercase(text)
    tokenize_text = text.split()
    for i in range(len(tokenize_text)):
        results = ABBR_TREE.xpath(f'//div[li[text()="{tokenize_text[i]}"]]/text()')
        if results:
            tokenize_text[i] = results[0].split()[0]
    return " ".join(tokenize_text)

# Convert text to lowercase and replace spaces with hyphens. 
def format_url_segment(segment):
    return re.sub(r"\s+", "-", segment.strip().lower())

# Construct the well details URL dynamically.
def construct_well_url(state, county, well_name, api_number):
    base_url = "https://www.drillingedge.com"
    state_formatted = format_url_segment(state)
    county_formatted = format_url_segment(county)
    well_name_formatted = format_url_segment(well_name)
    
    return f"{base_url}/{state_formatted}/{county_formatted}/wells/{well_name_formatted}/{api_number}"

# Scrape well details from the dynamically generated URL.
def scrape_well_data(state, county, well_name, api_number):
    # ABBR_TREE = etree.parse("Abbreviations.xml")

    # state = tokenize_phrase(state, ABBR_TREE)
    # county = tokenize_phrase(county, ABBR_TREE)

    well_url = construct_well_url(state, county, well_name, api_number)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    response = requests.get(well_url, headers=headers)
    
    if response.status_code != 200:
        print(f"[ERROR] Failed to access {well_url} (API# {api_number}, Well: {well_name})")
        return None
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    try:
        well_status = soup.select_one("th:contains('Well Status') + td").text.strip()
        well_type = soup.select_one("th:contains('Well Type') + td").text.strip()
        closest_city = soup.select_one("th:contains('Closest City') + td").text.strip()
        barrels_produced_element = soup.select_one("p.block_stat span.dropcap")
        barrels_produced = barrels_produced_element.text.strip() if barrels_produced_element else "0.00"
        mcf_gas_produced_element = soup.select("p.block_stat span.dropcap")
        mcf_gas_produced = mcf_gas_produced_element[1].text.strip() if len(mcf_gas_produced_element)>1 else '0.00'

        
        return {
            "well_status": well_status,
            "well_type": well_type,
            "closest_city": closest_city,
            "barrels_produced": barrels_produced,
            "mcf_gas_produced": mcf_gas_produced
        }

    except AttributeError:
        print(f"[ERROR] Data extraction failed for {well_url} (API# {api_number}, Well: {well_name})")
        return None

# Iterate through all wells in the database and update missing details.
def update_database():
    session = SessionLocal()
    
    wells = session.query(WellStatus).all()
    
    for well in wells:
        print(f"[INFO] Processing API# {well.API} ({well.well_name})...")
        
        well_data = scrape_well_data(well.state, well.county, well.well_name, well.API)
        
        if well_data:
            stmt = (
                update(WellStatus)
                .where(WellStatus.API == well.API)
                .values(
                    well_status=well_data["well_status"],
                    well_type=well_data["well_type"],
                    closest_city=well_data["closest_city"],
                    barrels_produced=well_data["barrels_produced"],
                    mcf_gas_produced=well_data["mcf_gas_produced"]
                )
            )

            session.execute(stmt)
            session.commit()
            print(f"[SUCCESS] Updated API# {well.API} ({well.well_name}) in the database.")
        
        time.sleep(2)  
    
    session.close()

# def update_database():
#     session = SessionLocal()
    
#     try:
#         wells = session.query(WellNotes).all()
        
#         for well in wells:
#             print(f"[INFO] Processing API# {well.API} ({well.well_name})...")

#             well_data = scrape_well_data(well.state, well.county, well.well_name, well.API)
            
#             if well_data:
#                 try:
#                     well.well_status = well_data.get("well_status")
#                     well.well_type = well_data.get("well_type")
#                     well.closest_city = well_data.get("closest_city")
#                     well.barrels_produced = well_data.get("barrels_produced")
#                     well.mcf_gas_produced = well_data.get("mcf_gas_produced")

#                     session.commit()
#                     print(f"[SUCCESS] Updated API# {well.API} ({well.well_name}) in the database.")
#                 except SQLAlchemyError as e:
#                     session.rollback()
#                     print(f"[ERROR] Failed to update API# {well.API}: {e}")

#             time.sleep(2)  # Prevents getting blocked by the server

#     except SQLAlchemyError as e:
#         print(f"[ERROR] Database query failed: {e}")
    
#     finally:
#         session.close()


if __name__ == "__main__":
    update_database()
