import requests
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session
from sqlalchemy import update
import time
import re
from database import WellInfo, SessionLocal

# convert the abbreviations to the full words/phrases??????
# update database.py??????

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
        barrels_produced = soup.select_one("p.block_stat span.dropcap").text.strip()
        mcf_gas_produced = soup.select("p.block_stat span.dropcap")[1].text.strip()

        
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

def update_database():
    """ Iterate through all wells in the database and update missing details. """
    session = SessionLocal()
    
    wells = session.query(WellInfo).all()
    
    for well in wells:
        print(f"[INFO] Processing API# {well.API} ({well.well_name})...")
        
        well_data = scrape_well_data(well.state, well.county, well.well_name, well.API)
        
        if well_data:
            stmt = (
                update(WellInfo)
                .where(WellInfo.API == well.API)
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
        
        time.sleep(2)  # Prevents getting blocked by the server
    
    session.close()

# Run the update function
update_database()
