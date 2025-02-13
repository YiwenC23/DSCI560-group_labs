import os
import re
import sys
import time
import json
import openai
import asyncio
import pandas as pd
import pyarrow as pa
import sqlalchemy as sql
from pyzerox import zerox
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.common.exceptions import TimeoutException, WebDriverException

# from database import SessionLocal, RawData


#* TODO: Define the function to retrieve the data from the reddit
def data_retrieval(post_cnt=1000):
    global post_dict
    
    url = "https://www.reddit.com/r/datascience/"
    
    options = webdriver.ChromeOptions()
    
    #? Setting a common user-agent
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                            "AppleWebKit/537.36 (KHTML, like Gecko)"
                            "Chrome/116.0.5845.96 Safari/537.36")
    
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    
    driver = webdriver.Chrome(
        options=options,
        service=ChromeService(
            ChromeDriverManager().install()
        )
    )
    
    driver.get(url)
    time.sleep(3)
    
    current_cnt = 0
    attempts = 0
    max_attempts = 5
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while current_cnt < post_cnt:
        attempts += 1
        
        try:
            page = driver.page_source
            soup = BeautifulSoup(page, "html.parser")

            post_dict = data_preprocessing(soup)
            
            current_cnt = len(post_dict)
            print(f"Retrieved {current_cnt} posts")
            
            #? Scroll down to the bottom of the page
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print("No more posts to retrieve, stopping the process")
                break
            
            last_height = new_height
            attempts = 0
        
        except TimeoutException as e:
            print(f"\nTimeout error: {e}")
            print(f"Attempt {attempts + 1} of {max_attempts}")
            if attempts >= max_attempts:
                time.sleep(10)
            continue
            
        except WebDriverException as e:
            print(f"\nBrowser error: {e}")
            break
    
    driver.quit()
    return post_dict


#* TODO: Define the function to preprocess the data
def data_preprocessing(soup):
    global post_dict
    
    articles = soup.find_all("article")
    
    for article in articles:
        post = article.find("shreddit-post")
        
        #? Check if the element is a post
        if not post:
            continue
        
        #? Check if it is the data science post rather than advertisement or something else
        domain = post.get("domain")
        if domain != "self.datascience":
            continue
        
        #? Check if the post is already in the dictionary
        post_id = post.get("id")
        if post_id in post_dict:
            continue
        
        #? Get the post information
        title = post.get("post-title")
        author_id = post.get("author-id")
        comment_cnt = post.get("comment-count")
        post_url = post.get("content-href")
        
        #? Extract and preprocess the post content
        paragraphs = post.find_all("p")
        content = " ".join([p.get_text() for p in paragraphs])
        content = re.sub(r"\s+", " ", content).strip()
        
        #? Extract and preprocess the datetime
        dt = post.get("created-timestamp")
        dt_obj = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f%z")
        dt_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        
        #? Add the post to the dictionary
        post_dict[post_id] = {
            "title": title,
            "post_id": post_id,
            "author_id": author_id,
            "content": content,
            "comment_count": comment_cnt,
            "datetime": dt_str,
            "url": post_url
        }
    
    return post_dict


#* TODO: Define the function store the data into parquet file
def store_data(data, output_path):
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.reset_index().drop(columns=["index"])
    df.to_parquet(output_path)


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(CURRENT_DIR, "../data/processed_data/reddit_datascience.parquet")
    
    post_dict = {}
    data = data_retrieval()
    store_data(data, output_path)
