import os
import sys
import json
import openai
import asyncio
import requests
import pandas as pd
import sqlalchemy as sql
from pyzerox import zerox
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService

from database import SessionLocal, RawData


#* TODO: Define the function to retrieve the data from the reddit
def get_data(subreddit: str, num_posts: int):
    pass


#* TODO: Define the function store the data into json file
def store_data(data: dict, file_path: str):
    pass
