import os
import re
import time
import requests
import pprint as pp
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.common.exceptions import TimeoutException, WebDriverException

from database import SessionLocal, PostInfo


#* Define the function to retrieve the data from the reddit
def post_retrieval(post_cnt):
    global post_dict
    
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
    
    driver.get(base_url)
    time.sleep(3)
    
    current_cnt = 0
    attempts = 0
    max_attempts = 5
    #? mimic the scrolling behavior of the human
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while current_cnt < post_cnt:
        attempts += 1
        
        try:
            page = driver.page_source
            soup = BeautifulSoup(page, "html.parser")
           
            post_dict = post_preprocessing(soup)
            
            current_cnt = len(post_dict) + 1
            print(f"\r[INFO] Retrieved Number of Posts: {current_cnt}", end="", flush=True)
            if current_cnt > post_cnt:
                overflow = current_cnt - post_cnt
                post_dict = dict(list(post_dict.items())[:-overflow])
                break
            elif current_cnt == post_cnt:
                break
            
            #? Scroll down to the bottom of the page
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(10)
            
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print("[INFO] No more posts to retrieve, stopping the process...")
                break
            
            last_height = new_height
            attempts = 0
        
        except TimeoutException as e:
            print(f"[ERROR] Timeout error: {e}")
            print(f"[INFO] Attempt {attempts + 1} of {max_attempts}")
            if attempts >= max_attempts:
                time.sleep(10)
            continue
        
        except WebDriverException as e:
            print(f"[ERROR] Browser error: {e}")
            break
    
    driver.quit()
    return post_dict


#* Define the function to preprocess the data
def post_preprocessing(soup):
    global post_dict
    
    articles = soup.find_all("article")
    for article in articles:
        post = article.find("shreddit-post")
        
        #? Check if the element is a post
        if not post:
            continue
        
        #? Check if it is the data science post rather than advertisement or something else
        domain = post.get("domain")
        if domain != "self.MachineLearning":
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
        
        #? Add the post to the dictionary
        post_dict[post_id] = {
            "title": title,
            "post_id": post_id,
            "author_id": author_id,
            "content": content,
            "comment_count": comment_cnt,
            "datetime": dt_obj,
            "url": post_url
        }
    return post_dict


#* Define the function to retrieve the comments
def comment_retrieval(base_url, id):
#    reddit_url = base_url.replace("r/MachineLearning/", "")
    post_id = id.replace("t3_", "")
    comment_url = base_url + "comments/" + post_id + ".json"
    
    headers = {"Accept": "*/*"}
    response = None
    sleep_time = 3
    
    while True:
        try:
            response = requests.get(comment_url, headers=headers, timeout=60)
        except requests.RequestException:
            time.sleep(sleep_time)
            sleep_time *= 2
            continue
        
        if response.status_code == 200:
            break
        
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                wait_time = int(retry_after)
            else:
                wait_time = sleep_time
            
            time.sleep(wait_time)
            sleep_time *= 2
        else:
            time.sleep(sleep_time)
            sleep_time *= 2
    
    comments_data = response.json()
    pp.pformat(comments_data, indent=4)
    
    return comments_data


def extract_comment_text(comments):
    text_list = []
    
    if isinstance(comments, dict) and "data" in comments:
        comment_data = comments["data"]
        
        if comments.get("kind") == "t1" and "body" in comment_data:
            text_list.append(comment_data["body"])
        
        children = comment_data.get("children", [])
        if isinstance(children, list):
            for child in children:
                text_list.extend(extract_comment_text(child))
        
        replies = comment_data.get("replies")
        if replies and isinstance(replies, dict):
            replies_data = replies.get("data", {})
            reply_children = replies_data.get("children", [])
            if isinstance(reply_children, list):
                for reply in reply_children:
                    text_list.extend(extract_comment_text(reply))
    
    elif isinstance(comments, list):
        for item in comments:
            text_list.extend(extract_comment_text(item))
    
    return text_list


def comment_preprocessing():
    global post_dict
    
    for post_id, data in post_dict.items():
        if "comments" in data and isinstance(data["comments"], dict):
            comments = data["comments"]
            for i, comment in comments.items():
                text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", "", comment).strip()
                comments[i] = text
            post_dict[post_id]["comments"] = comments
    
    return post_dict


#* Define the function store the data into txt file
def store_data(post_id, output_path):
    global post_dict
    
    session = SessionLocal()
    post_title = post_dict[post_id]["title"]
    post_text = post_dict[post_id]["content"]
    comment_dict = post_dict[post_id]["comments"]
    
    file_path = os.path.join(output_path, f"{post_id}.txt")
    with open(file_path, "w") as f:
        f.write(f"{post_title}\n\n")
        f.write(f"{post_text}\n\n")
        if comment_dict:
            for comment in comment_dict.values():
                f.write(f"{comment}\n\n")
    
    try:
        existing_post = session.query(PostInfo).filter_by(post_id=post_id).first()
        if existing_post:
                existing_post.title = post_title
                existing_post.author_id = post_dict[post_id]["author_id"]
                existing_post.comment_count = post_dict[post_id]["comment_count"]
                existing_post.datetime = post_dict[post_id]["datetime"]
                existing_post.url = post_dict[post_id]["url"]
                existing_post.file_path = file_path
        else:
            session.add(PostInfo(
                post_id=post_id,
                title=post_title,
                author_id=post_dict[post_id]["author_id"],
                comment_count=post_dict[post_id]["comment_count"],
                datetime=post_dict[post_id]["datetime"],
                url=post_dict[post_id]["url"],
                file_path=file_path
            ))
        
        session.commit()
    
    except Exception as e:
        print(f"[ERROR] Failed to store data: {e}")
    finally:
        session.close()


#* Define the main function
def workflow(post_cnt):
    global post_dict, base_url
    
    print("\n[INFO] Retrieving the posts from Reddit...")
    post_retrieval(post_cnt)
    
    for post_id in post_dict.keys():
        comments = comment_retrieval(base_url, post_id)
        time.sleep(1)
        comment_list = extract_comment_text(comments)
        if comment_list is None:
            continue
        
        post_dict[post_id]["comments"] = dict(enumerate(comment_list))
    print(f"[INFO] Successfully retrieved {len(post_dict)} posts!")
    
    print("\n[INFO] Preprocessing the post data...")
    comment_preprocessing()
    print("[INFO] Successfully preprocessed the post data!")
    
    print("\n[INFO] Storing the data into txt file...")
    for post_id in post_dict.keys():
        store_data(post_id, output_path)
    print("[INFO] Successfully stored the data into txt file!")


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(CURRENT_DIR, "../data/processed_data/")
    print(output_path)
    post_dict = {}
    base_url = "https://www.reddit.com/r/MachineLearning/"
    
    while True:
        post_cnt = input("\n[System] Enter the number of posts to retrieve (skip for all): ")
        if post_cnt == "":
            post_cnt = 100000
        elif post_cnt.isdigit():
            post_cnt = int(post_cnt)
        else:
            print("\n[INFO]Invalid input. Please enter a valid number.")
    
        workflow(post_cnt)
        break
