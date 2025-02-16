import re
import os
import sys
import time

from database import SessionLocal, PostInfo
from data_retrieval import comment_retrieval, extract_comment_text

def update_database(updating_time):
    
    allowed_time = int(updating_time) * 60
    start_time = time.time()
    
    session = SessionLocal()
    post_ids = PostInfo.get_post_id(session)
    post_comment_count = PostInfo.get_post_comment_count(session, post_ids)
    
    for post_id in post_ids:
        if time.time() - start_time > allowed_time:
            print(f"[INFO] Time limit of {updating_time} minutes reached. Stopping the update...")
            break
        
        comment_json = comment_retrieval(base_url, post_id)
        if isinstance(comment_json, dict) and "data" in comment_json:
            data = comment_json.get("data", {})
            
            if comment_json.get("kind") == "t3" and "num_comments" in data:
                new_comment_count = data["num_comments"]
                old_comment_count = post_comment_count.get(post_id, 0)
                
                if new_comment_count > old_comment_count:
                    post_title = data.get("title", "")
                    post_content = data.get("selftext", "")
                    post_content = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", "", post_content).strip()
                    
                    comments_text = extract_comment_text(comment_json)
                    comments_text = [re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", "", comment).strip() for comment in comments_text]
                    
                    with open(os.path.join(output_path, f"{post_id}.txt"), "a", encoding="utf-8") as f:
                        f.write(f"{post_title}\n\n")
                        f.write(f"{post_content}\n\n")
                        for comment in comments_text:
                            f.write(f"{comment}\n\n")
                    
                    PostInfo.update_comment_count(session, post_id, new_comment_count)
    
    session.close()


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(CURRENT_DIR, "../data/processed_data/")
    base_url = "https://www.reddit.com/r/datascience/"
    
    updating_time = sys.argv[1]
    update_database(updating_time)