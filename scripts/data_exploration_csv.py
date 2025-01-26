import subprocess
import pandas as pd
def download_dataset():
    subprocess.run(["kaggle", "datasets", "download", "-d", "everydaycodings/job-opportunity-dataset"])
    print("Dataset downloaded successfully.")
def extract_and_save():
    subprocess.run(["unzip", "job-opportunity-dataset.zip", "-d", "job_dataset"])
    df = pd.read_csv('job_dataset/job.csv')
    df.to_csv('job_dataset/job.csv', index=False)
    df.to_excel('job_dataset/job.xlsx', index=False)
    print("Dataset saved in CSV and Excel format.")
if __name__ == "__main__":
    download_dataset()
    extract_and_save()
