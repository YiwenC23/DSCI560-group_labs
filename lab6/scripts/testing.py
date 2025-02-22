import re
import pandas as pd 
df = pd.read_csv("API.csv", delimiter = ",")
df.columns = df.columns.str.strip()
#df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
def clean_data(value):
    if isinstance(value, str):
        # Example: remove any leading/trailing spaces and non-alphanumeric characters
        return re.sub(r'\s+', '', value.strip())  # Replace multiple spaces with a single space
    return value 
#pd.set_option('display.max_columns', None)  # To display all columns
#pd.set_option('display.width', None)  # To avoid truncation of long columns
#df.columns = df.columns.str.strip()
df = df.applymap(clean_data)
print(df["API"].to_string(index=False))
#print(df.columns)
