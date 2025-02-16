# DSCI 560: Group Laboratory 5

## Table of Contents

- [Team Members](#team-members)
- [Installation](#installation)
- [Execution](#execution)

## Team Members

1. Hanlu Ma (USC ID: 1392-9443-71)

2. Zhenyu Chen (USC ID: 2242-3773-15)

4. Fariha Sheikh (USC ID: 9625-7343-53)

## Installation

Install the require libraries:

1. Install the spaCy library and execute the command to download the model for the data preprocessing process:

```bash
# Step 1
pip install spacy

# Step 2
python -m spacy download en_core_web_sm
```

2. Install the **nltk** Library and download the stop words corpus:

```bash
pip install nltk

python3
import nltk
nltk.download('stopwords')
```

3. Install the **gensim** Library:
```bash
pip install gensim
```

4. Install the **scikit-learn** Library:
``` bash
pip install numpy
```

5. Install the **pandas** Library:
``` bash
pip install pandas
```

6. Install the **matplotlib** Library:
``` bash
pip install matplotlib
```

7. Install the **sqlalchemy** Library:
``` bash
pip install sqlalchemy
```

8. Install the **requests** Library:
```bash
pip install requests
```

9. Install the **bs4** (BeautifulSoup) Library:
```bash
pip install bs4
```

10. Install the **selenium** Library:
```bash
pip install selenium
```

10. Install the **webdriver-manager** Library:
```bash
pip install webdriver-manager
```

## Execution

```bash
python update_database.py
"Please enter the username for the database:" [username] #Enter MySQL username.
"Please enter the password for the database:" [password] #Enter MySQL passowrd.
"Please enter the database name:" [database] #Enter the database name where you want to store the data.
```

### Notes
We include many other programs in the script folder, but they are not necessarily all useful as we mainly use it as a function call reference if we need. You can freely check any other scripts for more information that you are interested in. However, our main scripts for this lab will be ‘algorithm.py’, 'data_retrieval.py’, and 'update_database.py'; and everything needed for this lab can all be achieved by running the 'update_database.py'.
