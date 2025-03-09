# DSCI 560: Group Laboratory 8

## Table of Contents

- [Team Members](#team-members)
- [Prerequisite](#prerequisite)
- [Installation](#installation)
- [Execution](#execution)

## Team Members

1. Hanlu Ma (USC ID: 1392-9443-71)

2. Zhenyu Chen (USC ID: 2242-3773-15)

4. Fariha Sheikh (USC ID: 9625-7343-53)

## Installation

Install the require Modules:

1. Install the `sqlalchemy` module:

```bash
pip install sqlalchemy
```

2. Install the `requests` module:

```bash
pip install requests
```

3. Install the `beautifulsoup4` module:

```bash
pip install beautifulsoup4
```

4. Install the `selenium` module:

```bash
pip install selenium
```

5. Install the `webdriver-manager` module:

```bash
pip install webdriver-manager
```

6. Install the `spacy` module:

```bash
pip install spacy
```

7. Install the `numpy` module:

```bash
pip install numpy
```

8. Install the `nltk` module:

```bash
pip install nltk

# Over your python terminal, do the following:
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

9. Install the `numpy` module:

```bash
pip install numpy
```

10. Install the `genism` module:

```bash
pip install genism
```

11. Install the `scikit-learn` module:

```bash
pip install scikit-learn
```

12. Install the `matplotlib` module:

```bash
pip install matplotlib
```

## Execution

### Stage 1: Generate the data

Step 1. Execute the `database` script:

```bash
python3 database.py

"Please enter the username for the database:" [username] #Enter MySQL username.
"Please enter the password for the database:" [password] #Enter MySQL passowrd.
"Please enter the database name:" [database] #Enter the database name where you want to store the data.
```

Step 2. Execute the `data_retrieval` script:

```bash
python3 data_retrieval.py

"Please enter the username for the database:" [username] #Enter MySQL username.
"Please enter the password for the database:" [password] #Enter MySQL passowrd.
"Please enter the database name:" [database] #Enter the database name where you want to store the data.
```

### Stage 2: Run the model and evaluate the performance

Step 3. Execute the `evaluation` script:

Since we directly use the extracted files for analysis, please change the file/directory path in line 18 of `doc2vec` script and line 21 of `word2vec_BoW` script to make sure that the files are located properly.

```bash
python3 evaluation.py
```