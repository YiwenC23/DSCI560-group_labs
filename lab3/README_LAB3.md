# DSCI 560: Group Laboratory 3

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

1. Install the SQLAlchemy Library:

```bash
pip install sqlalchemy
```

2. Install the PyMySQL Library:

```bash
pip install pymysql
```

3. Install the YaHoo Finance API:

```bash
pip install yfinance
```

## Execution

1. Data Retrieval and Insertion:

```bash
python yfinance_retrieve.py
"Please enter the username for the database:" [username] #Enter MySQL username.
"Please enter the password for the database:" [password] #Enter MySQL passowrd.
"Please enter the database name:" [database] #Enter the database name where you want to store the data.
```

2. Data Preprocessing:

```bash
python data_preprocessing.py
"Please enter the username for the database:" [username] #Enter MySQL username.
"Please enter the password for the database:" [password] #Enter MySQL passowrd.
"Please enter the database name:" [database] #Enter the database name.
```

3. Portfolio Manipulation and Validation:

```bash
python manipulation_validation.py
"Please enter the username for the database:" [username] #Enter MySQL username.
"Please enter the password for the database:" [password] #Enter MySQL passowrd.
"Please enter the database name:" [database] #Enter the database name.

"Your Options Are:"
"1. Add stock to portfolio"
"2. Remove stock from portfolio"
"3. Display all portfolios"
"4. Exit"
"Select one of the above:" [option] #Choose the appropriate option to operate accordingly.
```

