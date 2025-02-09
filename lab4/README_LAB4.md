# DSCI 560: Group Laboratory 4

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

4. Install the requests library:
```bash
pip install requests
```

## Execution

```bash
python mock_trading_env.py
"Please enter the username for the database:" [username] #Enter MySQL username.
"Please enter the password for the database:" [password] #Enter MySQL passowrd.
"Please enter the database name:" [database] #Enter the database name where you want to store the data.


"Your Options Are:
1. Add stock to portfolio
2. Remove stock from portfolio
3. Display portfolio information
4. Enter the transaction interface
5. Exit
Select one of the above:
"
# Freely enter one number based on your needs. The program will take you to the next step with clear instructions; simply follow what the program generates next.
```

### Notes
We include many other programs in the script folder, but they are not necessarily all useful as we mainly use it as a function call reference if we need. You can freely check any other scripts for more information that you are interested in. However, our main scripts for this lab will be ‘algorithm.py’ and ‘mock_trading_env.py’; and everything needed for the mock trading can all be achieved by running the 'mock_trading_env.py'.
