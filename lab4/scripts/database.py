import sys
import sqlalchemy as sql
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

#* Define the function to connect to the database
def connect_db():
    
    #? Get the database credentials from the user
    db_username = input("Please enter the username for the database: ")
    db_password = input("Please enter the password for the database: ")
    db_name = input("Please enter the database name: ")
    
    conn_url = f"mysql+pymysql://{db_username}:{db_password}@localhost/{db_name}"
    
    try:
        
        print(f"Establishing connection to {db_name} database as {db_username}...")
        #? Create database engine with connection pool configuration
        engine = sql.create_engine(
            conn_url,
            pool_size=20,     # Number of maintained idle connections in the pool
            pool_recycle=3600,    # Recycle connections hourly to prevent connection timeout
            max_overflow=10,    # Allow up to 10 additional connections to the pool
            pool_pre_ping=True,     # Validate connection viability before use
            echo=False    # Disable engine logging
            )
        

        
        print("Successfully connected to the database!")
    
        return engine
    except Exception as e:
        print(f"Connection failed: {e}")
        raise

#? Create the engine
engine = connect_db()

#? Create the Session Library
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,    # Require explicit commit() for transaction control
    autoflush=False,    # Delay SQL emission until flush()/commit() called, enables batch operations
    expire_on_commit=False,    # Keep object attributes accessible after commit
    class_=sql.orm.Session    # Use the SQLAlchemy Session class
)
