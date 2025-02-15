import sqlalchemy as sql
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

#* Define the Raw Data Table Class
class PostInfo(Base):
    __tablename__ = "post_info"
    post_id = sql.Column(sql.String(255), primary_key=True, index=True)
    title = sql.Column(sql.String(255))
    author_id = sql.Column(sql.String(255))
    comment_count = sql.Column(sql.Integer)
    datetime = sql.Column(sql.DateTime)
    url = sql.Column(sql.String(255))
    file_path = sql.Column(sql.String(255))
    
    __table_args__ = (
        sql.Index("idx_post_info_post_id", "post_id", unique=True),
        {"extend_existing": True}
    )


#* Define the function to connect to the database
def connect_db():
    while True:
        #? Get the database credentials from the user
        db_username = input("[System] Please enter the username for the database: ")
        db_password = input("[System] Please enter the password for the database: ")
        db_name = input("[System] Please enter the database name: ")
        
        conn_url = f"mysql+pymysql://{db_username}:{db_password}@localhost/{db_name}"
    
        try:
            
            print(f"\n[System] Establishing connection to {db_name} database as {db_username}...")
            #? Create database engine with connection pool configuration
            engine = sql.create_engine(
                conn_url,
                pool_size=20,     # Number of maintained idle connections in the pool
                pool_recycle=3600,    # Recycle connections hourly to prevent connection timeout
                max_overflow=10,    # Allow up to 10 additional connections to the pool
                pool_pre_ping=True,     # Validate connection viability before use
                echo=False    # Disable engine logging
                )
            
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            print("\n[INFO] Successfully connected to the database!")
            return engine
        
        except Exception as e:
            print(f"\n[ERROR] Connection failed: {e}")
            print("\n[INFO] Please check your credentials and try again.\n")

#? Create the engine
engine = connect_db()

#? Create the tables
Base.metadata.create_all(bind=engine)

#? Create the Session Library
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,    # Require explicit commit() for transaction control
    autoflush=False,    # Delay SQL emission until flush()/commit() called, enables batch operations
    expire_on_commit=False,    # Keep object attributes accessible after commit
    class_=sql.orm.Session    # Use the SQLAlchemy Session class
)
