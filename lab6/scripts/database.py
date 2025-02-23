import sqlalchemy as sql
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

#* Define the WellInfo Table Class
class WellInfo(Base):
    __tablename__ = "well_info"
    well_name = sql.Column(sql.VARCHAR(255), primary_key=True)
    API = sql.Column(sql.CHAR(12), primary_key=True)
    operator = sql.Column(sql.VARCHAR(255))
    county = sql.Column(sql.VARCHAR(255))
    state = sql.Column(sql.VARCHAR(255))
    footages = sql.Column(sql.VARCHAR(255))
    section = sql.Column(sql.VARCHAR(255))
    township = sql.Column(sql.VARCHAR(255))
    range = sql.Column(sql.VARCHAR(255))
    latitude = sql.Column(sql.VARCHAR(30))
    longitude = sql.Column(sql.VARCHAR(30))
    date_stimulated = sql.Column(sql.VARCHAR(50))
    stimulated_formation = sql.Column(sql.VARCHAR(255))
    top = sql.Column(sql.VARCHAR(255))
    bottom = sql.Column(sql.VARCHAR(255))
    stimulation_stages = sql.Column(sql.Integer)
    volume = sql.Column(sql.VARCHAR(50))
    volume_unites = sql.Column(sql.VARCHAR(50))
    type_treatment = sql.Column(sql.VARCHAR(255))
    acid = sql.Column(sql.VARCHAR(50))
    lbs_proppant = sql.Column(sql.VARCHAR(50))
    maximum_treatment_pressure = sql.Column(sql.VARCHAR(50))
    maximum_treatment_rate = sql.Column(sql.VARCHAR(50))
    details = sql.Column(sql.Text)
    well_status = sql.Column(sql.VARCHAR(50))        
    well_type = sql.Column(sql.VARCHAR(50))         
    closest_city = sql.Column(sql.VARCHAR(50))       
    barrels_produced = sql.Column(sql.VARCHAR(100))  
    mcf_gas_produced = sql.Column(sql.VARCHAR(100))
    
    __table_args__ = (
        sql.Index("idx_well_info_API", "API", unique=True),
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
