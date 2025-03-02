import psycopg2
import logging
from typing import Dict, Any, Tuple, NamedTuple, Optional
import hashlib
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to Python path to import models
sys.path.append(str(Path(__file__).parent.parent))
from models.cryptoModel import Crypto
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PostgreSQLDatabase:
    """Handles PostgreSQL database operations for soil nailing optimization"""

    def __init__(self) -> None:
        """Initialize database connection with configuration."""
        self.DB_NAME = "bitcoin"
        self.DB_CONFIG = {
            "user": "postgres",
            "password": "123", 
            "host": "127.0.0.2", # Changed from 127.0.0.2 to localhost
            "port": "5432"
        }
        self._config = self.DB_CONFIG.copy()
        self._create_database()
        self._config["dbname"] = self.DB_NAME
        self._conn = None
        self._cursor = None
        self._connect()

    def _create_database(self) -> None:
        """Create database if it doesn't exist."""
        try:
            # Connect to default postgres database first
            conn = psycopg2.connect(**self.DB_CONFIG, dbname="postgres")
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (self.DB_NAME,))
            exists = cursor.fetchone()
            
            if not exists:
                cursor.execute(f'CREATE DATABASE {self.DB_NAME}')
                logger.info(f"Database {self.DB_NAME} created successfully")
            else:
                logger.info(f"Database {self.DB_NAME} already exists")
                
            cursor.close()
            conn.close()
            
        except psycopg2.Error as e:
            logger.error(f"Database creation failed: {e}")
            #raise

    def _connect(self) -> None:
        """Establish connection to PostgreSQL database."""
        try:
            self._conn = psycopg2.connect(**self._config)
            self._cursor = self._conn.cursor()
            logger.info("Successfully connected to PostgreSQL database")
        except psycopg2.Error as e:
            logger.error("Database connection failed: %s", e)
            raise

    def _create_Crypto_data_table(self, cryptoname: str, timeframe: str, period: str) -> None:
        """Create required database tables if they don't exist."""
        try:
            TableName = str.lower(cryptoname + "_" + timeframe + "_" + period)
            # Check if table exists
            self._cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    AND table_name = %s
                )
            """, (TableName,))
            
            table_exists = self._cursor.fetchone()[0]
            
            if table_exists:
                # Drop existing table
                self._cursor.execute(f"DROP TABLE {TableName}")
                logger.info(f"Dropped existing table {TableName}")
            
            # Create table
            self._cursor.execute(f"""
                CREATE TABLE {TableName} (
                    id SERIAL PRIMARY KEY,
                    cryptoname VARCHAR(50) NOT NULL,
                    timeframe VARCHAR(20) NOT NULL,
                    period VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    open DECIMAL(20, 8),
                    high DECIMAL(20, 8),
                    low DECIMAL(20, 8),
                    close DECIMAL(20, 8),
                    volume DECIMAL(20, 8),
                    num_trades INTEGER,
                    vwap DECIMAL(20, 8)
                )
            """)
            logger.info("Crypto data table created successfully")

            self._conn.commit()
            logger.info("Database tables initialized successfully")
        except psycopg2.Error as e:
            logger.error("Table initialization failed: %s", e)
            raise
   
    def save_crypto_data(self, crypto_data: list[Crypto]) -> None:
        """Save cryptocurrency data to the database"""
        if not crypto_data:
            logger.warning("No crypto data provided to save")
            return

        # Get details from first crypto record for logging
        sample_crypto = crypto_data[0]
        table_name = f"{sample_crypto.cryptoname}_{sample_crypto.timeframe}_{sample_crypto.period}"

        try:
            # Create table if it doesn't exist
            self._create_Crypto_data_table(sample_crypto.cryptoname, 
                                         sample_crypto.timeframe, 
                                         str(sample_crypto.period))

            # Prepare batch insert
            insert_query = f"""
                INSERT INTO {table_name} (
                    cryptoname,
                    timeframe, 
                    period,
                    timestamp,
                    open,
                    high,
                    low,
                    close,
                    volume
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            # Convert crypto objects to tuples for batch insert
            values = [(c.cryptoname, c.timeframe, c.period, c.timestamp,
                      c.open, c.high, c.low, c.close, c.volume) 
                     for c in crypto_data]

            # Execute batch insert
            self._cursor.executemany(insert_query, values)
            self._conn.commit()

            logger.info(f"Successfully saved {len(crypto_data)} rows of data to {table_name}")

        except psycopg2.Error as e:
            self._conn.rollback()
            logger.error(f"Error saving data to {table_name}: {e}")
            raise

if __name__ == "__main__":
    # Example configuration and usage
    db = PostgreSQLDatabase()
    db._create_Crypto_data_table("bitcoin", "1h", "3y")

