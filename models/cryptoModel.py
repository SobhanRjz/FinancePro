from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

@dataclass
class Crypto:
    """Data class representing cryptocurrency information"""
    cryptoname: str
    timeframe: str 
    period: int
    timestamp: datetime = datetime.now()
    open: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    close: Optional[Decimal] = None
    volume: Optional[Decimal] = None
    num_trades: Optional[int] = None
    vwap: Optional[Decimal] = None

    @staticmethod
    def get_recommended_period(timeframe: str) -> str:
        """
        Returns the recommended period for a given timeframe based on data availability.
        
        Args:
            timeframe (str): The timeframe (e.g., '3m', '5m', '1h', '4h', '1d')
            
        Returns:
            str: Recommended period for the given timeframe
        """
        timeframe_period_map = {     # 3-minute candles: 1 day period
            '5m': '1y',      # 5-minute candles: 1 year period (recent only)
            '1h': '3y',      # 1-hour candles: 2-3 years
            '4h': '5y',      # 4-hour candles: 5 years
            '1d': '10y'      # 1-day candles: 10 years
        }
        
        return timeframe_period_map.get(timeframe, '1d')  # Default to 1d if timeframe not found
