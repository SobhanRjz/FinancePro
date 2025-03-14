import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime, timedelta

def get_bitcoin_tweets_by_user(username: str, start_date: str, end_date: str, interval: str = "1d"):
    """
    Fetches the count of tweets mentioning 'Bitcoin' from a specific Twitter user.
    
    Parameters:
        username (str): Twitter username (without '@').
        start_date (str): Start date in format 'YYYY-MM-DD'.
        end_date (str): End date in format 'YYYY-MM-DD'.
        interval (str): Time interval ('1d' for daily, '7d' for weekly).
        
    Returns:
        DataFrame with tweet counts per interval.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    delta = timedelta(days=int(interval.replace('d', '')))  # Convert interval to days
    
    results = []

    current = start
    while current < end:
        next_interval = current + delta
        query = f"from:{username} Bitcoin since:{current.strftime('%Y-%m-%d')} until:{next_interval.strftime('%Y-%m-%d')}"
        tweet_count = sum(1 for _ in sntwitter.TwitterSearchScraper(query).get_items())

        results.append({
            "date": current.strftime("%Y-%m-%d"),
            "tweets": tweet_count
        })
        
        print(f"Fetched {tweet_count} tweets from @{username} between {current.strftime('%Y-%m-%d')} and {next_interval.strftime('%Y-%m-%d')}")
        
        current = next_interval

    df = pd.DataFrame(results)


    return df

# Example usage: Get daily Bitcoin tweet counts from 'elonmusk' between 2023-01-01 and 2023-01-10
get_bitcoin_tweets_by_user("elonmusk", "2023-01-01", "2023-01-10", "1d")
