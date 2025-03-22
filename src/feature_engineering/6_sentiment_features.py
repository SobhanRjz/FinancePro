import requests
import snscrape.modules.twitter as sntwitter
import praw
import pandas as pd
from datetime import datetime, timedelta
from textblob import TextBlob
from pytrends.request import TrendReq
import logging
import glob
import os
import webbrowser
from selenium.webdriver.common.action_chains import ActionChains
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import pyautogui
import requests
import time
import json
import os
import re
import random

#import requests
import curlify  # Converts `requests` object to cURL command
import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import threading
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import pickle

# Get proxy server information from environment variables
proxy_host = os.environ.get('HTTP_PROXY')  # or 'http_proxy' depending on your environment
proxy_port = os.environ.get('HTTP_PROXY_PORT')  # or 'http_proxy_port' depending on your environment

# If the proxy server requires authentication, also get the username and password from environment variables
proxy_username = os.environ.get('HTTP_PROXY_USERNAME')
proxy_password = os.environ.get('HTTP_PROXY_PASSWORD')
# Set Proxy for WebDriver Manager
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:10808"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:10808"

# Build the proxy dictionary
# Configure proxies to route traffic through V2Ray (assuming it's listening on localhost:1080)
proxies = {
    "http": "socks5://127.0.0.1:10808",
    "https": "socks5://127.0.0.1:10808"
}



class SentimentFeatureExtractor:
    def __init__(self, data_dir='data'):
        self.CHROME_PROFILE_PATH = r"C:\Users\sobha\AppData\Local\Google\Chrome\User Data\Default"
        self.Google_proxies = ['http://127.0.0.1:10808']
        self.proxies = {
            "http": "socks5://127.0.0.1:10808",
            "https": "socks5://127.0.0.1:10808"
        }
        self.timeframe_map = {
            '1d': 'D',
            '4h': '4H',
            '1h': 'H',
            '5m': '5min'
        }
        self.minTimeWait = 4
        self.maxTimeWait = 10
        
        #self.pytrends = TrendReq(hl='en-US', tz=360, proxies=self.Google_proxies, timeout=(10, 25))
        self.data_dir = data_dir
        self.Github_api_key = "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"
        #self.driver = self.Initialize_driver()
        self.headers = {
        "authority": "trends.google.com",
        "method": "GET",
        "path": "/trends/api/widgetdata/multiline?hl=en-US&tz=-210&req=%7B%22time%22:%222025-03-17T08%5C%5C:47%5C%5C:11+2025-03-18T08%5C%5C:47%5C%5C:11%22,%22resolution%22:%22EIGHT_MINUTE%22,%22locale%22:%22en-US%22,%22comparisonItem%22:%5B%7B%22geo%22:%7B%22country%22:%22IR%22%7D,%22complexKeywordsRestriction%22:%7B%22keyword%22:%5B%7B%22type%22:%22BROAD%22,%22value%22:%22bit%22%7D%5D%7D%7D%5D,%22requestOptions%22:%7B%22property%22:%22%22,%22backend%22:%22CM%22,%22category%22:0%7D,%22userConfig%22:%7B%22userType%22:%22USER_TYPE_LEGIT_USER%22%7D%7D&token=APP6_UEAAAAAZ9qEj82hrTsHHh6KYrXp6zdtxv8rGYgP&tz=-210",
        "scheme": "https",
        "accept": "application/json, text/plain, */*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "no-cache",
        "cookie": "__utma=10102256.1966798290.1742111854.1742210876.1742210876.1; __utmz=10102256.1742210876.1.1.utmcsr=trends.google.com|utmccn=(referral)|utmcmd=referral|utmcct=/; SEARCH_SAMESITE=CgQIqJ0B; AEC=AVcja2dX8gFHncmQVBrQzowFtUCNUH-I3xVbhZfwscwuEi7tKT9H5vxq_HY; HSID=A1BuInV6_Dan_AB1C; SSID=A-n7XV67uef2qp5Tc; APISID=JnSO-yZHt59YXQSm/AnKKIv_Kfo0Q0xPbp; SAPISID=nnnMbE9ZUnrse0yX/ARblQaJyt7VgcOwqZ; __Secure-1PAPISID=nnnMbE9ZUnrse0yX/ARblQaJyt7VgcOwqZ; __Secure-3PAPISID=nnnMbE9ZUnrse0yX/ARblQaJyt7VgcOwqZ; OTZ=7996798_42_42_114990_38_379890; SID=g.a000uwhL9TuywDBMZYfCvL9mgZGglZ5FXOElzTduoSJC-bOCU7WuL9YR6l8bOdJvDwk9WTMlJgACgYKAWESARYSFQHGX2Mig4u_wRlAaVWEINFSTzRb_xoVAUF8yKrxKZYbXdX1Sb6gJq7uWiUG0076; __Secure-1PSID=g.a000uwhL9TuywDBMZYfCvL9mgZGglZ5FXOElzTduoSJC-bOCU7WudTGCsCd1WZKmIt4a7IE8PgACgYKAVYSARYSFQHGX2MiTl1hLcbJzRN5UxeManG5rBoVAUF8yKqffMmRM-YJ0QohmeyOe3QC0076; __Secure-3PSID=g.a000uwhL9TuywDBMZYfCvL9mgZGglZ5FXOElzTduoSJC-bOCU7WujIv8gpiHX24i7YGfDtP_2gACgYKAQoSARYSFQHGX2MiVbcoQuPOFVisz2l4rJTIjxoVAUF8yKqRM6BojnLT19M66By5HBwI0076; _gid=GA1.3.86954785.1742210873; _ga=GA1.1.1966798290.1742111854; NID=522=FD5Q-0LYUrdtJKELYEdTrmWeHd8DyUrrq_h5_79njQuHwGkoT9MLgxNOdzDOe_BYgPlCqKLCm0LuXYiPUn_2tDTz4FvyNf490zm8cfLPvHizA00J_DMjkIsk1OU_tLR1ruSqMWN2VsccsqeqbJ4cVFOWQip03YrlJNKRaCJFnbSYnPMMLk7zLHEqlosEmO9RGXOVIWZJiGz-FDdP9nUpuY_vVp3BTwFSRQC9cflOrhf7okBxTBIrWd465LqLXTIth8a8N2gHnql1B9pm5Y2TqL1J3z7CS2tdG7mOZifRnuMo2kMUtO6Hu_qFVByN7f8M2hNFCNIPKsOWppQN1paES3v9pKeKAoHFYNJ2mlKqVKKQj0ucEFf4_vrJ4GTAcZTRj4zhoK4pw6htEWDgxx63FU26F8l8W6h2Fv_BZeR3P9vcQeFXVe5ZrnGbxqYVUwtxGuqrQCGTTeYdZtMiJ3Iy9lfX-jr7akj37SY18_AP9SGCR5KWwJNkPyNyJcsahGIodx1FRQ_nguj6A1446VeHIDa74enG0gvFYO1t66wZKBL-2lPtIk-XkRujYl6C-TJ2-zBvqO9425nh-r1zsefE9w1v9nsO9dpxt-83W4ZrhtAjLLHgJGAcKICAn2BLJQgw3Xt3TchzP_qLR8GyVz4zIT2yg15Fg4CgUvePRoLEOmBcwqqoK8JbNNL7JmGhkSAoQ03zaIZX1LZ-kBwXpHip7VLlaUcUElUwdEgKaYHWFK8FB-hiTZ5sbEgaktAzphNCM17nLSUcN8HtVEgdOVk5L36mm4_VguNInZD_gewqePkmvnCh00N2ZOb_SHlK-JKcZhcvjFxKaaFMWZrOwPvHNXrS-z6kWbFSwxiryVD04IrLC0uC3sIGZBgqm6h5cBpIfjmcxod8PnPOHpFOxQ-7MUy-qgGz7RcLYD_B93iGzA4G5KdRzypFzM9XZTS42dN9TousjOWHYhxZMNe_DgEFU6bWIS8hddqDYTIxQI8uTMmrf8rBdQm0JcH4gI5MeuNPDiLK-Q2PO1xf-0_z5XydNwe9ImRxhsPXvrWmjNXoz6F6QofcsvYqgzDBfxIUnfTyY0WPjcnj9bj_cXcoDQleygYiSBBJnyMe7uLzMbaoaAuU4UsOVLu0lWHgNHlmPn1Uv00pXzXYP_CDuAb9GmV0MjIIzLNArMoZcJFROLFAL7VzdS9ga4CzPeii-O-nuTkw_zKzPEUSVnP9DqhCjUux6Q4YZRJ4kTYKckj0_pT15ECKlINMGJvnMVi0bOAMB_SNOSxIIr1UGwaYXnyT1KR0UiJKwzmwH0G-8fJyey4suKiiNumNsF8fWQkCUmr2Sa06sWveyj7ILMmh5dXOrqlEHYCdeW8tA_pBTBD7pJ_Ux2mSkAbEy2ZGxVSKR3gIUbvvrgVsHaUVofAYLWwHQtOAPcmAYldr_mkZ4mGD3hnBZEzQ-g7LW4nrwFS-xkzNlupd; __Secure-ENID=26.SE=LvJv2MjrZRWajc0HmSLWdE2nkG2sT3crvyc-A4kwargb-Ph7IV7Ez18iIiHcHDLeFRyuhDvTY2M83TYWpbXXhI6g4Hs9Ttv3ZjhGkeFZruenwN0RcQouNfM8h1Da1XSWQ8K2Io3oP0IUR1kj_o_spn36U2CVvqKXsFsPSIVm5s1vwaYpeE4ZFxonGLIgX8gpDvG6VSCzLECSJ47yuVkmK4-Q2ZHjFz404mqbBZ76nLp23e0rjVA5OuCP9FCp5xzr3rfWECBgh3uL-_t5IvmJelogUE-7zR6xDymskNhM6lcGGgBiFV212fnt-tTH1SHPrjMXm3dX-Jz0E-gFAPexqEYnwlCzU0uiLNWve5fegCtS3osabh50jNr76FapcNgIZ9qntevLq_wpxqjXTMMlbrD5QxiSZWuQAZR140aUZt8CpL4LmRy8rzFBpx82Y0TkkHqMAobL2hsJMG9lcQ0BfOOP3hA10UQi8n9N3_4twVDEE5pcjm59SUeM1ITfmEkZeyRSohvHMixAqMw0K36cGvup4pgW7zGZJ8gB; __Secure-1PSIDCC=AKEyXzXTHFdHg-rpg9vMDBDI_7qaOeNbkND-BsqD82TGEqFg_Jtsq5f69AZv28VoiptQrcf4jYs; __Secure-3PSIDCC=AKEyXzXyj93AY_xCz1cMMRTdiiRIZySneKzD4PpNHXhyLMDJE6kZTR9hvI-MEZJwySR3yzCR2y0",
        "pragma": "no-cache",
        "priority": "u=1, i",
        "referer": "https://trends.google.com/trends/explore?q=bit&date=now%201-d&geo=IR&hl=en",
        "sec-ch-ua": "\"Chromium\";v=\"134\", \"Not:A-Brand\";v=\"24\", \"Google Chrome\";v=\"134\"",
        "sec-ch-ua-arch": "x86",
        "sec-ch-ua-bitness": "64",
        "sec-ch-ua-form-factors": "Desktop",
        "sec-ch-ua-full-version": "134.0.6998.89",
        "sec-ch-ua-full-version-list": "\"Chromium\";v=\"134.0.6998.89\", \"Not:A-Brand\";v=\"24.0.0.0\", \"Google Chrome\";v=\"134.0.6998.89\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-model": "\"\"",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-ch-ua-platform-version": "15.0.0",
        "sec-ch-ua-wow64": "?0",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
        }
        self.logs = []
        self.stop_logging = False
        #self.log_thread = threading.Thread(target=self._collect_logs, daemon=True)
        #self.log_thread.start()

    def Initialize_driver(self):
        # Configure Chrome options
        options = webdriver.ChromeOptions()
        options.add_argument(f"--user-data-dir={self.CHROME_PROFILE_PATH}")
        options.add_argument("--profile-directory=Default")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-features=AutomationControlled")
        options.set_capability("goog:loggingPrefs", {"performance": "ALL", "browser": "ALL"})
        options.add_argument("--disable-gpu")
        options.add_argument("--enable-logging")  # Enable logging  
        options.add_argument("--log-level=0")  # Capture all logs
        # Enable Chrome DevTools logging
        caps = DesiredCapabilities.CHROME
        caps["goog:loggingPrefs"] = {"performance": "ALL"}
        
        #options.add_argument(f'--proxy-server=socks5://127.0.0.1:10808')
        #options.add_argument("--headless=new")
        #options.add_argument("--disable-dev-shm-usage")
        #options.add_argument("--remote-debugging-port=9222")

        driver = uc.Chrome(options=options, desired_capabilities=caps)
        return driver
    
    def clear_cookies_and_cache(self):
        """Clear browser cookies and cache."""
        self.driver.delete_all_cookies()  # Clear cookies
        self.driver.execute_script("window.localStorage.clear();")  # Clear local storage
        self.driver.execute_script("window.sessionStorage.clear();")  # Clear session storage
        self.driver.get("chrome://settings/clearBrowserData")
        time.sleep(2)  # Wait for settings page to load

        # Simulate pressing "Tab" key and "Enter" to clear cache
        actions = webdriver.ActionChains(self.driver)
        actions.send_keys(Keys.TAB * 12 + Keys.ENTER)
        actions.perform()
        time.sleep(5)  # Wait for cache to clear

    def reset_driver(self):
        """Reset the driver and clear cookies and cache."""
        self.driver.quit()
        self.driver = self.Initialize_driver()
        self.clear_cookies_and_cache()

    def resample_data(self, df: pd.DataFrame, timeframe: str, start_time: str) -> pd.DataFrame:
        """
        Resample data to the specified timeframe
        
        Args:
            df (pd.DataFrame): DataFrame to resample
            timeframe (str): Timeframe to resample to (e.g., '1d', '4h', '1h', '5m')
            
        Returns:
            pd.DataFrame: Resampled DataFrame
        """
        # Get resample frequency from map, default to daily
        freq = self.timeframe_map.get(timeframe.lower(), 'D')
        df = df.bfill().ffill()
        # For other data, use forward fill
        resampled_df = df.resample(freq, origin=df.index[0]).asfreq()
        resampled_df = resampled_df.ffill().bfill()
            
        # Find the closest timestamp to start_time
        start_timestamp = pd.to_datetime(start_time)
        resampled_df = resampled_df[resampled_df.index >= (start_timestamp - pd.Timedelta(minutes=resampled_df.index.freq.n))]
        
        # Adjust index to align with start_time
        if len(resampled_df) > 0:
            time_diff = resampled_df.index[0] - start_timestamp
            resampled_df.index = resampled_df.index - pd.Timedelta(minutes=time_diff.total_seconds()/60)
        return resampled_df
    

    def fetch_github_activity(self, repo="bitcoin/bitcoin", start_date="2015-01-01", timeframe="D"):
        """
        Fetch GitHub repository commits and activity metrics for blockchain projects.
        
        Args:
            repo (str): Repository name in format 'owner/repo'
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format (defaults to current date)
            api_key (str): GitHub API key for authentication
            
        Returns:
            pd.DataFrame: DataFrame with daily GitHub activity metrics
        """
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Authentication
        headers = {
            "Authorization": f"token {self.Github_api_key}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Date range
        date_range = pd.date_range(start=start_date, end=end_date)
        daily_commit_counts = []
        
        # Fetch commits for each day
        for single_date in date_range:
            day_str = single_date.strftime("%Y-%m-%d")
            
            url = f"https://api.github.com/search/commits?q=repo:{repo}+committer-date:{day_str}..{day_str}"
            
            while True:
                try:
                    response = requests.get(url, headers=headers, proxies=proxies)
                    
                    if response.status_code == 200:
                        commit_data = response.json()
                        commit_count = commit_data.get("total_count", 0)
                        
                        daily_commit_counts.append({
                            "date": day_str, 
                            "commit_count": commit_count
                        })
                        logging.info(f"✅ {day_str}: {commit_count} commits")
                        break  # Successfully got data, exit the loop
                    else:
                        logging.error(f"❌ Error {response.status_code} for {day_str}: {response.text}")
                        time.sleep(5)  # Respect GitHub rate limits
                        continue
                
                except Exception as e:
                    logging.error(f"Error fetching {day_str}: {e}")
                    time.sleep(5)  # Add delay before retry
                    continue
        
        # Convert to DataFrame
        if not daily_commit_counts:
            logging.warning(f"No commits found for {repo} in the specified date range")
            return None
            
        df = pd.DataFrame(daily_commit_counts)
        df['timestamp'] = pd.to_datetime(df['date'])
        df.set_index('timestamp', inplace=True)
        df = self.resample_data(df, timeframe, start_date)

        df.ffill().bfill()
        df.drop(columns=['date'], inplace=True)
        
        logging.info(f"Successfully processed {len(df)} days of activity data")
        return df
    def fetch_google_trends_score(self, keyword="Bitcoin"):
        """Fetch Google Trends search interest for the keyword."""
        self.pytrends.build_payload([keyword], timeframe='now 7-d')
        trends = self.pytrends.interest_over_time()
        score = trends[keyword].iloc[-1] if not trends.empty else 0
        logging.info(f"Google Trends Score for {keyword}: {score}")
        return score
        
    def fetch_daily_google_trends(self, keywords, timeframe, start_date, end_date):
        """
        Fetch Google Trends data for multiple keywords over a date range.
        
        Args:
            keywords (list): List of search keywords
            timeframe (str): Time interval ('1h', '1d', etc)
            start_date (datetime): Start of date range
            end_date (datetime): End of date range
            
        Returns:
            pd.DataFrame: DataFrame with trends data for each keyword
        """
        try:
            json_data_list = []
            chunk_size = '8ME' if self.timeframe_map[timeframe] == 'D' else '7D'
            
            # Split date range into chunks
            date_range = pd.date_range(start=start_date, end=end_date, freq=chunk_size)
            # Check if start date is earlier than the first date in range
            if pd.to_datetime(start_date) < pd.to_datetime(date_range[0]):
                date_range = pd.DatetimeIndex([start_date]).append(date_range)
            # Check if end date is later than the last date in range
            if pd.to_datetime(date_range[-1]) < pd.to_datetime(end_date):
                date_range = date_range.append(pd.DatetimeIndex([end_date]))

            # Process each date chunk
            for i in range(len(date_range)-1):
                while True:
                    try:
                        chunk_start = date_range[i].strftime("%Y-%m-%d" + ("T%H" if chunk_size == '7D' else ''))
                        chunk_end = date_range[i+1].strftime("%Y-%m-%d" + ("T%H" if chunk_size == '7D' else ''))
                            
                        # Process keywords in chunks of 5 (Google Trends limit)
                        chunk_dfs = []
                        for j in range(0, len(keywords), 5):
                            keyword_chunk = keywords[j:j+5]
                            encoded_keywords = ','.join(k.replace(' ', '%20') for k in keyword_chunk)
                            url = f"https://trends.google.com/trends/explore?date={chunk_start}%20{chunk_end}&q={encoded_keywords}&hl=en"
                            
                            # Fetch data with retries
                            df = self._fetch_chunk_data(url, keyword_chunk)
                            if not df.empty:
                                chunk_dfs.append(df)
                                logging.info(f"Successfully fetched trends data for keywords {keyword_chunk}")
                        
                        # Merge dataframes with same timestamp index
                        if chunk_dfs:
                            merged_df = pd.concat(chunk_dfs, axis=1, join='outer')
                            json_data_list.append(merged_df)
                            logging.info(f"Successfully fetched and merged trends data from {chunk_start} to {chunk_end}")
                            break
                    except Exception as e:
                        logging.error(f"Error fetching trends data for {chunk_start} to {chunk_end}: {e}")
                        continue
            if json_data_list:

                df = pd.concat(json_data_list)
                df.ffill().bfill()
                if df.index.duplicated().any():
                    df = df[~df.index.duplicated(keep='last')]
                df = self.resample_data(df, timeframe, start_date)
                return df
            

        except Exception as e:
            df = pd.concat(json_data_list)
            df.ffill().bfill()
            df = self.resample_data(df, timeframe, start_date)
            return df

    def _collect_logs(self):
        """Continuously collects performance logs asynchronously."""
        while not self.stop_logging:
            try:
                new_logs = self.driver.get_log("performance")
                if new_logs:
                    self.logs.extend(new_logs)
            except Exception as e:
                logging.warning(f"Error collecting logs: {e}")
            time.sleep(2)  # Avoid excessive CPU usage

    def login_google(self):
        """Login to Google Trends."""
        self.google_username = "sobhantempzadeh2@gmail.com"
        self.google_password = "dD236cGbwu$b"
        self.backup_code = "5451 6609"
        try:
            # Navigate to Google login page
            self.driver.get("https://accounts.google.com/signin")
            time.sleep(random.randint(3, 7))
            
            # Check if already logged in
            if "myaccount.google.com" in self.driver.current_url:
                logging.info("Already logged in to Google")
                return True
                
            # Enter email
            email_field = WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.ID, "identifierId"))
            )
            actions = webdriver.ActionChains(self.driver)
            actions.move_to_element(email_field)
            actions.pause(random.uniform(0.2, 0.5))
            actions.click()
            actions.send_keys(self.google_username)
            actions.pause(random.uniform(0.3, 0.7))
            actions.send_keys(Keys.ENTER)
            actions.perform()
            time.sleep(random.randint(self.minTimeWait, self.maxTimeWait))
            # Try alternative login method with backup code
            try:
                # Click on "Try another way" or similar button
                another_way_button = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Try another way')]"))
                )
                actions = ActionChains(self.driver)
                actions.move_to_element(another_way_button)
                actions.pause(random.uniform(0.2, 0.5))
                actions.click()
                actions.perform()
                time.sleep(random.randint(self.minTimeWait, self.maxTimeWait))
                # Select passkey option with human-like interaction
                # Scroll to ensure the passkey option is fully visible
                # Get the location of the browser window on the second monitor
                                # Wait for the passkey option to be stable and clickable
                passkey_option = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//div[contains(@class, 'VV3oRb') and contains(@role, 'link') and .//div[contains(text(), 'Use your passkey')]]"))
                )

                browser_window = self.driver.get_window_rect()
                
                # Calculate absolute screen coordinates considering multi-monitor setup
                screen_x = browser_window['x'] + passkey_option.location['x'] + passkey_option.size['width'] // 2
                screen_y = passkey_option.location['y'] + passkey_option.size['height'] / 2  + browser_window['y'] + 100
                
                # Get current mouse position before clicking
                original_x, original_y = pyautogui.position()
                
                # Scroll to ensure the passkey option is fully visible
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", passkey_option)
                # Click using absolute screen coordinates
                pyautogui.click(screen_x, screen_y)
                
                # Simulate human hesitation
                time.sleep(random.randint(self.minTimeWait, self.maxTimeWait))
                # Manually enter the 6075 code and press enter
                pyautogui.typewrite('6075')
                time.sleep(random.uniform(0.5, 1.5))
                pyautogui.press('enter')
                
                # Add a small delay to simulate human interaction
                time.sleep(random.uniform(0.5, 1.5))


            except Exception as backup_error:
                logging.error(f"Backup code login also failed: {backup_error}")
            
            logging.info("Successfully logged in to Google")
            return True
            
        except Exception as e:
            logging.error(f"Failed to login to Google: {e}")
            return False

    def click_captcha(self):
        """
        Handle Google reCAPTCHA verification process with robust error handling and logging.
        
        Attempts to solve the captcha by interacting with the reCAPTCHA iframe and verification elements.
        Implements multiple strategies to bypass captcha with human-like interaction.
        
        Raises:
            Exception: If unable to handle captcha after maximum attempts.
        """
        try:
            max_attempts = 5
            attempt = 0
            
            while attempt < max_attempts:
                # Refresh page if on Google Trends to reset potential captcha state
                time.sleep(random.randint(self.minTimeWait, self.maxTimeWait))
                if "trends.google.com/trends/explore" in self.driver.title:
                    logging.info("Detected Google Trends page, refreshing before captcha attempt")
                    self.driver.refresh()
                else:
                    logging.info("Successfully bypassed captcha")
                    break

                # Locate and switch to reCAPTCHA iframe
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.frame_to_be_available_and_switch_to_it(
                            (By.CSS_SELECTOR, "iframe[name^='a-'][src^='https://www.google.com/recaptcha/api2/anchor?']")
                        )
                    )
                except TimeoutException:
                    logging.warning("Could not find primary reCAPTCHA iframe")
                    continue

                # Locate reCAPTCHA checkbox with multiple fallback strategies
                try:
                    recaptcha_checkbox = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, "div.recaptcha-checkbox-border"))
                    )
                except TimeoutException:
                    try:
                        recaptcha_checkbox = WebDriverWait(self.driver, 10).until(
                            EC.element_to_be_clickable((By.XPATH, "//span[@id='recaptcha-anchor']"))
                        )
                    except TimeoutException:
                        logging.error("Failed to locate reCAPTCHA element")
                        self.driver.switch_to.default_content()
                        continue

                # Perform human-like mouse interaction
                actions = ActionChains(self.driver)
                try:
                    actions.move_by_offset(0, 0)
                    actions.move_to_element(recaptcha_checkbox)
                    actions.pause(random.uniform(0.3, 0.8))
                    actions.click()
                    actions.perform()
                except Exception as movement_error:
                    logging.warning(f"Mouse movement failed: {movement_error}")
                    try:
                        recaptcha_checkbox.click()
                    except Exception as click_error:
                        logging.error(f"Direct click failed: {click_error}")
                        continue

                # Attempt to click verification button
                time.sleep(random.randint(self.minTimeWait, self.maxTimeWait))
                try:
                    self.driver.switch_to.default_content()
                    verification_button = self._find_verification_button()
                    
                    if verification_button:
                        actions = ActionChains(self.driver)
                        actions.move_to_element(verification_button)
                        actions.pause(random.uniform(0.2, 0.5))
                        actions.click()
                        actions.perform()
                        logging.info("Clicked verification button")
                except Exception as verification_error:
                    logging.error(f"Verification button interaction failed: {verification_error}")

                self.driver.switch_to.default_content()
                time.sleep(random.randint(self.minTimeWait, self.maxTimeWait))

                # Check page status
                if "trends.google.com/trends/explore" in self.driver.title:
                    logging.info(f"Still on Google Trends after attempt {attempt+1}")
                    attempt += 1
                else:
                    logging.info("Successfully passed captcha")
                    break
                    
            if attempt >= max_attempts:
                logging.warning(f"Failed to pass captcha after {max_attempts} attempts")
                
        except Exception as e:
            logging.error(f"Captcha handling failed: {e}")

    def _find_verification_button(self):
        """
        Locate the 'I'm human' verification button across different iframe contexts.
        
        Returns:
            WebElement: Verification button element if found, else None
        """
        for frame in self.driver.find_elements(By.TAG_NAME, 'iframe'):
            try:
                self.driver.switch_to.default_content()
                self.driver.switch_to.frame(frame)

                verification_buttons = self.driver.find_elements(
                    By.CLASS_NAME, 'button-holder.help-button-holder'
                )
                
                if verification_buttons:
                    return verification_buttons[0]
            except Exception:
                continue
        
        return None

    def _restart_driver(self, url = "https://trends.google.com/trends/explore?date=2024-01-01%202024-01-07&q=bitcoin&hl=en"):
        """Restart the Selenium driver and reload the page."""
        logging.info("Restarting WebDriver...")
        self.clear_cookies_and_cache()
        self.driver.close()
        self.driver.quit()
        self.driver = self.Initialize_driver()
        self.login_google()

        time.sleep(random.randint(self.minTimeWait, self.maxTimeWait))
        self.driver.get(url)
        self.click_captcha()

    def _fetch_chunk_data(self, url, keywords, max_retries=10000):
        """Fetch and parse data for a single chunk."""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
        ]
        self.driver.execute_cdp_cmd("Network.setUserAgentOverride", {"userAgent": random.choice(user_agents)})
        self.driver.execute_cdp_cmd("Network.enable", {})
        self.driver.get(url)
        
        
        if not os.path.exists("cookies.pkl"):
            pickle.dump(self.driver.get_cookies(), open("cookies.pkl", "wb"))




        time.sleep(random.randint(self.minTimeWait, self.maxTimeWait))
        #self._restart_driver()
        wait = WebDriverWait(self.driver, 60)
        try:
            wait.until(lambda d: d.execute_script('return document.readyState') == 'complete')
        except TimeoutException:
            logging.warning("Page load timeout, continuing anyway")

        retry_count = 0
        while retry_count < max_retries:
            try:
                unique_items = list({json.dumps(d, sort_keys=True) for d in self.logs})
                self.logs.clear()  # Prevent duplication
                unique_items = [json.loads(d) for d in unique_items]

                for entry in unique_items:
                    try:
                        message = json.loads(entry["message"])["message"]
                        if (message.get("method") == "Network.responseReceived" and 
                            "/api/widgetdata/multiline" in message.get("params", {}).get("response", {}).get("url", "")):
                            
                            request_id = message["params"]["requestId"]
                            response_body = self.driver.execute_cdp_cmd("Network.getResponseBody", {"requestId": request_id})
                            data = self._parse_trends_response(response_body.get("body"), keywords)
                            if not data.empty:
                                return data
                    except Exception as e:
                        logging.warning(f"Error processing log entry: {e}")
                        continue

                retry_count += 1
                if retry_count % 3000 == 0:
                    # Load previously saved cookies
                    #self.driver.refresh()
                    self._restart_driver(url)
                    retry_count = 0
                elif retry_count % 1000 == 0:
                    self.driver.get(url)
                    time.sleep(random.randint(self.minTimeWait, self.maxTimeWait))
                
            
            except Exception as e:
                logging.error(f"Failed to fetch data: {e}")
                retry_count += 1
                if retry_count > 1000 :
                    self._restart_driver(url)
                    retry_count = 0
                time.sleep(3)

        return pd.DataFrame()

    def stop_log_collection(self):
        """Stops the log collection thread."""
        self.stop_logging = True
        self.log_thread.join()

    def _parse_trends_response(self, response_text, keywords):
        """Helper method to parse Google Trends API response."""
        if response_text.startswith(")]}'"):
            response_text = response_text[6:]
            
        json_data = json.loads(response_text)
        timeline_data = json_data['default']['timelineData']
        
        df = pd.DataFrame()
        df['timestamp'] = [entry['formattedTime'] for entry in timeline_data]
        
        # Extract values for each keyword
        for i, keyword in enumerate(keywords):
            df[keyword] = [entry['value'][i] for entry in timeline_data]
            
        # Explicitly specify format when parsing timestamps to avoid warning
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=False)
        df.set_index('timestamp', inplace=True)
        
        return df


    def process_all_files(self):
        """
        Process all order book files, and fetch daily Google Trends data to match each file's timeframe.
        """
        self.keywords = [
            "crypto",
            "bitcoin",
            "ethereum",
            "dogecoin",
            "shiba inu",
            "solana",
            "cardano",
            "Donald Trump",
            "elon musk",
            "bitcoin price",
        ]


        # Find all order book data files
        file_pattern = os.path.join(self.data_dir, 'OHLCV', "*.pkl.gz")
        file_paths = glob.glob(file_pattern)

        if not file_paths:
            logging.warning(f"No order book files found matching: {file_pattern}")
            return
        # Output directory
        output_dir = os.path.join(self.data_dir)
        os.makedirs(output_dir, exist_ok=True)

        # List_of_PKfiles = [r'data\6_process_sentiment_features\Sentiment_Features_2020-03-11_18-30_2021-07-28_18-30.pkl.gz'
        # ,r'data\6_process_sentiment_features\Sentiment_Features_2021-07-28_21-30_2023-04-05_21-30.pkl.gz'
        # ,r'data\6_process_sentiment_features\Sentiment_Features_2023-04-06_00-30_2024-06-06_00-30.pkl.gz'
        # ,r'data\6_process_sentiment_features\Sentiment_Features_2024-06-06_03-30_2025-03-10_14-30.pkl.gz']
        # concat_df = pd.DataFrame()
        # for file_path in List_of_PKfiles:
        #     try:
        #         df = pd.read_pickle(file_path)
        #         concat_df = pd.concat([concat_df, df], axis=0)
        #         logging.info(f"Loaded and concatenated data from: {file_path}")
        #     except Exception as e:
        #         logging.error(f"Failed to load data from {file_path}: {e}")
        
        # # Remove duplicate indices, keeping the last occurrence
        # if concat_df.index.duplicated().any():
        #     concat_df = concat_df[~concat_df.index.duplicated(keep='last')]
        
        # # Sort the index to ensure chronological order
        # concat_df.sort_index(inplace=True)
        # Main_df = concat_df.copy()

        ListBTC_files = [r'data\6_process_sentiment_features\Sentiment_Features_BTCUSDT_1d_10y.pkl.gz',
                         r'data\6_process_sentiment_features\Sentiment_Features_BTCUSDT_1h_3y.pkl.gz',
                         r'data\6_process_sentiment_features\Sentiment_Features_BTCUSDT_4h_5y.pkl.gz',
                         r'data\6_process_sentiment_features\Sentiment_Features_BTCUSDT_5m_1y.pkl.gz',
                         ]
        # logging.info(f"Total concatenated dataframe shape: {concat_df.shape}")
        for file_path in file_paths[3:]:
            #file_path = file_paths[2]

            logging.info(f"Processing file: {file_path}")

            file_name = os.path.basename(file_path)
            
            # Extract symbol, timeframe and period from filename
            match = re.search(r'([A-Z]+)_(\w+)_(\w+)', file_name)
            if match:
                symbol, timeframe, period = match.groups()
                logging.info(f"Processing {symbol} data for {timeframe} timeframe over {period}")

            try:
                # Load order book data
                # Check if the current file is in the ListBTC_files
                # Find the matching file in ListBTC_files and load the corresponding dataframe
                matching_file = next((fileN for fileN in ListBTC_files if file_name in fileN), None)
                if matching_file:
                    dfmain = pd.read_pickle(matching_file)
                    logging.info(f"Loaded main dataframe from: {matching_file}")

                if file_name in ListBTC_files:
                    dfmain = pd.read_pickle(file_path)
                    logging.info(f"Loaded main dataframe from: {file_path}")

                df = pd.read_pickle(file_path)
                start_date = df.index.min()
                end_date = df.index.max()

                #Google_trends = self.fetch_daily_google_trends(self.keywords, timeframe, start_date, end_date)
                # daily_github_activity = self.fetch_github_activity(repo="bitcoin/bitcoin", start_date=start_date, timeframe=timeframe)
                # daily_github_activity = daily_github_activity[daily_github_activity.index >= pd.to_datetime(start_date)]
                # daily_github_activity = daily_github_activity[daily_github_activity.index <= pd.to_datetime(end_date)]
                # daily_github_activity.rename(columns={"commit_count": "github_activity"}, inplace=True)


                daily_github_activity = pd.read_pickle(r'data\6_process_sentiment_features\OnlyGit_Sentiment_Features_BTCUSDT_1d_10y.pkl.gz')
                daily_github_activity = daily_github_activity[daily_github_activity.index >= pd.to_datetime(start_date)]
                daily_github_activity = daily_github_activity[daily_github_activity.index <= pd.to_datetime(end_date)]
                daily_github_activity = self.resample_data(daily_github_activity, timeframe, start_date)
                if daily_github_activity.index.max() < pd.to_datetime(end_date):
                    last_row = daily_github_activity.iloc[-1]
                    additional_dates = pd.date_range(start=daily_github_activity.index.max(), end=pd.to_datetime(end_date), freq=daily_github_activity.index.freq)
                    additional_df = pd.DataFrame(index=additional_dates, columns=daily_github_activity.columns)
                    additional_df.loc[:] = last_row.values
                    daily_github_activity = pd.concat([daily_github_activity, additional_df])

                if daily_github_activity.index.min() > pd.to_datetime(start_date):
                    first_row = daily_github_activity.iloc[0]
                    additional_dates = pd.date_range(start=pd.to_datetime(start_date), end=daily_github_activity.index.min(), freq=daily_github_activity.index.freq)
                    additional_df = pd.DataFrame(index=additional_dates, columns=daily_github_activity.columns)
                    additional_df.loc[:] = first_row.values
                    daily_github_activity = pd.concat([additional_df, daily_github_activity])   

                dfmain = pd.concat([dfmain, daily_github_activity], axis=1)
                dfmain.index.rename('timestamp', inplace=True)
                # Create output directories if they don't exist
                os.makedirs(os.path.join(output_dir, '6_process_sentiment_features'), exist_ok=True)
            
                logging.info(f"Created output directories for sentiment features")
                # Save trends data
                output_file = os.path.join(output_dir, '6_process_sentiment_features', 'GG_Sentiment_Features_' + '_'.join(file_name.split('_')[-3:]))
                dfmain.to_pickle(output_file)


            #     Main_df = Main_df[Main_df.index >= pd.to_datetime(start_date)]
            #     Main_df = Main_df[Main_df.index <= pd.to_datetime(end_date)]
            #     Main_df.sort_index(inplace=True)
            #     Main_df = self.resample_data(Main_df, timeframe, start_date)
            #     Main_df.ffill(inplace=True)
            #     Main_df.bfill(inplace=True)
            #     # Fill remaining time with last available data point
            #     if Main_df.index.max() < pd.to_datetime(end_date):
            #         last_row = Main_df.iloc[-1]
            #         additional_dates = pd.date_range(start=Main_df.index.max(), end=pd.to_datetime(end_date), freq=Main_df.index.freq)
            #         additional_df = pd.DataFrame(index=additional_dates, columns=Main_df.columns)
            #         additional_df.loc[:] = last_row.values
            #         Main_df = pd.concat([Main_df, additional_df[~additional_df.index.isin(Main_df.index)]])
            #         Main_df.sort_index(inplace=True)

            #     # Create output directories if they don't exist
            #     os.makedirs(os.path.join(output_dir, '6_process_sentiment_features'), exist_ok=True)
            
            #     logging.info(f"Created output directories for sentiment features")
            #     # Save trends data
            #     output_file = os.path.join(output_dir, '6_process_sentiment_features', 'Sentiment_Features_' + '_'.join(file_name.split('_')[-3:]))
            #     Main_df.to_pickle(output_file)



            #     logging.info(f"Start date: {start_date}, End date: {end_date}")


            #     #data = pd.read_pickle(file_path)
            #     logging.info(f"Loaded order book data from: {file_path}")
            # except Exception as e:
            #     logging.error(f"Failed to load data from {file_path}: {e}")
            #     continue

            # try:


                


            except Exception as e:
                logging.error(f"Error processing Google Trends for {file_name}: {e}")



if __name__ == "__main__":
    collector = SentimentFeatureExtractor()
    #collector._restart_driver()

    # Process all order book files and save Google Trends data
    collector.process_all_files()
