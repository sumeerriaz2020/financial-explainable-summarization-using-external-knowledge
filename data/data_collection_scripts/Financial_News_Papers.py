import requests
from datetime import datetime, timedelta
import pandas as pd
import time

class WaybackNewsDownloader:
    """
    Use Internet Archive's Wayback Machine to access historical news pages
    Completely free, legal, and has YEARS of historical data
    """
    
    def __init__(self):
        self.wayback_api = "http://archive.org/wayback/available"
    
    def check_if_archived(self, url, date):
        """
        Check if a URL was archived on a specific date
        """
        params = {
            'url': url,
            'timestamp': date.strftime('%Y%m%d')
        }
        
        try:
            response = requests.get(self.wayback_api, params=params, timeout=10)
            data = response.json()
            
            if 'archived_snapshots' in data:
                closest = data['archived_snapshots'].get('closest', {})
                if closest.get('available'):
                    return closest.get('url')
            
            return None
            
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def download_historical_reuters_headlines(self, start_date, end_date):
        """
        Download historical Reuters business headlines from Wayback Machine
        """
        
        reuters_business_url = "https://www.reuters.com/business"
        
        articles = []
        current = start_date
        
        while current <= end_date:
            print(f"Checking archives for {current.strftime('%Y-%m-%d')}...")
            
            archived_url = self.check_if_archived(reuters_business_url, current)
            
            if archived_url:
                print(f"  ✓ Found archive: {archived_url}")
                
                # Download the archived page
                try:
                    response = requests.get(archived_url, timeout=15)
                    # Parse the content (simplified - would need full parsing)
                    
                    articles.append({
                        'date': current.strftime('%Y-%m-%d'),
                        'archive_url': archived_url,
                        'source': 'Reuters (Archived)'
                    })
                    
                except Exception as e:
                    print(f"  ✗ Error downloading: {e}")
            else:
                print(f"  ✗ No archive found")
            
            current += timedelta(days=7)  # Check weekly
            time.sleep(2)  # Be respectful
        
        return articles


# Usage
from datetime import datetime

downloader = WaybackNewsDownloader()

start = datetime(2024, 1, 1)
end = datetime(2024, 10, 12)

articles = downloader.download_historical_reuters_headlines(start, end)