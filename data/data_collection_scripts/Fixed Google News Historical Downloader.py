import feedparser
import pandas as pd
from datetime import datetime, timedelta
import time
from urllib.parse import quote, urlencode

class GoogleNewsHistoricalDownloader:
    """
    Fixed version with proper URL encoding
    """
    
    def download_google_news_by_date_range(self, query, start_date, end_date):
        """
        Download Google News for specific date range with proper URL encoding
        """
        
        # Build the search query with date parameters
        date_query = f"{query} after:{start_date} before:{end_date}"
        
        # Properly encode the query
        encoded_query = quote(date_query)
        
        # Build the complete URL
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        print(f"Searching: {query} from {start_date} to {end_date}")
        
        try:
            feed = feedparser.parse(url)
            
            articles = []
            for entry in feed.entries:
                articles.append({
                    'title': entry.get('title', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'source': entry.get('source', {}).get('title', 'Unknown'),
                    'query': query,
                    'search_start': start_date,
                    'search_end': end_date
                })
            
            print(f"  ✓ Found {len(articles)} articles")
            return articles
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return []
    
    def download_historical_news(self, queries, start_date, end_date, interval_days=30):
        """
        Download historical news by breaking into smaller time periods
        """
        
        all_articles = []
        
        # Convert strings to datetime
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Break into intervals
        current = start
        period_count = 0
        
        while current < end:
            next_date = min(current + timedelta(days=interval_days), end)
            
            current_str = current.strftime("%Y-%m-%d")
            next_str = next_date.strftime("%Y-%m-%d")
            
            period_count += 1
            print(f"\n{'='*60}")
            print(f"Period {period_count}: {current_str} to {next_str}")
            print('='*60)
            
            for query in queries:
                articles = self.download_google_news_by_date_range(
                    query, current_str, next_str
                )
                all_articles.extend(articles)
                time.sleep(2)  # Be respectful with rate limiting
            
            current = next_date
        
        return all_articles


# Usage - THIS WILL WORK NOW
downloader = GoogleNewsHistoricalDownloader()

# Financial queries
queries = [
    'Apple stock',
    'Microsoft earnings',
    'Tesla',
    'S&P 500',
    'Federal Reserve'
]

print("\n" + "="*80)
print("GOOGLE NEWS HISTORICAL FINANCIAL NEWS DOWNLOADER")
print("="*80)
print(f"Downloading articles from 2024-01-01 to 2024-10-12")
print(f"Queries: {len(queries)}")
print("="*80)

# Download from Jan 2024 to now
articles = downloader.download_historical_news(
    queries=queries,
    start_date='2024-01-01',
    end_date='2024-10-12',
    interval_days=30  # Monthly chunks
)

# Save results
if articles:
    df = pd.DataFrame(articles)
    filename = 'google_news_historical_2024.csv'
    df.to_csv(filename, index=False)
    
    print(f"\n{'='*80}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Total articles downloaded: {len(articles)}")
    print(f"✓ Saved to: {filename}")
    print(f"\nTop sources:")
    print(df['source'].value_counts().head(10))
else:
    print("\n✗ No articles downloaded")