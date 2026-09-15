import feedparser
import pandas as pd
from datetime import datetime, timedelta
import time
import requests
from urllib.parse import urlencode

class ImprovedGoogleNewsDownloader:
    """
    Even better approach using requests to build URLs properly
    """
    
    def __init__(self):
        self.base_url = "https://news.google.com/rss/search"
    
    def build_google_news_url(self, query, start_date, end_date):
        """
        Build properly formatted Google News URL
        """
        # Create the search query
        search_query = f"{query} after:{start_date} before:{end_date}"
        
        # Build parameters dictionary
        params = {
            'q': search_query,
            'hl': 'en-US',
            'gl': 'US',
            'ceid': 'US:en'
        }
        
        # Use urlencode to properly format the URL
        url = f"{self.base_url}?{urlencode(params)}"
        
        return url
    
    def download_news(self, query, start_date, end_date):
        """
        Download news articles for a query and date range
        """
        url = self.build_google_news_url(query, start_date, end_date)
        
        print(f"  Downloading: {query} ({start_date} to {end_date})")
        
        try:
            # Parse the RSS feed
            feed = feedparser.parse(url)
            
            articles = []
            for entry in feed.entries:
                # Clean up the Google News redirect URL
                link = entry.get('link', '')
                
                articles.append({
                    'date_range_start': start_date,
                    'date_range_end': end_date,
                    'query': query,
                    'title': entry.get('title', ''),
                    'link': link,
                    'published': entry.get('published', ''),
                    'source': entry.get('source', {}).get('title', 'Unknown'),
                    'summary': entry.get('summary', '')[:200]  # First 200 chars
                })
            
            print(f"    ✓ Found {len(articles)} articles")
            return articles
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return []
    
    def download_bulk_historical(self, queries, start_date_str, end_date_str, days_per_period=30):
        """
        Download historical news in bulk
        """
        
        all_articles = []
        
        # Parse dates
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        
        # Calculate total periods
        total_days = (end_date - start_date).days
        total_periods = (total_days // days_per_period) + 1
        
        print(f"\nTotal periods to download: {total_periods}")
        print(f"Queries per period: {len(queries)}")
        print(f"Estimated total requests: {total_periods * len(queries)}")
        print(f"Estimated time: {(total_periods * len(queries) * 2) / 60:.1f} minutes\n")
        
        # Break into periods
        current_date = start_date
        period_num = 0
        
        while current_date < end_date:
            next_date = min(current_date + timedelta(days=days_per_period), end_date)
            
            period_num += 1
            current_str = current_date.strftime("%Y-%m-%d")
            next_str = next_date.strftime("%Y-%m-%d")
            
            print(f"\n[Period {period_num}/{total_periods}] {current_str} to {next_str}")
            print("-" * 60)
            
            period_articles = []
            
            for query in queries:
                articles = self.download_news(query, current_str, next_str)
                period_articles.extend(articles)
                all_articles.extend(articles)
                time.sleep(2)  # Rate limiting
            
            print(f"  Period total: {len(period_articles)} articles")
            print(f"  Running total: {len(all_articles)} articles")
            
            current_date = next_date
        
        return all_articles


# COMPLETE WORKING EXAMPLE
def main():
    """
    Complete working example - downloads 2024 financial news
    """
    
    print("="*80)
    print("GOOGLE NEWS HISTORICAL DOWNLOADER - FIXED VERSION")
    print("="*80)
    
    downloader = ImprovedGoogleNewsDownloader()
    
    # Define your queries
    queries = [
        'Apple stock',
        'Microsoft stock',
        'Amazon stock',
        'Tesla stock',
        'NVIDIA stock',
        'Google stock',
        'Meta stock',
        'S&P 500',
        'Dow Jones',
        'NASDAQ',
        'Federal Reserve',
        'interest rates',
        'inflation',
        'stock market',
        'earnings report'
    ]
    
    # Download historical news
    articles = downloader.download_bulk_historical(
        queries=queries,
        start_date_str='2024-01-01',
        end_date_str='2024-10-12',
        days_per_period=30  # Monthly periods
    )
    
    # Save to CSV
    if articles:
        df = pd.DataFrame(articles)
        filename = 'financial_news_2024_complete.csv'
        df.to_csv(filename, index=False)
        
        print(f"\n{'='*80}")
        print("DOWNLOAD COMPLETE!")
        print(f"{'='*80}")
        print(f"✓ Total articles: {len(articles)}")
        print(f"✓ Date range: 2024-01-01 to 2024-10-12")
        print(f"✓ Saved to: {filename}")
        
        print(f"\n📊 STATISTICS:")
        print(f"{'='*80}")
        print(f"Unique sources: {df['source'].nunique()}")
        print(f"\nTop 10 sources:")
        print(df['source'].value_counts().head(10))
        
        print(f"\nArticles by query:")
        print(df['query'].value_counts())
        
        print(f"\nDate range breakdown:")
        print(df['date_range_start'].value_counts().sort_index())
        
        return df
    else:
        print("\n✗ No articles downloaded. Check your internet connection.")
        return None


if __name__ == "__main__":
    df = main()