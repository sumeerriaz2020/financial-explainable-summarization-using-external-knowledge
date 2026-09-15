import requests
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime
import pandas as pd
from pathlib import Path
import time

class FreeFinancialNewsDownloader:
    """
    Download financial news from FREE sources
    All of these are legal and don't require subscriptions
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def download_yahoo_finance_news(self, ticker):
        """Yahoo Finance - Completely free"""
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = []
            # Find news items
            news_items = soup.find_all('h3')
            
            for item in news_items[:20]:
                link = item.find('a')
                if link:
                    articles.append({
                        'title': link.get_text(strip=True),
                        'url': link.get('href', ''),
                        'source': 'Yahoo Finance'
                    })
            
            return articles
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def download_marketwatch_news(self, ticker):
        """MarketWatch - Free access"""
        url = f"https://www.marketwatch.com/investing/stock/{ticker}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = []
            headlines = soup.find_all('a', {'class': 'link'})
            
            for headline in headlines[:20]:
                if headline.get_text(strip=True):
                    articles.append({
                        'title': headline.get_text(strip=True),
                        'url': headline.get('href', ''),
                        'source': 'MarketWatch'
                    })
            
            return articles
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def download_cnbc_rss(self):
        """CNBC RSS Feeds - Free"""
        feeds = {
            'top_news': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'investing': 'https://www.cnbc.com/id/15839135/device/rss/rss.html',
            'earnings': 'https://www.cnbc.com/id/15839069/device/rss/rss.html',
        }
        
        all_articles = []
        
        for category, feed_url in feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:20]:
                    all_articles.append({
                        'title': entry.get('title', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'summary': entry.get('summary', ''),
                        'source': 'CNBC',
                        'category': category
                    })
                
                time.sleep(1)  # Be respectful
                
            except Exception as e:
                print(f"Error with {category}: {e}")
        
        return all_articles
    
    def download_reuters_rss(self):
        """Reuters RSS - Free headlines"""
        feeds = {
            'business': 'https://www.reuters.com/business',
            'markets': 'https://www.reuters.com/markets',
        }
        
        articles = []
        
        for category, url in feeds.items():
            try:
                # Reuters has changed their RSS structure
                # This is a basic scraper for headlines
                response = requests.get(url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article headlines
                headlines = soup.find_all('a', {'data-testid': 'Link'})
                
                for headline in headlines[:20]:
                    title = headline.get_text(strip=True)
                    if title and len(title) > 20:  # Filter out short/navigation links
                        articles.append({
                            'title': title,
                            'url': 'https://www.reuters.com' + headline.get('href', ''),
                            'source': 'Reuters',
                            'category': category
                        })
                
                time.sleep(2)
                
            except Exception as e:
                print(f"Error with Reuters {category}: {e}")
        
        return articles
    
    def download_seeking_alpha_free(self, ticker):
        """Seeking Alpha - Free articles (limited)"""
        url = f"https://seekingalpha.com/symbol/{ticker}/news"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = []
            # Note: Seeking Alpha structure changes frequently
            links = soup.find_all('a', href=True)
            
            for link in links:
                if '/article/' in link.get('href', ''):
                    articles.append({
                        'title': link.get_text(strip=True),
                        'url': 'https://seekingalpha.com' + link['href'],
                        'source': 'Seeking Alpha'
                    })
            
            return articles[:20]
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def download_finviz_news(self, ticker):
        """Finviz - Free news aggregator"""
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = []
            news_table = soup.find('table', {'class': 'fullview-news-outer'})
            
            if news_table:
                rows = news_table.find_all('tr')
                
                for row in rows:
                    link = row.find('a')
                    date = row.find('td', {'align': 'right'})
                    
                    if link:
                        articles.append({
                            'title': link.get_text(strip=True),
                            'url': link.get('href', ''),
                            'date': date.get_text(strip=True) if date else '',
                            'source': 'Finviz Aggregator'
                        })
            
            return articles
        except Exception as e:
            print(f"Error: {e}")
            return []


def main():
    """Download financial news from free sources"""
    
    downloader = FreeFinancialNewsDownloader()
    
    print("\n" + "="*80)
    print("FREE FINANCIAL NEWS DOWNLOADER")
    print("="*80)
    
    ticker = 'AAPL'  # Example
    
    # 1. Yahoo Finance
    print(f"\n1. Downloading from Yahoo Finance ({ticker})...")
    yahoo_news = downloader.download_yahoo_finance_news(ticker)
    print(f"   ✓ Found {len(yahoo_news)} articles")
    
    # 2. MarketWatch
    print(f"\n2. Downloading from MarketWatch ({ticker})...")
    mw_news = downloader.download_marketwatch_news(ticker)
    print(f"   ✓ Found {len(mw_news)} articles")
    
    # 3. CNBC RSS
    print(f"\n3. Downloading from CNBC RSS feeds...")
    cnbc_news = downloader.download_cnbc_rss()
    print(f"   ✓ Found {len(cnbc_news)} articles")
    
    # 4. Reuters
    print(f"\n4. Downloading from Reuters...")
    reuters_news = downloader.download_reuters_rss()
    print(f"   ✓ Found {len(reuters_news)} articles")
    
    # 5. Finviz
    print(f"\n5. Downloading from Finviz ({ticker})...")
    finviz_news = downloader.download_finviz_news(ticker)
    print(f"   ✓ Found {len(finviz_news)} articles")
    
    # Combine all
    all_news = yahoo_news + mw_news + cnbc_news + reuters_news + finviz_news
    
    # Save to CSV
    if all_news:
        df = pd.DataFrame(all_news)
        df.to_csv('financial_news_free.csv', index=False)
        print(f"\n✓ Total articles collected: {len(all_news)}")
        print(f"✓ Saved to: financial_news_free.csv")


if __name__ == "__main__":
    main()