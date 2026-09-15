import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path

class EarningsReleaseDownloader:
    """
    Download earnings PRESS RELEASES from SEC (not transcripts)
    These are the official financial results, but no Q&A
    """
    
    def __init__(self, name, email):
        self.headers = {
            'User-Agent': f'{name} {email}',
            'Accept-Encoding': 'gzip, deflate'
        }
    
    def get_company_cik(self, ticker):
        """Get CIK from ticker"""
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers=self.headers)
        time.sleep(0.11)
        
        data = response.json()
        for company in data.values():
            if company['ticker'].upper() == ticker.upper():
                return str(company['cik_str']).zfill(10)
        return None
    
    def download_earnings_releases(self, ticker):
        """Download earnings press releases from 8-K filings"""
        print(f"\nDownloading earnings releases for {ticker}...")
        
        cik = self.get_company_cik(ticker)
        if not cik:
            print(f"✗ CIK not found for {ticker}")
            return []
        
        # Get 8-K filings
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        response = requests.get(url, headers=self.headers)
        time.sleep(0.11)
        
        if response.status_code != 200:
            print(f"✗ Error: {response.status_code}")
            return []
        
        data = response.json()
        recent = data.get('filings', {}).get('recent', {})
        
        releases = []
        for i in range(len(recent.get('form', []))):
            if recent['form'][i] == '8-K':
                filing = {
                    'date': recent['filingDate'][i],
                    'accession': recent['accessionNumber'][i]
                }
                
                # Download the filing
                acc_no_dashes = filing['accession'].replace('-', '')
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dashes}/{filing['accession']}.txt"
                
                file_response = requests.get(filing_url, headers=self.headers)
                time.sleep(0.11)
                
                if file_response.status_code == 200:
                    # Check if it's an earnings release
                    content = file_response.text
                    if any(keyword in content.lower() for keyword in 
                          ['earnings', 'quarterly results', 'financial results']):
                        
                        # Save it
                        filename = f"earnings_releases/{ticker}_{filing['date']}.txt"
                        Path(filename).parent.mkdir(exist_ok=True)
                        
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        releases.append(filing)
                        print(f"  ✓ Downloaded: {filing['date']}")
                
                if len(releases) >= 20:  # Limit to recent 20
                    break
        
        return releases


# This WILL work
downloader = EarningsReleaseDownloader("Student", "jrapplicationa@gmail.com")

# Test with a few companies
test_tickers = ['AAPL', 'MSFT', 'GOOGL']

for ticker in test_tickers:
    releases = downloader.download_earnings_releases(ticker)
    print(f"✓ Downloaded {len(releases)} earnings releases for {ticker}")