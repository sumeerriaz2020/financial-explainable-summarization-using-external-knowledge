import requests
import pandas as pd
from pathlib import Path
import time
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin

class EarningsTranscriptDownloader:
    
    def __init__(self, company_name, email):
        """
        SEC REQUIRES proper User-Agent with your identity
        Format: "Company/Name email@example.com"
        """
        self.headers = {
            'User-Agent': f'{company_name} {email}',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        self.base_url = "https://www.sec.gov"
        self.data_url = "https://data.sec.gov"
        
    def get_sp500_tickers(self):
        """Get list of S&P 500 companies from Wikipedia"""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            # Different headers for Wikipedia
            wiki_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            tables = pd.read_html(url, header=0)
            sp500 = tables[0]
            return sp500[['Symbol', 'Security', 'CIK']].to_dict('records')
        except Exception as e:
            print(f"Error getting S&P 500 list: {e}")
            return []
    
    def get_company_cik(self, ticker):
        """Get CIK number for a ticker using SEC API"""
        try:
            # Use the SEC's company tickers JSON
            url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(url, headers=self.headers, timeout=10)
            time.sleep(0.11)  # Rate limiting: max 10 requests/second
            
            if response.status_code == 200:
                data = response.json()
                
                # Search for ticker
                for key, company in data.items():
                    if company['ticker'].upper() == ticker.upper():
                        cik = str(company['cik_str']).zfill(10)
                        return cik
            
            return None
        except Exception as e:
            print(f"Error getting CIK for {ticker}: {e}")
            return None
    
    def get_company_filings(self, cik, form_type='8-K', count=100):
        """Get recent filings for a company"""
        try:
            url = f"{self.data_url}/submissions/CIK{cik}.json"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            time.sleep(0.11)  # Rate limiting
            
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                return []
            
            data = response.json()
            
            # Get recent filings
            recent = data.get('filings', {}).get('recent', {})
            
            filings = []
            for i in range(len(recent.get('form', []))):
                if recent['form'][i] == form_type:
                    accession = recent['accessionNumber'][i]
                    filing_date = recent['filingDate'][i]
                    
                    filings.append({
                        'accessionNumber': accession,
                        'filingDate': filing_date,
                        'form': recent['form'][i],
                        'primaryDocument': recent.get('primaryDocument', [''])[i]
                    })
                    
                    if len(filings) >= count:
                        break
            
            return filings
            
        except Exception as e:
            print(f"Error getting filings for CIK {cik}: {e}")
            return []
    
    def get_filing_content(self, cik, accession_number):
        """Download the actual filing content"""
        try:
            # Format: Remove dashes from accession number for URL
            acc_no_dashes = accession_number.replace('-', '')
            
            # Try to get the filing index page
            url = f"{self.base_url}/cgi-bin/viewer?action=view&cik={cik}&accession_number={accession_number}&xbrl_type=v"
            
            response = requests.get(url, headers=self.headers, timeout=15)
            time.sleep(0.11)
            
            if response.status_code == 200:
                return response.text
            else:
                # Try alternative URL format
                url = f"{self.base_url}/Archives/edgar/data/{cik}/{acc_no_dashes}/{accession_number}.txt"
                response = requests.get(url, headers=self.headers, timeout=15)
                time.sleep(0.11)
                
                if response.status_code == 200:
                    return response.text
            
            return None
            
        except Exception as e:
            print(f"Error downloading filing: {e}")
            return None
    
    def check_for_transcript(self, content):
        """Check if filing contains earnings call transcript"""
        if not content:
            return False
        
        content_lower = content.lower()
        
        # Keywords indicating earnings call transcript
        keywords = [
            'earnings call transcript',
            'conference call transcript',
            'prepared remarks',
            'earnings conference call',
            'quarterly results conference call',
            'q&a session',
            'operator:',  # Common in transcripts
            'presentation transcript'
        ]
        
        # Count keyword matches
        matches = sum(1 for keyword in keywords if keyword in content_lower)
        
        # If multiple keywords found, likely a transcript
        return matches >= 2
    
    def save_transcript(self, ticker, filing_date, content, output_dir='transcripts'):
        """Save transcript to file"""
        Path(output_dir).mkdir(exist_ok=True)
        
        filename = f"{output_dir}/{ticker}_{filing_date}_transcript.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filename


def main():
    # IMPORTANT: Replace with your information
    COMPANY_NAME = "Student"  # Your name or organization
    EMAIL = "jrapplicationa@gmail.com"   # Your email
    
    print("="*60)
    print("SEC Earnings Call Transcript Downloader")
    print("="*60)
    
    downloader = EarningsTranscriptDownloader(COMPANY_NAME, EMAIL)
    
    # Option 1: Get S&P 500 list
    print("\nFetching S&P 500 company list...")
    sp500_companies = downloader.get_sp500_tickers()
    
    if not sp500_companies:
        print("Failed to get S&P 500 list. Using manual list.")
        # Fallback: manual list of popular tickers
        sp500_companies = [
            {'Symbol': 'AAPL', 'Security': 'Apple Inc.', 'CIK': '0000320193'},
            {'Symbol': 'MSFT', 'Security': 'Microsoft Corporation', 'CIK': '0000789019'},
            {'Symbol': 'GOOGL', 'Security': 'Alphabet Inc.', 'CIK': '0001652044'},
            {'Symbol': 'AMZN', 'Security': 'Amazon.com Inc.', 'CIK': '0001018724'},
            {'Symbol': 'NVDA', 'Security': 'NVIDIA Corporation', 'CIK': '0001045810'},
        ]
    
    print(f"Found {len(sp500_companies)} companies")
    
    # Process companies
    results = []
    transcripts_found = 0
    
    # Limit to first 10 companies for testing
    for i, company in enumerate(sp500_companies[:10], 1):
        ticker = company['Symbol']
        company_name = company['Security']
        
        print(f"\n[{i}/10] Processing {ticker} - {company_name}")
        
        # Get CIK
        if 'CIK' in company and company['CIK']:
            cik = str(company['CIK']).zfill(10)
        else:
            cik = downloader.get_company_cik(ticker)
        
        if not cik:
            print(f"  ✗ Could not find CIK for {ticker}")
            continue
        
        print(f"  CIK: {cik}")
        
        # Get recent 8-K filings
        filings = downloader.get_company_filings(cik, form_type='8-K', count=20)
        print(f"  Found {len(filings)} recent 8-K filings")
        
        # Check each filing for transcripts
        for filing in filings:
            filing_date = filing['filingDate']
            accession = filing['accessionNumber']
            
            print(f"    Checking {filing_date}...", end='')
            
            content = downloader.get_filing_content(cik, accession)
            
            if content and downloader.check_for_transcript(content):
                print(" ✓ TRANSCRIPT FOUND!")
                
                filename = downloader.save_transcript(ticker, filing_date, content)
                transcripts_found += 1
                
                results.append({
                    'ticker': ticker,
                    'company': company_name,
                    'date': filing_date,
                    'accession': accession,
                    'filename': filename
                })
            else:
                print(" ✗")
            
            # Rate limiting is crucial
            time.sleep(0.11)
    
    # Save summary
    if results:
        df = pd.DataFrame(results)
        df.to_csv('transcripts_found.csv', index=False)
        
        print(f"\n{'='*60}")
        print("DOWNLOAD COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Transcripts found: {transcripts_found}")
        print(f"📄 Summary saved to: transcripts_found.csv")
        print(f"📁 Transcripts saved to: transcripts/")
    else:
        print("\n✗ No transcripts found in the filings checked.")
        print("Note: Not all companies include transcripts in 8-K filings.")
        print("Consider using Financial Modeling Prep API for better coverage.")


if __name__ == "__main__":
    main()