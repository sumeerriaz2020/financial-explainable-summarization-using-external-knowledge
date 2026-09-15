import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import time
from datetime import datetime

class FederalReserveDownloader:
    """
    Download Federal Reserve regulatory documents and publications
    All documents are public and free
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.base_url = "https://www.federalreserve.gov"
    
    def download_fomc_minutes(self, year):
        """
        Download FOMC (Federal Open Market Committee) meeting minutes
        These are key policy documents
        """
        
        url = f"{self.base_url}/monetarypolicy/fomccalendars.htm"
        
        print(f"Downloading FOMC minutes for {year}...")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            documents = []
            
            # Find all FOMC meeting links
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True)
                
                # Look for minutes links
                if 'minutes' in href.lower() and str(year) in href:
                    doc_url = self.base_url + href if not href.startswith('http') else href
                    
                    documents.append({
                        'type': 'FOMC Minutes',
                        'year': year,
                        'title': text,
                        'url': doc_url
                    })
            
            print(f"  ✓ Found {len(documents)} FOMC minutes")
            return documents
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return []
    
    def download_fed_speeches(self, start_year, end_year):
        """
        Download Federal Reserve speeches and testimonies
        """
        
        url = f"{self.base_url}/newsevents/speeches.htm"
        
        print(f"Downloading Fed speeches ({start_year}-{end_year})...")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            speeches = []
            
            # Find speech links
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                if '/newsevents/speech/' in href:
                    # Extract year from URL
                    try:
                        year_match = href.split('/')[-1][:4]
                        year = int(year_match)
                        
                        if start_year <= year <= end_year:
                            doc_url = self.base_url + href if not href.startswith('http') else href
                            
                            speeches.append({
                                'type': 'Speech/Testimony',
                                'year': year,
                                'title': link.get_text(strip=True),
                                'url': doc_url
                            })
                    except:
                        continue
            
            print(f"  ✓ Found {len(speeches)} speeches")
            return speeches
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return []
    
    def download_fed_reports(self):
        """
        Download Federal Reserve regulatory reports
        """
        
        reports_urls = {
            'Monetary Policy Report': f"{self.base_url}/monetarypolicy/mpr_default.htm",
            'Financial Stability Report': f"{self.base_url}/publications/financial-stability-report.htm",
            'Beige Book': f"{self.base_url}/monetarypolicy/beige-book-default.htm",
            'Bank Supervision': f"{self.base_url}/publications/supervision_regs.htm"
        }
        
        all_reports = []
        
        for report_type, url in reports_urls.items():
            print(f"Downloading {report_type}...")
            
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find PDF links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    
                    if href.endswith('.pdf'):
                        doc_url = self.base_url + href if not href.startswith('http') else href
                        
                        all_reports.append({
                            'type': report_type,
                            'title': link.get_text(strip=True),
                            'url': doc_url,
                            'format': 'PDF'
                        })
                
                time.sleep(1)
                
            except Exception as e:
                print(f"  ✗ Error with {report_type}: {e}")
        
        print(f"  ✓ Total reports found: {len(all_reports)}")
        return all_reports
    
    def download_document(self, url, save_dir='fed_documents'):
        """
        Download actual document file (PDF, HTML)
        """
        
        Path(save_dir).mkdir(exist_ok=True)
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            
            # Extract filename from URL
            filename = url.split('/')[-1]
            if not filename:
                filename = f"document_{int(time.time())}.pdf"
            
            filepath = Path(save_dir) / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"    ✓ Downloaded: {filename}")
            return str(filepath)
            
        except Exception as e:
            print(f"    ✗ Error downloading {url}: {e}")
            return None


# Usage Example
def download_all_fed_documents():
    """
    Download comprehensive Federal Reserve documents
    """
    
    downloader = FederalReserveDownloader()
    
    print("="*80)
    print("FEDERAL RESERVE DOCUMENT DOWNLOADER")
    print("="*80)
    
    all_documents = []
    
    # 1. FOMC Minutes (2020-2024)
    print("\n1. FOMC Meeting Minutes:")
    print("-"*80)
    for year in range(2020, 2025):
        docs = downloader.download_fomc_minutes(year)
        all_documents.extend(docs)
    
    # 2. Fed Speeches
    print("\n2. Federal Reserve Speeches:")
    print("-"*80)
    speeches = downloader.download_fed_speeches(2020, 2024)
    all_documents.extend(speeches)
    
    # 3. Fed Reports
    print("\n3. Federal Reserve Reports:")
    print("-"*80)
    reports = downloader.download_fed_reports()
    all_documents.extend(reports)
    
    # Save index
    if all_documents:
        df = pd.DataFrame(all_documents)
        df.to_csv('fed_documents_index.csv', index=False)
        
        print(f"\n{'='*80}")
        print("FEDERAL RESERVE DOCUMENTS INDEX")
        print(f"{'='*80}")
        print(f"✓ Total documents found: {len(all_documents)}")
        print(f"✓ Index saved to: fed_documents_index.csv")
        
        print("\nBreakdown by type:")
        print(df['type'].value_counts())
        
        # Option to download actual files
        download_files = input("\nDownload actual document files? (yes/no): ")
        
        if download_files.lower() == 'yes':
            print("\nDownloading files...")
            for i, doc in enumerate(all_documents[:10], 1):  # Download first 10 as example
                print(f"[{i}/10] {doc['title'][:50]}...")
                downloader.download_document(doc['url'])
                time.sleep(1)
        
        return df
    
    return None


# Run it
fed_docs = download_all_fed_documents()
