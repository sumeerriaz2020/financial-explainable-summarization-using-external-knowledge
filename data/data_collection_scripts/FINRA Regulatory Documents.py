import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import time

class FINRADownloader:
    """
    Download FINRA rules, notices, and guidance
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.base_url = "https://www.finra.org"
    
    def download_finra_rules(self):
        """
        Download FINRA Rulebook
        """
        
        url = f"{self.base_url}/rules-guidance/rulebooks/finra-rules"
        
        print("Downloading FINRA Rules...")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            rules = []
            
            # Find rule links
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True)
                
                if '/rules-guidance/rulebooks/' in href:
                    doc_url = self.base_url + href if not href.startswith('http') else href
                    
                    rules.append({
                        'type': 'FINRA Rule',
                        'title': text,
                        'url': doc_url
                    })
            
            print(f"  ✓ Found {len(rules)} rules")
            return rules
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return []
    
    def download_regulatory_notices(self, start_year=2020, end_year=2024):
        """
        Download FINRA Regulatory Notices
        """
        
        url = f"{self.base_url}/rules-guidance/notices"
        
        print(f"Downloading FINRA Regulatory Notices ({start_year}-{end_year})...")
        
        try:
            notices = []
            
            for year in range(start_year, end_year + 1):
                year_url = f"{url}/{year}"
                
                response = requests.get(year_url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find notice links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    
                    if '/rules-guidance/notices/' in href and str(year) in href:
                        doc_url = self.base_url + href if not href.startswith('http') else href
                        
                        notices.append({
                            'type': 'Regulatory Notice',
                            'year': year,
                            'title': link.get_text(strip=True),
                            'url': doc_url
                        })
                
                time.sleep(1)
            
            print(f"  ✓ Found {len(notices)} notices")
            return notices
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return []
    
    def download_guidance_documents(self):
        """
        Download FINRA guidance and FAQs
        """
        
        guidance_urls = {
            'Key Topics': f"{self.base_url}/rules-guidance/key-topics",
            'Guidance': f"{self.base_url}/rules-guidance/guidance",
        }
        
        all_guidance = []
        
        for guidance_type, url in guidance_urls.items():
            print(f"Downloading FINRA {guidance_type}...")
            
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    
                    if '/rules-guidance/' in href:
                        doc_url = self.base_url + href if not href.startswith('http') else href
                        
                        all_guidance.append({
                            'type': f'FINRA {guidance_type}',
                            'title': link.get_text(strip=True),
                            'url': doc_url
                        })
                
                time.sleep(1)
                
            except Exception as e:
                print(f"  ✗ Error with {guidance_type}: {e}")
        
        print(f"  ✓ Total guidance documents: {len(all_guidance)}")
        return all_guidance
    
    def download_disciplinary_actions(self):
        """
        Download FINRA disciplinary actions database
        """
        
        url = f"{self.base_url}/rules-guidance/oversight-enforcement/finra-disciplinary-actions"
        
        print("Downloading FINRA Disciplinary Actions...")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            actions = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                if 'disciplinary-actions' in href:
                    doc_url = self.base_url + href if not href.startswith('http') else href
                    
                    actions.append({
                        'type': 'Disciplinary Action',
                        'title': link.get_text(strip=True),
                        'url': doc_url
                    })
            
            print(f"  ✓ Found {len(actions)} actions")
            return actions
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return []


# Usage
def download_all_finra_documents():
    """
    Download comprehensive FINRA regulatory documents
    """
    
    downloader = FINRADownloader()
    
    print("="*80)
    print("FINRA REGULATORY DOCUMENT DOWNLOADER")
    print("="*80)
    
    all_documents = []
    
    # 1. FINRA Rules
    print("\n1. FINRA Rules:")
    print("-"*80)
    rules = downloader.download_finra_rules()
    all_documents.extend(rules)
    
    # 2. Regulatory Notices
    print("\n2. FINRA Regulatory Notices:")
    print("-"*80)
    notices = downloader.download_regulatory_notices(2020, 2024)
    all_documents.extend(notices)
    
    # 3. Guidance Documents
    print("\n3. FINRA Guidance:")
    print("-"*80)
    guidance = downloader.download_guidance_documents()
    all_documents.extend(guidance)
    
    # 4. Disciplinary Actions
    print("\n4. FINRA Disciplinary Actions:")
    print("-"*80)
    actions = downloader.download_disciplinary_actions()
    all_documents.extend(actions)
    
    # Save index
    if all_documents:
        df = pd.DataFrame(all_documents)
        df.to_csv('finra_regulatory_documents_index.csv', index=False)
        
        print(f"\n{'='*80}")
        print("FINRA REGULATORY DOCUMENTS INDEX")
        print(f"{'='*80}")
        print(f"✓ Total documents found: {len(all_documents)}")
        print(f"✓ Index saved to: finra_regulatory_documents_index.csv")
        
        print("\nBreakdown by type:")
        print(df['type'].value_counts())
        
        return df
    
    return None


# Run it
finra_docs = download_all_finra_documents()