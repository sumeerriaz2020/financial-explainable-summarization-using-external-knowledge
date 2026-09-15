import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import time

class SECRegulatoryDownloader:
    """
    Download SEC rules, releases, and guidance documents
    """
    
    def __init__(self, name, email):
        self.headers = {
            'User-Agent': f'{name} {email}',
            'Accept-Encoding': 'gzip, deflate'
        }
        self.base_url = "https://www.sec.gov"
    
    def download_sec_rules(self, start_year=2020, end_year=2024):
        """
        Download SEC final rules and proposed rules
        """
        
        url = f"{self.base_url}/rules/final.shtml"
        
        print(f"Downloading SEC rules ({start_year}-{end_year})...")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            time.sleep(0.15)
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            rules = []
            
            # Find all rule links
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True)
                
                # Look for rule releases
                if '/rules/final/' in href or 'rules/proposed/' in href:
                    # Try to extract year
                    try:
                        year_match = href.split('/')[-1].split('-')[0]
                        if year_match.isdigit() and len(year_match) == 4:
                            year = int(year_match)
                            
                            if start_year <= year <= end_year:
                                doc_url = self.base_url + href if not href.startswith('http') else href
                                
                                rule_type = 'Final Rule' if '/final/' in href else 'Proposed Rule'
                                
                                rules.append({
                                    'type': rule_type,
                                    'year': year,
                                    'title': text,
                                    'url': doc_url
                                })
                    except:
                        continue
            
            print(f"  ✓ Found {len(rules)} rules")
            return rules
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return []
    
    def download_sec_releases(self, release_type='litigation'):
        """
        Download SEC releases
        Types: litigation, admin, alj, opinions, other
        """
        
        url = f"{self.base_url}/litigation/litreleases.shtml"
        
        print(f"Downloading SEC {release_type} releases...")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            time.sleep(0.15)
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            releases = []
            
            # Find release links
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                if '/litigation/' in href and href.endswith('.htm'):
                    doc_url = self.base_url + href if not href.startswith('http') else href
                    
                    releases.append({
                        'type': f'SEC {release_type.title()} Release',
                        'title': link.get_text(strip=True),
                        'url': doc_url
                    })
            
            print(f"  ✓ Found {len(releases)} releases")
            return releases
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return []
    
    def download_sec_guidance(self):
        """
        Download SEC staff guidance and interpretations
        """
        
        guidance_urls = {
            'Compliance and Disclosure Interpretations': f"{self.base_url}/divisions/corpfin/guidance.shtml",
            'Staff Accounting Bulletins': f"{self.base_url}/interps/account.shtml",
            'Staff Legal Bulletins': f"{self.base_url}/interps/legal.shtml"
        }
        
        all_guidance = []
        
        for guidance_type, url in guidance_urls.items():
            print(f"Downloading {guidance_type}...")
            
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                time.sleep(0.15)
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find guidance documents
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    
                    if href.endswith('.htm') or href.endswith('.pdf'):
                        doc_url = self.base_url + href if not href.startswith('http') else href
                        
                        all_guidance.append({
                            'type': guidance_type,
                            'title': link.get_text(strip=True),
                            'url': doc_url
                        })
                
            except Exception as e:
                print(f"  ✗ Error with {guidance_type}: {e}")
        
        print(f"  ✓ Total guidance documents: {len(all_guidance)}")
        return all_guidance
    
    def download_investor_bulletins(self):
        """
        Download SEC Investor Bulletins and Alerts
        """
        
        url = f"{self.base_url}/oiea/investor-alerts-bulletins.html"
        
        print("Downloading SEC Investor Bulletins...")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            time.sleep(0.15)
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            bulletins = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                if '/oiea/' in href or '/investor/' in href:
                    doc_url = self.base_url + href if not href.startswith('http') else href
                    
                    bulletins.append({
                        'type': 'Investor Bulletin',
                        'title': link.get_text(strip=True),
                        'url': doc_url
                    })
            
            print(f"  ✓ Found {len(bulletins)} bulletins")
            return bulletins
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return []


# Usage
def download_all_sec_regulatory_docs():
    """
    Download comprehensive SEC regulatory documents
    """
    
    downloader = SECRegulatoryDownloader("YourName", "your.email@example.com")
    
    print("="*80)
    print("SEC REGULATORY DOCUMENT DOWNLOADER")
    print("="*80)
    
    all_documents = []
    
    # 1. SEC Rules
    print("\n1. SEC Rules and Regulations:")
    print("-"*80)
    rules = downloader.download_sec_rules(2020, 2024)
    all_documents.extend(rules)
    
    # 2. SEC Releases
    print("\n2. SEC Litigation Releases:")
    print("-"*80)
    releases = downloader.download_sec_releases('litigation')
    all_documents.extend(releases)
    
    # 3. SEC Guidance
    print("\n3. SEC Staff Guidance:")
    print("-"*80)
    guidance = downloader.download_sec_guidance()
    all_documents.extend(guidance)
    
    # 4. Investor Bulletins
    print("\n4. SEC Investor Bulletins:")
    print("-"*80)
    bulletins = downloader.download_investor_bulletins()
    all_documents.extend(bulletins)
    
    # Save index
    if all_documents:
        df = pd.DataFrame(all_documents)
        df.to_csv('sec_regulatory_documents_index.csv', index=False)
        
        print(f"\n{'='*80}")
        print("SEC REGULATORY DOCUMENTS INDEX")
        print(f"{'='*80}")
        print(f"✓ Total documents found: {len(all_documents)}")
        print(f"✓ Index saved to: sec_regulatory_documents_index.csv")
        
        print("\nBreakdown by type:")
        print(df['type'].value_counts())
        
        return df
    
    return None


# Run it
sec_docs = download_all_sec_regulatory_docs()