import requests
import pandas as pd
from pathlib import Path
import time
from datetime import datetime

class SECBulkDownloader:
    def __init__(self, company_name, email):
        # SEC requires User-Agent with identity
        self.headers = {
            'User-Agent': f'{company_name} {email}'
        }
        self.base_url = "https://www.sec.gov/Archives"
        
    def download_quarter_index(self, year, quarter):
        """Download master index for a specific quarter"""
        url = f"{self.base_url}/edgar/full-index/{year}/QTR{quarter}/master.idx"
        print(f"Downloading index: {year} Q{quarter}")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            time.sleep(0.11)  # Rate limiting: ~9 requests/second
            return response.text
        except Exception as e:
            print(f"Error downloading index for {year} Q{quarter}: {e}")
            return None
    
    def parse_index(self, index_text):
        """Parse index file into DataFrame"""
        if not index_text:
            return pd.DataFrame()
        
        lines = index_text.split('\n')
        
        # Find where data starts (after the header)
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('CIK|Company Name|Form Type|Date Filed|Filename'):
                data_start = i + 1
                break
        
        records = []
        for line in lines[data_start:]:
            if not line.strip():
                continue
                
            parts = line.split('|')
            if len(parts) >= 5:
                records.append({
                    'CIK': parts[0].strip(),
                    'Company': parts[1].strip(),
                    'Form': parts[2].strip(),
                    'Date': parts[3].strip(),
                    'Filename': parts[4].strip()
                })
        
        return pd.DataFrame(records)
    
    def download_filing(self, filename, save_dir):
        """Download individual filing"""
        url = f"{self.base_url}/{filename}"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            time.sleep(0.11)  # Rate limiting
            
            # Create directory structure
            filepath = Path(save_dir) / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Save file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return False

def main():
    # CONFIGURE THESE
    company_name = "Student"  # Replace with your company/name
    email = "jrapplicationa@gmail.com"  # Replace with your email
    save_directory = "sec_filings"  # Where to save files
    
    # Initialize downloader
    downloader = SECBulkDownloader(company_name, email)
    
    # Step 1: Download and parse all indices
    print("=" * 60)
    print("STEP 1: Downloading SEC Master Indices (2023-2024)")
    # print("STEP 1: Downloading SEC Master Indices (2015-2024)")
    print("=" * 60)
    
    all_filings = []
    
    for year in range(2023, 2025):
    # for year in range(2015, 2025):
        for quarter in range(1, 5):
            # Skip future quarters
            if year == 2024 and quarter > 4:
                continue
                
            index_text = downloader.download_quarter_index(year, quarter)
            
            if index_text:
                df = downloader.parse_index(index_text)
                
                # Filter for only 10-K, 10-Q, and 8-K
                df_filtered = df[df['Form'].isin(['10-K', '10-Q', '8-K', '10-K/A', '10-Q/A', '8-K/A'])]
                
                print(f"  Found {len(df_filtered)} relevant filings in {year} Q{quarter}")
                all_filings.append(df_filtered)
    
    # Combine all filings
    if not all_filings:
        print("No filings found!")
        return
    
    all_filings_df = pd.concat(all_filings, ignore_index=True)
    print(f"\n✓ Total filings found: {len(all_filings_df):,}")
    
    # Save index to CSV
    index_file = 'sec_filings_index_2023_2024.csv'
    # index_file = 'sec_filings_index_2015_2024.csv'
    all_filings_df.to_csv(index_file, index=False)
    print(f"✓ Index saved to: {index_file}")
    
    # Show breakdown
    print("\nFilings breakdown:")
    print(all_filings_df['Form'].value_counts())
    
    # Step 2: Download actual filings
    print("\n" + "=" * 60)
    print("STEP 2: Downloading Actual Filings")
    print("=" * 60)
    print(f"This will download {len(all_filings_df):,} documents")
    
    # Ask for confirmation
    response = input("\nDo you want to proceed with downloading? (yes/no): ")
    if response.lower() != 'yes':
        print("Download cancelled. Index file saved for later use.")
        return
    
    # Download filings
    successful = 0
    failed = 0
    
    for idx, row in all_filings_df.iterrows():
        if idx % 100 == 0:
            print(f"Progress: {idx:,}/{len(all_filings_df):,} ({idx/len(all_filings_df)*100:.1f}%) - Success: {successful}, Failed: {failed}")
        
        if downloader.download_filing(row['Filename'], save_directory):
            successful += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"✓ Successfully downloaded: {successful:,}")
    print(f"✗ Failed: {failed:,}")
    print(f"📁 Files saved to: {Path(save_directory).absolute()}")

if __name__ == "__main__":
    main()