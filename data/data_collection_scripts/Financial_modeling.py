import requests
import json
from datetime import datetime

class FMPTranscriptTester:
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api"
    
    def test_api_key(self):
        """Test if API key is valid"""
        url = f"{self.base_url}/v3/profile/AAPL"
        params = {'apikey': self.api_key}
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            print("✓ API Key is valid!")
            return True
        else:
            print(f"✗ API Key issue: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    
    def check_transcript_availability(self, ticker):
        """Check what transcripts are available for a ticker"""
        
        # Try different endpoints
        endpoints = [
            f"/v3/earning_call_transcript/{ticker}",  # Latest
            f"/v4/batch_earning_call_transcript/{ticker}",  # Batch (v4)
            f"/v3/transcripts/{ticker}",  # Alternative
        ]
        
        print(f"\nChecking transcript availability for {ticker}:")
        print("="*60)
        
        for endpoint in endpoints:
            url = f"{self.base_url}{endpoint}"
            params = {'apikey': self.api_key}
            
            print(f"\nTrying endpoint: {endpoint}")
            response = requests.get(url, params=params)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data:
                        print(f"✓ Found data! Type: {type(data)}")
                        if isinstance(data, list) and len(data) > 0:
                            print(f"  Number of transcripts: {len(data)}")
                            if isinstance(data[0], dict):
                                print(f"  Keys: {data[0].keys()}")
                                if 'date' in data[0]:
                                    print(f"  Latest date: {data[0]['date']}")
                        print(f"  Sample response: {json.dumps(data[:1] if isinstance(data, list) else data, indent=2)[:500]}")
                    else:
                        print("  ✗ Empty response")
                except Exception as e:
                    print(f"  Error parsing response: {e}")
                    print(f"  Raw response: {response.text[:500]}")
            else:
                print(f"  Response: {response.text[:200]}")
    
    def list_available_transcripts(self, ticker):
        """List all available transcript dates for a ticker"""
        
        # Use the batch endpoint which shows all available transcripts
        url = f"{self.base_url}/v4/batch_earning_call_transcript/{ticker}"
        params = {'apikey': self.api_key}
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                print(f"\n✓ Available transcripts for {ticker}:")
                for transcript in data[:10]:  # Show first 10
                    date = transcript.get('date', 'Unknown')
                    quarter = transcript.get('quarter', 'N/A')
                    year = transcript.get('year', 'N/A')
                    print(f"  - {date} (Q{quarter} {year})")
                return data
        
        return []
    
    def download_latest_transcript(self, ticker):
        """Download the most recent transcript"""
        url = f"{self.base_url}/v3/earning_call_transcript/{ticker}"
        params = {'apikey': self.api_key}
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                transcript = data[0]
                
                print(f"\n✓ Latest transcript for {ticker}:")
                print(f"  Date: {transcript.get('date', 'N/A')}")
                print(f"  Quarter: {transcript.get('quarter', 'N/A')}")
                print(f"  Year: {transcript.get('year', 'N/A')}")
                
                content = transcript.get('content', '')
                print(f"  Length: {len(content)} characters")
                print(f"\n  Preview: {content[:500]}...")
                
                return transcript
        
        print(f"✗ No transcript found for {ticker}")
        return None
    
    def download_specific_transcript(self, ticker, year, quarter):
        """Download specific quarter transcript"""
        url = f"{self.base_url}/v3/earning_call_transcript/{ticker}"
        params = {
            'year': year,
            'quarter': quarter,
            'apikey': self.api_key
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                return data[0]
        
        return None


# Usage - Test your API
def main():
    API_KEY = "NdhfMhgaxwYCQOAkzVJhGgRcYuHFp9iU"  # Replace with your actual key
    
    tester = FMPTranscriptTester(API_KEY)
    
    # Step 1: Test API key
    print("Step 1: Testing API Key")
    print("="*60)
    if not tester.test_api_key():
        print("\n⚠️  Please check your API key!")
        print("Get a free key at: https://site.financialmodelingprep.com/developer/docs/")
        return
    
    # Step 2: Test with known companies
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    
    for ticker in test_tickers:
        print(f"\n{'='*60}")
        print(f"Testing {ticker}")
        print('='*60)
        
        # Check availability
        tester.check_transcript_availability(ticker)
        
        # Try to download latest
        tester.download_latest_transcript(ticker)
        
        # List available transcripts
        available = tester.list_available_transcripts(ticker)
        
        if available:
            # Download a specific one
            first = available[0]
            year = first.get('year')
            quarter = first.get('quarter')
            
            if year and quarter:
                print(f"\nTrying to download Q{quarter} {year}...")
                transcript = tester.download_specific_transcript(ticker, year, quarter)
                if transcript:
                    print("✓ Successfully downloaded!")
        
        break  # Just test first ticker for now


if __name__ == "__main__":
    main()