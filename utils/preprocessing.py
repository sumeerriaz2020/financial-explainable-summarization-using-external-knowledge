"""
Document Preprocessing Pipeline
================================

Comprehensive preprocessing pipeline for financial documents including
text cleaning, normalization, tokenization, and structure extraction.

Reference: Section 3.5 (Dataset Construction and Preprocessing)
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Preprocessed document"""
    text: str
    cleaned_text: str
    sentences: List[str]
    paragraphs: List[str]
    metadata: Dict
    sections: Dict[str, str]
    statistics: Dict


class DocumentPreprocessor:
    """
    Document preprocessing pipeline for financial texts
    
    Handles: cleaning, normalization, segmentation, structure extraction
    """
    
    def __init__(
        self,
        lowercase: bool = False,
        remove_numbers: bool = False,
        expand_contractions: bool = True,
        min_sentence_length: int = 10
    ):
        """
        Initialize preprocessor
        
        Args:
            lowercase: Convert to lowercase
            remove_numbers: Remove numerical values
            expand_contractions: Expand contractions ("don't" -> "do not")
            min_sentence_length: Minimum sentence length
        """
        self.lowercase = lowercase
        self.remove_numbers = remove_numbers
        self.expand_contractions = expand_contractions
        self.min_sentence_length = min_sentence_length
        
        # Contraction mappings
        self.contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "hasn't": "has not", "haven't": "have not",
            "hadn't": "had not", "doesn't": "does not", "didn't": "did not",
            "I'm": "I am", "you're": "you are", "he's": "he is",
            "she's": "she is", "it's": "it is", "we're": "we are",
            "they're": "they are", "I've": "I have", "you've": "you have",
            "we've": "we have", "they've": "they have"
        }
        
        # Financial abbreviations
        self.abbreviations = {
            "YoY": "year over year",
            "QoQ": "quarter over quarter",
            "EBITDA": "earnings before interest taxes depreciation and amortization",
            "ROI": "return on investment",
            "P/E": "price to earnings",
            "EPS": "earnings per share",
            "M&A": "mergers and acquisitions"
        }
        
        logger.info("Document Preprocessor initialized")
    
    def preprocess(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> ProcessedDocument:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Raw document text
            metadata: Optional document metadata
            
        Returns:
            ProcessedDocument with cleaned text and structure
        """
        original_text = text
        
        # Step 1: Basic cleaning
        cleaned = self._clean_text(text)
        
        # Step 2: Normalize
        normalized = self._normalize_text(cleaned)
        
        # Step 3: Segment
        sentences = self._segment_sentences(normalized)
        paragraphs = self._segment_paragraphs(normalized)
        
        # Step 4: Extract sections
        sections = self._extract_sections(normalized)
        
        # Step 5: Compute statistics
        stats = self._compute_statistics(original_text, normalized, sentences)
        
        return ProcessedDocument(
            text=original_text,
            cleaned_text=normalized,
            sentences=sentences,
            paragraphs=paragraphs,
            metadata=metadata or {},
            sections=sections,
            statistics=stats
        )
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text.strip()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text"""
        # Expand contractions
        if self.expand_contractions:
            for contraction, expansion in self.contractions.items():
                text = re.sub(
                    r'\b' + re.escape(contraction) + r'\b',
                    expansion,
                    text,
                    flags=re.IGNORECASE
                )
        
        # Expand financial abbreviations
        for abbr, expansion in self.abbreviations.items():
            text = re.sub(
                r'\b' + re.escape(abbr) + r'\b',
                expansion,
                text
            )
        
        # Normalize currency symbols
        text = re.sub(r'\$\s*(\d)', r'USD \1', text)
        text = re.sub(r'€\s*(\d)', r'EUR \1', text)
        text = re.sub(r'£\s*(\d)', r'GBP \1', text)
        
        # Normalize percentages
        text = re.sub(r'(\d+)\s*%', r'\1 percent', text)
        
        # Remove numbers if requested
        if self.remove_numbers:
            text = re.sub(r'\b\d+\.?\d*\b', ' NUMBER ', text)
        
        # Lowercase if requested
        if self.lowercase:
            text = text.lower()
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences"""
        # Simple sentence boundary detection
        sentences = []
        
        # Split on sentence boundaries
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        raw_sentences = re.split(pattern, text)
        
        for sent in raw_sentences:
            sent = sent.strip()
            
            # Filter short sentences
            if len(sent) >= self.min_sentence_length:
                sentences.append(sent)
        
        return sentences
    
    def _segment_paragraphs(self, text: str) -> List[str]:
        """Segment text into paragraphs"""
        # Split on double newlines or multiple spaces
        paragraphs = re.split(r'\n\s*\n|\n{2,}', text)
        
        # Clean and filter
        paragraphs = [
            p.strip() for p in paragraphs
            if p.strip() and len(p.strip()) > 50
        ]
        
        return paragraphs
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract document sections"""
        sections = {}
        
        # Common financial document sections
        section_patterns = {
            'executive_summary': r'(?i)(executive summary|overview)(.*?)(?=\n[A-Z]|\Z)',
            'financial_highlights': r'(?i)(financial highlights|key metrics)(.*?)(?=\n[A-Z]|\Z)',
            'business_overview': r'(?i)(business overview|operations)(.*?)(?=\n[A-Z]|\Z)',
            'risk_factors': r'(?i)(risk factors|risks)(.*?)(?=\n[A-Z]|\Z)',
            'outlook': r'(?i)(outlook|forecast|guidance)(.*?)(?=\n[A-Z]|\Z)'
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                section_text = match.group(2).strip()
                if len(section_text) > 100:  # Minimum section length
                    sections[section_name] = section_text[:1000]  # Limit length
        
        return sections
    
    def _compute_statistics(
        self,
        original: str,
        cleaned: str,
        sentences: List[str]
    ) -> Dict:
        """Compute document statistics"""
        words = cleaned.split()
        
        return {
            'original_length': len(original),
            'cleaned_length': len(cleaned),
            'num_sentences': len(sentences),
            'num_words': len(words),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'num_unique_words': len(set(words)),
            'vocabulary_richness': len(set(words)) / len(words) if words else 0
        }
    
    def batch_preprocess(
        self,
        documents: List[Dict]
    ) -> List[ProcessedDocument]:
        """
        Preprocess multiple documents
        
        Args:
            documents: List of dicts with 'text' and optional 'metadata'
            
        Returns:
            List of ProcessedDocument objects
        """
        processed = []
        
        for doc in documents:
            try:
                result = self.preprocess(
                    text=doc['text'],
                    metadata=doc.get('metadata')
                )
                processed.append(result)
            except Exception as e:
                logger.error(f"Error preprocessing document: {e}")
                continue
        
        logger.info(f"Preprocessed {len(processed)}/{len(documents)} documents")
        return processed


class FinancialTextCleaner:
    """Specialized cleaner for financial documents"""
    
    @staticmethod
    def remove_tables(text: str) -> str:
        """Remove ASCII tables"""
        # Remove lines with mostly | or + characters
        lines = text.split('\n')
        cleaned_lines = [
            line for line in lines
            if not (line.count('|') > 3 or line.count('+') > 3)
        ]
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def standardize_numbers(text: str) -> str:
        """Standardize number formats"""
        # Remove commas from numbers
        text = re.sub(r'(\d),(\d)', r'\1\2', text)
        
        # Standardize billions/millions
        text = re.sub(r'(\d+\.?\d*)\s*billion', r'\1B', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d+\.?\d*)\s*million', r'\1M', text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def extract_financial_figures(text: str) -> List[Dict]:
        """Extract financial figures with context"""
        figures = []
        
        # Pattern for financial figures
        pattern = r'(\w+(?:\s+\w+){0,3})\s+(?:of|was|is|:\s*)?(\$?\d+\.?\d*[BMK]?)'
        
        for match in re.finditer(pattern, text):
            figures.append({
                'context': match.group(1),
                'value': match.group(2),
                'position': match.start()
            })
        
        return figures


# Example usage
if __name__ == "__main__":
    print("Document Preprocessing Pipeline")
    print("=" * 70)
    
    # Sample financial text
    sample_text = """
    EXECUTIVE SUMMARY
    
    Apple Inc. reported Q4 2023 earnings of $89.5B, representing YoY growth 
    of 12%. The company's iPhone 15 launch led to increased revenue. EBITDA 
    improved by 15% QoQ. The P/E ratio stands at 28.5.
    
    FINANCIAL HIGHLIGHTS
    
    Revenue: $89.5 billion (up 12% YoY)
    Net Income: $22.3 billion
    EPS: $1.46
    Operating Margin: 29.8%
    
    BUSINESS OVERVIEW
    
    The company's services segment grew 18%, driven by App Store sales and 
    subscription revenue. M&A activity included the acquisition of a ML startup 
    for $200M. The company doesn't expect significant headwinds in Q1 2024.
    
    RISK FACTORS
    
    Currency fluctuations may impact international sales. Supply chain 
    disruptions remain a concern. Competition in the smartphone market is 
    intensifying.
    """
    
    # Initialize preprocessor
    preprocessor = DocumentPreprocessor(
        lowercase=False,
        remove_numbers=False,
        expand_contractions=True
    )
    
    # Preprocess document
    print("\nPreprocessing document...")
    result = preprocessor.preprocess(sample_text, metadata={'source': 'example'})
    
    # Display results
    print(f"\n{'=' * 70}")
    print("PREPROCESSING RESULTS:")
    print(f"{'=' * 70}")
    
    print(f"\nStatistics:")
    for key, value in result.statistics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nSections extracted: {len(result.sections)}")
    for section_name in result.sections.keys():
        print(f"  - {section_name}")
    
    print(f"\nSentences: {len(result.sentences)}")
    print(f"First sentence: {result.sentences[0][:100]}...")
    
    # Test financial cleaner
    print(f"\n{'=' * 70}")
    print("FINANCIAL TEXT CLEANING:")
    print(f"{'=' * 70}")
    
    cleaner = FinancialTextCleaner()
    
    # Extract financial figures
    figures = cleaner.extract_financial_figures(sample_text)
    print(f"\nExtracted {len(figures)} financial figures:")
    for fig in figures[:5]:
        print(f"  {fig['context']}: {fig['value']}")
    
    # Standardize numbers
    standardized = cleaner.standardize_numbers(sample_text)
    print(f"\nNumber standardization applied")
    
    print(f"\n{'=' * 70}")
    print("Preprocessing pipeline ready!")
    print(f"{'=' * 70}")
