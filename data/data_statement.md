# Data Statement

What is in this repository, what is not, and why.

## Summary

The paper reports experiments on **215,500 documents** (223,500 compiled, 215,500
retained after quality filtering) spanning 2015–2024. **That full corpus is not
redistributed here**, because a substantial part of it is licensed third-party
content that the authors do not have the right to republish.

What is provided instead:

1. **A sample of the public-domain portion** — real documents, not synthetic.
2. **The collection scripts** that rebuild the corpus from its original sources.
3. **Index files** listing document identifiers for part of the corpus (see coverage
   notes below). They do not cover every source or year, so the corpus can be
   rebuilt only in part from them.

## What ships in this repository

| Path | Contents | Size | Redistributable |
|---|---|---|---|
| `sec_filings_edgar/` | 21 SEC filings (2023), by accession number | 20.9 MB | ✅ publicly available (EDGAR) |
| `earnings_releases/` | 52 earnings releases — AAPL (20), GOOGL (17), MSFT (15), 2022–2025 (see note on 2025 files) | 33.1 MB | ✅ issuer-published |
| `fed_documents/` | 5 FOMC minutes (2020) | 2.7 MB | ✅ publicly available (Federal Reserve Board) |
| `sec_filings_index_2023_2024.csv` | Index of SEC filings in the test period | 17.3 MB | ✅ identifiers only |
| `fed_documents_index.csv` | Federal Reserve publication index | 0.02 MB | ✅ identifiers only |
| `finra_regulatory_documents_index.csv` | FINRA regulatory document index | 0.11 MB | ✅ identifiers only |
| `sec_regulatory_documents_index.csv` | SEC regulatory document index | <0.01 MB | ✅ identifiers only |
| `financial_news_2024_complete.csv` | News metadata (headlines, URLs, timestamps) | 8.85 MB | ⚠️ metadata only — see below |
| `google_news_historical_2024.csv` | Historical news metadata | 2.04 MB | ⚠️ metadata only — see below |
| `data_collection_scripts/` | 12 retrieval scripts | 0.1 MB | ✅ |

**Total shipped: ~85 MB**, against a full corpus in the tens of gigabytes.

### Coverage of the shipped files relative to the corpus period (2015–2024)

The shipped files are samples and indices; they do not mirror the corpus period
exactly.

- **Earnings releases:** 8 of the 52 files are dated 2025 (AAPL 4, GOOGL 4). They
  post-date the 2015–2024 corpus, were not part of the reported experiments, and
  are included only as illustrative examples of the document format.
- **SEC filings index** (`sec_filings_index_2023_2024.csv`): 197,680 entries dated
  2023–2024 (8-K, 10-K, 10-Q and amendments), i.e. the test period only. It is a
  filing index, not a list of the 47,000 SEC filings used; no index for 2015–2022
  is shipped.
- **Federal Reserve index** (`fed_documents_index.csv`): publications from
  1996–2006 and 2020–2023. Entries before 2015 are outside the corpus period.
- **FOMC minutes** (`fed_documents/`): 5 documents from 2020.
- **News metadata** (`financial_news_2024_complete.csv`,
  `google_news_historical_2024.csv`): 2024 only; no news index for 2015–2023 is
  shipped.
- **Earnings-call transcripts:** no index is shipped (licensed content).

## Provenance and licensing, by source

> The availability notes below reflect the authors' understanding of each source's
> terms. They are not a legal opinion. Anyone redistributing these documents further
> should check the terms of the original publisher.

### SEC filings (47,000 documents in the full corpus) — ✅ publicly available
Documents filed by companies with the U.S. Securities and Exchange Commission and
published free of charge on EDGAR. A sample is included; the full set is rebuildable
with `data_collection_scripts/secFiling.py` and `SEC_8_K_Filings.py` from
`sec_filings_index_2023_2024.csv`.

### Regulatory documents (8,500) — ✅ publicly available
Federal Reserve Board and FINRA publications, freely available from the publishers'
websites. Rebuildable with `Federal Reserve Board Publications.py` and
`FINRA Regulatory Documents.py`.

### Earnings call transcripts (12,000) — ⚠️ partly restricted
Issuer press releases are freely available and a sample is included. **Full
transcripts of the calls themselves are licensed** from commercial providers and
**are not redistributed**. `Earnings_Press_Releases.py` retrieves the public
portion.

### Financial news (156,000) — ❌ not redistributable
The largest single component, and the one that blocks release of the full corpus.
Article full text is copyright of the publishers. Only **metadata** — headline,
URL, publication timestamp, source — is shipped, which is sufficient to identify
each document but not to reconstruct its text without independent access.
Retrieval scripts: `Google_Financial_News.py`, `Financial_News_Sources.py`,
`Financial_News_Papers.py`, `Fixed Google News Historical Downloader.py`.

## FIBO ontology

FIBO version **2024-Q1** is not vendored here. It is published by the Enterprise
Data Management Council under the MIT License (see the `LICENSE` file in
<https://github.com/edmcouncil/fibo>) and should be obtained from
<https://spec.edmcouncil.org/fibo/>. The repository ships only the **custom
extension** built on top of it: `knowledge_graph/fibo-ext-temporal-2024Q1.owl`
(the file named in Section 3.2.1 of the manuscript).

Modules used: FND/Agents (73 classes, 152 properties), BE/LegalEntities (47, 84),
FBC/Products (68, 127) — 188 classes and 363 properties combined.

## Leakage-prevention protocol

- The split is a **single temporal holdout**: train 2015–2022, validation the tail
  of 2022, test 2023–2024. It is **not** a random split and **not** k-fold.
- The extended FIBO knowledge graph is built **exclusively from the training
  partition**, then **frozen**. Validation and test documents are linked and
  processed against that frozen graph; **no nodes or edges are added** at
  evaluation time.
- The BART+KG baseline queries the **same frozen snapshot**, so both systems
  operate under identical leakage constraints.
- **Entity-level disjointness across periods was deliberately not enforced.**
  Issuers recur across time, as they do in practice. The temporal design guards
  against exposure to test-period *content*, not against entity *familiarity*.
  This is a design choice.
- GPT-4 5-shot exemplars were drawn from the 2015–2022 training partition and are
  disjoint from the test period.

## What cannot be reconstructed

1. **The exact processed train/validation/test split files** were not retained.
   The split *rule* is fully specified (dates above) and is deterministic given
   the same corpus, but the materialised split files are gone.
2. **The 100-document expert-curated gold standard** contains expert-curated
   reference summaries. These are not included pending confirmation that the
   annotators' agreements permit release.
3. **Licensed news and transcript text**, as described above.

## Rebuilding the corpus

```bash
cd data/data_collection_scripts

python secFiling.py                       # SEC filings from EDGAR
python SEC_8_K_Filings.py                 # 8-K filings
python "Federal Reserve Board Publications.py"
python "FINRA Regulatory Documents.py"
python Earnings_Press_Releases.py         # public earnings releases
python Google_Financial_News.py           # news metadata; text requires
                                          # independent publisher access
```

These scripts reach live external services. Availability, rate limits and site
structure change over time, so a rebuild in a later year will not return a
byte-identical corpus. Documents are retrieved by identifier where the source
supports it, which makes the filing and regulatory portions stable; the news
portion is the least stable.
