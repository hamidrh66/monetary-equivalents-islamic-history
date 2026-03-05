# Monetary Information Extraction from Arabic Historical Texts

[![DOI](https://zenodo.org/badge/1173369563.svg)](https://doi.org/10.5281/zenodo.18874976)

## Dissertation Context

This software was developed as part of the PhD dissertation:

**Hamid Reza Hakimi**  
University of Hamburg  

*Monetary Equivalents in Premodern Islamic Historical and Biographical Texts (1–1000 AH/600–1600 CE): Algorithmic Analysis into Economic History*  
(PhD dissertation, Chapter 1, §1.2.2)

---

## Project Overview

This repository contains a manifest-driven batch-processing pipeline for extracting structured monetary information from Arabic historical texts using the OpenAI Responses API with JSON Schema–constrained outputs.

The system supports:

- Large corpora (100k+ MIUs)
- Deterministic chunking
- Batch submission via OpenAI Batch API
- Automatic retry logic with exponential backoff
- Streaming merge
- Strict JSON Schema validation
- SHA-256 hashing for reproducibility

---

## Required Input Files

1. `sample_MIU_corpus.json`
2. `few_shot_examples.json`

---

### MIU File Structure

```json
{
  "miu_id_1": {
    "text": "Arabic text here"
  },
  "miu_id_2": {
    "text": "Another text"
  }
}
```

Each key is a unique MIU identifier.  
Each value must contain a `"text"` field with the Arabic source text.

---

## Installation

pip install -r requirements.txt

---

## Environment Setup

macOS / Linux: export OPENAI_API_KEY="sk-..."

Windows (PowerShell): setx OPENAI_API_KEY "sk-..."

Restart terminal after using setx.

---

## Usage

python monetary_extraction_batch_pipeline.py   --miu-json sample_MIU_corpus.json   --few-shot few_shot_examples.json   --work-dir batch_work   run

---

## Outputs

-   manifest.json
-   chunks/
-   outputs/
-   **merged_miu.json** (This will be the main output file.)
-   errors.json
-   empty_hits.json
-   merge_stats.json

---

## Citation

If you use this code, please cite the repository (see `CITATION.cff`) and/or the dissertation chapter.

---

## License

MIT License
