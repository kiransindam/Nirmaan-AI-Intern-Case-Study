# Nirmaan AI Intern Case Study – Self-Introduction Scoring Tool

This project implements an AI-powered rubric-based scoring system for evaluating students' **self-introduction transcripts** on communication skills. It combines **rule-based logic**, **NLP semantic analysis**, and **data-driven weighting** as specified in the Nirmaan AI case study.

The tool accepts a transcript (text), analyzes it across 8 communication criteria, and outputs:
- A normalized **overall score (0–100)**
- **Per-criterion scores** with weights
- **Actionable feedback** for each dimension

✅ Matches the rubric from `Case study for interns.xlsx`  
✅ Uses the sample transcript from `Sample text for case study.txt`  
✅ Implements all three required approaches: rule-based, NLP-based, and rubric-driven

---

## 📊 Rubric Overview

| Category              | Metric                     | Weight | Max Score |
|-----------------------|----------------------------|--------|----------|
| Content & Structure   | Salutation Level           | 5%     | 5        |
|                       | Keyword Presence           | 30%    | 30       |
|                       | Flow                       | 5%     | 5        |
| Speech Rate           | Words per Minute (WPM)     | 10%    | 10       |
| Language & Grammar    | Grammar Errors             | 10%    | 10       |
|                       | Vocabulary Richness (TTR)  | 10%    | 10       |
| Clarity               | Filler Word Rate           | 15%    | 15       |
| Engagement            | Sentiment/Positivity       | 15%    | 15       |
| **Total**             |                            | **100%** | **100**    |

> **Expected output for sample transcript**: **86/100**

# Project Structure
```bash
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── assets/               # (Optional) for screenshots/video
---
```
## 🛠️ Tech Stack

- **Frontend**: Streamlit (Python-based web UI)
- **NLP Libraries**:
  - `sentence-transformers` → semantic similarity scoring
  - `nltk` + `vaderSentiment` → sentiment & text analysis
- **Rule Logic**: Custom keyword matching, filler detection, TTR, WPM
- **Grammar Proxy**: TextBlob (for lightweight demo); can be upgraded to LanguageTool

> 💡 No external APIs required — runs entirely offline after initial model download.

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.8+
- `pip`

### Steps

1. **Clone or create the project folder**
    ```bash
   git init
   git add .
   git commit -m "Initial commit: Nirmaan AI Communication Scorer"
   git remote add origin your-repo-url
   git push -u origin main
     ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
 3. **Run the app**
 ```bash
 streamlit run app.py
 ```
4. **App URL**
   link : https://nirmaan-ai-intern-case-study.streamlit.app/ 
