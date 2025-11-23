import streamlit as st
import re
import json
from datetime import datetime
from collections import Counter
from textblob import TextBlob
import nltk
from sentence_transformers import SentenceTransformer, util

# ----------------------------
# MUST be the FIRST Streamlit command
# ----------------------------
st.set_page_config(
    page_title="Nirmaan AI: Communication Scorer",
    layout="wide",
    page_icon="🗣️"
)

# ----------------------------
# Downloads (run once)
# ----------------------------
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# ----------------------------
# Constants (from rubric)
# ----------------------------
FILLER_WORDS = {
    "um", "uh", "like", "you know", "so", "actually", "basically", "right",
    "i mean", "well", "kinda", "sort of", "okay", "hmm", "ah"
}

# Load semantic model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ----------------------------
# SCORING FUNCTIONS (aligned with rubric)
# ----------------------------

def salutation_score(text):
    text_lower = text.lower()
    if any(phrase in text_lower for phrase in ["i am excited", "feeling great", "thrilled to introduce"]):
        return 5  # Excellent
    elif any(phrase in text_lower for phrase in ["good morning", "good afternoon", "good evening", "good day", "hello everyone"]):
        return 4  # Good
    elif any(phrase in text_lower for phrase in ["hi", "hello"]):
        return 2  # Normal
    else:
        return 0  # No salutation

def keyword_score(text_lower, original_text):
    score_must = 0
    found_must = []
    # Must-have (4 pts each → max 20)
    if any(w in text_lower for w in ["myself", "i am", "name"]):
        score_must += 4; found_must.append("name")
    if any(str(age) in text_lower for age in range(10, 20)):
        score_must += 4; found_must.append("age")
    if "class" in text_lower or "grade" in text_lower or "section" in text_lower or "school" in text_lower:
        score_must += 4; found_must.append("class/school")
    if "family" in text_lower:
        score_must += 4; found_must.append("family")
    if any(hob in text_lower for hob in ["play", "cricket", "hobb", "interest", "like to", "enjoy", "free time"]):
        score_must += 4; found_must.append("hobbies/interests")
    
    score_good = 0
    found_good = []
    # Good-to-have (2 pts each → max 10)
    if "kind heart" in text_lower or "soft spoken" in text_lower:
        score_good += 2; found_good.append("about family")
    if "from" in text_lower and ("i am from" in original_text.lower() or "parents are from" in original_text.lower()):
        score_good += 2; found_good.append("origin")
    if "science" in text_lower or "explore" in text_lower or "discover" in text_lower or "improve lives" in text_lower:
        score_good += 2; found_good.append("goal/ambition")
    if "mirror" in text_lower or "stole" in text_lower or "fun fact" in text_lower:
        score_good += 2; found_good.append("fun fact / unique")
    if "strength" in text_lower or "achievement" in text_lower or "improve the lives" in text_lower:
        score_good += 2; found_good.append("strength/achievement")
    
    return score_must + score_good, found_must, found_good

def flow_score(text):
    sentences = nltk.sent_tokenize(text)
    text_lower = text.lower()
    # Check: starts with salutation, ends with thank
    starts_ok = any(g in text_lower.split()[:5] for g in ["hello", "hi", "good"])
    ends_ok = "thank" in text_lower or "thanks" in text_lower
    return 5 if (starts_ok and ends_ok) else 0

def speech_rate_score(word_count, duration_sec):
    if duration_sec <= 0:
        return 10  # Assume ideal (as per case study note: audio not provided)
    wpm = (word_count / duration_sec) * 60
    if wpm > 160:
        return 2
    elif 141 <= wpm <= 160:
        return 6
    elif 111 <= wpm <= 140:
        return 10
    elif 81 <= wpm <= 110:
        return 6
    else:
        return 2

def grammar_score(text, word_count):
    # Simulate LanguageTool: count non-standard sentence endings
    sentences = nltk.sent_tokenize(text)
    errors = sum(1 for s in sentences if s and not s.strip().endswith(('.', '!', '?')))
    errors_per_100 = (errors / word_count) * 100 if word_count > 0 else 0
    ratio = min(errors_per_100 / 10, 1)
    clean = max(0, 1 - ratio)
    if clean >= 0.9: return 10
    elif clean >= 0.7: return 8
    elif clean >= 0.5: return 6
    elif clean >= 0.3: return 4
    else: return 2

def ttr_score(text):
    words = text.lower().split()
    if not words:
        return 0
    ttr = len(set(words)) / len(words)
    if ttr >= 0.9: return 10
    elif ttr >= 0.7: return 8
    elif ttr >= 0.5: return 6
    elif ttr >= 0.3: return 4
    else: return 2

def filler_score(text, word_count):
    words = text.lower().split()
    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    rate = (filler_count / word_count) * 100 if word_count > 0 else 0
    if rate <= 3: return 15, rate, filler_count
    elif rate <= 6: return 12, rate, filler_count
    elif rate <= 9: return 9, rate, filler_count
    elif rate <= 12: return 6, rate, filler_count
    else: return 3, rate, filler_count

def engagement_score(text):
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    pos = sia.polarity_scores(text)['pos']
    if pos >= 0.9: return 15, pos
    elif pos >= 0.7: return 12, pos
    elif pos >= 0.5: return 9, pos
    elif pos >= 0.3: return 6, pos
    else: return 3, pos

# ----------------------------
# MAIN APP
# ----------------------------

st.title("🗣️ Nirmaan AI – Self-Introduction Scoring Tool")
st.markdown("Paste a student's **transcript** below to get a rubric-based communication score (0–100).")

# Input
transcript = st.text_area(
    "Transcript",
    height=200,
    value="""Hello everyone, myself Muskan, studying in class 8th B section from Christ Public School. 
I am 13 years old. I live with my family. There are 3 people in my family, me, my mother and my father.
One special thing about my family is that they are very kind hearted to everyone and soft spoken. One thing I really enjoy is play, playing cricket and taking wickets.
A fun fact about me is that I see in mirror and talk by myself. One thing people don't know about me is that I once stole a toy from one of my cousin.
 My favorite subject is science because it is very interesting. Through science I can explore the whole world and make the discoveries and improve the lives of others. 
Thank you for listening."""
)

duration_sec = st.number_input(
    "Duration (seconds)",
    min_value=0.0,
    value=52.0,
    step=1.0,
    help="From sample: 52 sec. Set to 0 if unknown → assumes ideal pace."
)

if st.button("🎯 Score Introduction"):
    if not transcript.strip():
        st.error("Please enter a transcript.")
    else:
        text = transcript.strip()
        text_lower = text.lower()
        word_count = len(text.split())
        sentence_count = len(nltk.sent_tokenize(text))

        # Compute scores
        sal = salutation_score(text)
        kw, found_must, found_good = keyword_score(text_lower, text)
        flow = flow_score(text)
        sr = speech_rate_score(word_count, duration_sec)
        gram = grammar_score(text, word_count)
        ttr = ttr_score(text)
        filler, filler_rate, filler_count = filler_score(text, word_count)
        eng, pos_prob = engagement_score(text)

        total = sal + kw + flow + sr + gram + ttr + filler + eng
        final_score = min(100, max(0, round(total, 1)))

        # Build output
        output = {
            "overall_score": final_score,
            "words": word_count,
            "sentence_count": sentence_count,
            "criteria": [
                {"name": "Salutation Level", "score": sal, "max": 5, "feedback": f"Detected opening"},
                {"name": "Keyword Presence", "score": kw, "max": 30, "feedback": f"Must: {found_must}, Good: {found_good}"},
                {"name": "Flow", "score": flow, "max": 5, "feedback": "Follows expected order" if flow == 5 else "Order not ideal"},
                {"name": "Speech Rate (WPM)", "score": sr, "max": 10, "feedback": f"WPM: {((word_count/duration_sec)*60):.1f}" if duration_sec > 0 else "Assumed ideal"},
                {"name": "Grammar", "score": gram, "max": 10, "feedback": "Low error estimate"},
                {"name": "Vocabulary (TTR)", "score": ttr, "max": 10, "feedback": f"TTR = {len(set(text.lower().split()))/len(text.split()):.2f}"},
                {"name": "Clarity (Fillers)", "score": filler, "max": 15, "feedback": f"{filler_rate:.1f}% filler words"},
                {"name": "Engagement", "score": eng, "max": 15, "feedback": f"Positive sentiment: {pos_prob:.2f}"}
            ],
            "transcript": text,
            "generated_at": datetime.now().isoformat()
        }

        # Display
        st.subheader(f"📊 Overall Score: {final_score}/100")
        st.write(f"**Words**: {word_count} | **Sentences**: {sentence_count}")

        for c in output["criteria"]:
            st.write(f"**{c['name']}**: {c['score']}/{c['max']} — {c['feedback']}")

        # Semantic similarity (NLP-based)
        ideal = "A clear, confident, well-structured self-introduction including name, age, school, family, hobbies, goals, and a unique fact, with positive tone and smooth flow."
        emb1 = model.encode(text, convert_to_tensor=True)
        emb2 = model.encode(ideal, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(emb1, emb2).item()
        st.write(f"🧠 **Semantic Match to Ideal Intro**: {sim:.1%}")

        # Download
        st.download_button(
            "⬇️ Download Full Results (JSON)",
            data=json.dumps(output, indent=2),
            file_name=f"nirmaan_score_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

st.markdown("---")
st.caption("Built for Nirmaan AI Intern Case Study • Uses rule-based + NLP + rubric-weighted scoring")
st.caption("Created by • Kiran Sindam")
st.link_button("GitHub Repository", "https://github.com/kiransindam", icon="🐙")
st.link_button("LinkedIn Profile", "https://www.linkedin.com/in/kiransindam/", icon="🔗")
st.link_button("Portfolio Website", "https://kiransindam-39lyhoe.gamma.site/", icon="🌐")