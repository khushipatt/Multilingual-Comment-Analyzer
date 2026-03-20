# 🌐 Multilingual Comment Analyzer — NLP Sem 6 Project

This is an app I built that takes product reviews or comments written in any language — Hindi, Japanese, Arabic, Bengali, Korean, you name it — and figures out whether each one is positive, negative, or neutral. It also auto-translates non-English comments before scoring them, so the sentiment analysis actually works across languages.

The sample dataset (`comments.txt`) has over 100 real-looking reviews written in 20+ languages. Some are even code-switched — like *"Totally paisa vasool! Best product mila hai"* — and it handles those too.

---

## 🧠 What's actually going on?

Here's the pipeline for every single comment that gets processed:

1. **Language detection** — `langdetect` figures out what language the comment is written in
2. **Translation** — if it's not English, `deep-translator` (backed by Google Translate) converts it to English
3. **Sentiment scoring** — `TextBlob` analyzes the English text and gives it a polarity score (-1 to +1) and a subjectivity score (0 to 1)
4. **Labeling** — based on the polarity, it gets tagged as Positive, Negative, or Neutral

So a Hindi review like *"यह उत्पाद बहुत अच्छा है!"* becomes *"This product is very good!"* internally, and then gets scored as Positive. The original text is always shown — translation is just for the analysis step.

---

## 🚀 Running it

**1. Install the packages**
```bash
pip install -r requirements.txt
```

**2. Launch the web app**
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. You can paste comments directly, use the built-in sample data, or upload your own `.txt` or `.csv` file.

**Or use the terminal version** if you don't want the UI:
```bash
python analyze.py --sample                        # runs the built-in 100+ comment dataset
python analyze.py --file comments.txt             # your own file
python analyze.py --text "यह बहुत अच्छा है!"     # single comment
python analyze.py --file comments.txt --save      # also exports results to results.csv
python analyze.py --file comments.txt --limit 20  # only analyze first 20
```

---

## 📁 What's in the repo

```
├── app.py           # Streamlit web app — full UI with charts and comment cards
├── analyze.py       # CLI version — same logic, runs in terminal with colored output
├── comments.txt     # Sample dataset — 100+ reviews in 20+ languages
├── requirements.txt # Python packages
└── config.toml      # Streamlit dark theme settings
```

---

## 💬 What the sample data looks like

The `comments.txt` file has reviews from all over the place — Spanish, Japanese, Dutch, Russian, Bengali, Tamil, Gujarati, Marathi, Korean, Arabic, and more. Some are completely positive, some are scathing complaints, and some are very mixed. A few favorites:

- *"Totally paisa vasool! Best product mila hai mujhe."* — Hinglish (Hindi + English)
- *"Le produit est good but delivery was very slow."* — French + English mid-sentence
- *"Das Produkt ist okay but I expected better quality for this price honestly."* — German + English

These code-switched ones are genuinely tricky and it was interesting to see how the app handled them.

---

## ⚙️ How the app works

**Web app (`app.py`)**

The sidebar lets you either paste comments into a text box, load the built-in samples, or upload a file. Hit analyze and it runs through every comment one by one, detecting language, translating, and scoring. Then it shows:

- Metric cards — total comments, how many languages found, positive/negative/neutral counts
- A donut chart of sentiment distribution
- A bar chart of language breakdown
- A scatter plot of polarity vs subjectivity (every dot is one comment)
- Individual comment cards, color-coded green/red/blue by sentiment, with the original text and optional English translation shown below
- Filters to narrow down by sentiment type or specific language
- Download buttons for full results or filtered results as CSV

**CLI version (`analyze.py`)**

Same pipeline, but prints color-coded results straight to your terminal. Green for positive, red for negative, blue for neutral. At the end it prints a summary with a mini bar chart of language counts. You can save to CSV with `--save`.

---

## 📦 Libraries used

| Library | What it does here |
|---|---|
| `langdetect` | Detects the language of each comment |
| `deep-translator` | Translates non-English comments to English via Google Translate |
| `textblob` | Scores sentiment — gives polarity and subjectivity values |
| `streamlit` | The whole web UI |
| `plotly` | The interactive charts (pie, bar, scatter) |
| `pandas` | Organizes results into a dataframe for filtering and export |

---

## 💡 Quick glossary (viva prep)

**Polarity** — A score from -1 to +1. Negative numbers = negative sentiment, positive = positive. Anything between -0.05 and +0.05 is treated as neutral.

**Subjectivity** — A score from 0 to 1. Close to 0 means the text is more factual/objective, close to 1 means it's more opinion-based. Complaints tend to be very subjective, straightforward descriptions less so.

**TextBlob** — A Python library built on top of NLTK that does basic NLP tasks including sentiment analysis. It works on English, which is why we translate first.

**langdetect** — A Python port of Google's language detection library. It uses character n-grams under the hood to figure out what language text is written in — no internet connection needed.

**deep-translator** — Wraps around Google Translate (and others) to do the actual translation. Needs internet.

**Code-switching** — When someone mixes two languages in the same sentence, like Hinglish or Spanglish. The detector handles these partially — it usually picks the dominant language.

---

*Semester 6 NLP project 🎓*
