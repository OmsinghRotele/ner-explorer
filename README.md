# 🏷️ Named Entity Recognition with BERT

A web app that identifies named entities in any text using the `dslim/bert-base-NER` model from Hugging Face Transformers, built with the Hugging Face `pipeline()` API and deployed as an interactive Gradio interface on HF Spaces.

🔗 **Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/OmsinghRotele/ner-explorer)

---

## What it does

Given any input text, the app detects and highlights the following entity types:

| Label | Type | Example |
|-------|------|---------|
| 🟣 PER | Person | *Elon Musk* |
| 🟠 ORG | Organization | *SpaceX, NASA* |
| 🟢 LOC | Location | *California, New York* |
| 🔵 MISC | Miscellaneous | *iPhone, Oscar* |

---

## Tech Stack

- **Model:** `dslim/bert-base-NER` (BERT fine-tuned on CoNLL-2003)
- **Framework:** Hugging Face Transformers
- **UI:** Gradio
- **Deployment:** Hugging Face Spaces

---

## Files

| File | Description |
|------|-------------|
| `app.py` | Main Gradio application |
| `requirements.txt` | Python dependencies |

---

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:7860` in your browser.
