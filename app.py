import gradio as gr
from transformers import pipeline
import json
import os

os.environ["HF_HOME"] = "/tmp/huggingface"

MODEL_NAME = "dslim/bert-base-NER"

COLOUR_MAP = {
    "PER":  ("#c084fc", "#2d1a4a"),
    "ORG":  ("#fb923c", "#3d1f0a"),
    "LOC":  ("#34d399", "#0a2d1f"),
    "MISC": ("#60a5fa", "#0a1a3d"),
}

EXAMPLES = [
    "Elon Musk founded SpaceX in Hawthorne, California and later acquired Twitter.",
    "The United Nations headquarters is located in New York City.",
    "Apple Inc. released the iPhone at Macworld in San Francisco.",
    "Cristiano Ronaldo plays for Al Nassr in Saudi Arabia.",
    "NASA launched the James Webb Space Telescope from French Guiana in 2021.",
]

ner_pipeline = None

def get_pipeline():
    global ner_pipeline
    if ner_pipeline is None:
        ner_pipeline = pipeline("ner", model=MODEL_NAME, aggregation_strategy="simple")
    return ner_pipeline

def run_ner(text: str):
    if not text.strip():
        return "<p style='color:#666'>Enter some text above.</p>", ""

    try:
        ner = get_pipeline()
        entities = ner(text)
    except Exception as e:
        return f"<p style='color:red'>Error loading model: {e}</p>", ""

    # Build highlighted HTML
    html = """
    <div style='font-family:sans-serif;font-size:1rem;line-height:2.2rem;
                padding:1rem;background:#0d0d15;border-radius:8px;color:#e8e8f0;'>
    """
    cursor = 0
    for ent in entities:
        start, end = ent["start"], ent["end"]
        label = ent["entity_group"]
        score = ent["score"]
        fg, bg = COLOUR_MAP.get(label, ("#aaa", "#222"))

        html += text[cursor:start]
        html += (
            f"<mark style='background:{bg};border:1.5px solid {fg};"
            f"border-radius:5px;padding:2px 6px;margin:0 2px;color:{fg};font-weight:600;'>"
            f"{text[start:end]}"
            f"<sup style='font-size:0.6rem;margin-left:3px;'>"
            f"{label} {score:.0%}</sup></mark>"
        )
        cursor = end

    html += text[cursor:] + "</div>"

    summary = json.dumps([
        {"entity": e["entity_group"], "word": e["word"],
         "score": round(float(e["score"]), 4), "start": int(e["start"]), "end": int(e["end"])}
        for e in entities
    ], indent=2)

    return html, summary


css = """
.gradio-container { max-width: 900px !important; margin: auto; }
#title { text-align: center; padding: 1.5rem 0 0.5rem; }
#title h1 {
    font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(135deg, #c084fc, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
#title p { color: #555570; font-size: 0.85rem; letter-spacing: 0.08em; }
.legend { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:0.5rem; }
.pill { padding:3px 12px; border-radius:99px; font-size:0.72rem;
        font-weight:700; letter-spacing:0.06em; }
"""

with gr.Blocks(css=css, title="NER Explorer") as demo:

    gr.HTML("""
    <div id='title'>
      <h1>🏷️ NER Explorer</h1>
      <p>NAMED ENTITY RECOGNITION · BERT · CoNLL-2003</p>
      <div class='legend'>
        <span class='pill' style='background:#2d1a4a;border:1px solid #c084fc;color:#c084fc'>PER · Person</span>
        <span class='pill' style='background:#3d1f0a;border:1px solid #fb923c;color:#fb923c'>ORG · Organization</span>
        <span class='pill' style='background:#0a2d1f;border:1px solid #34d399;color:#34d399'>LOC · Location</span>
        <span class='pill' style='background:#0a1a3d;border:1px solid #60a5fa;color:#60a5fa'>MISC · Miscellaneous</span>
      </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Type or paste any text here…",
                lines=6,
            )
            run_btn = gr.Button("▶ Detect Entities", variant="primary", size="lg")
            gr.Examples(examples=EXAMPLES, inputs=text_input, label="Try an example")

        with gr.Column(scale=1):
            html_output = gr.HTML(label="Highlighted Output")
            json_output = gr.Textbox(label="JSON Output", lines=10, interactive=False)

    run_btn.click(fn=run_ner, inputs=text_input, outputs=[html_output, json_output])
    text_input.submit(fn=run_ner, inputs=text_input, outputs=[html_output, json_output])

demo.launch()
