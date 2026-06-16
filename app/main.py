"""
Campus Orientation Assistant - Streamlit App

Supports three input modes:
  1. Image upload (campus building / signage photo)
  2. Voice recording / audio file upload (transcribed via Whisper)
  3. Typed text query

Output panel shows: matched location name, description, opening hours,
events, and a map reference / directional text.
"""

import os
import sys
import json
import tempfile
import streamlit as st
import torch

# Ensure project root is on sys.path so `scripts` package can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.clip_faiss_retrieval import (
    load_clip, load_index, retrieve_location, encode_image, KB_PATH as CLIP_KB_PATH
)
from scripts.fusion import FusionRouter

KB_PATH = "data/knowledge_base/locations.json"
INTENT_MODEL_DIR = "models/intent_classifier/final"
FUSION_MODEL_PATH = "models/fusion/fusion_mlp.pt"


# ----------------------------------------------------------------------
# Cached model loaders
# ----------------------------------------------------------------------

@st.cache_resource
def get_clip():
    return load_clip()


@st.cache_resource
def get_faiss_index():
    model, processor = get_clip()
    return load_index()


@st.cache_resource
def get_whisper_model():
    import whisper
    return whisper.load_model("base")


@st.cache_resource
def get_intent_classifier():
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

    if os.path.isdir(INTENT_MODEL_DIR):
        tokenizer = DistilBertTokenizerFast.from_pretrained(INTENT_MODEL_DIR)
        model = DistilBertForSequenceClassification.from_pretrained(INTENT_MODEL_DIR)
        model.eval()
        return tokenizer, model
    else:
        return None, None


@st.cache_resource
def get_fusion_router():
    return FusionRouter(kb_path=KB_PATH, model_path=FUSION_MODEL_PATH)


@st.cache_resource
def get_kb():
    with open(KB_PATH, "r") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Pipeline helper functions
# ----------------------------------------------------------------------

def classify_image(image_path):
    model, processor = get_clip()
    index, id_map, kb = get_faiss_index()
    results = retrieve_location(model, processor, index, id_map, kb, image_path, top_k=3)
    return results[0], results  # top-1, all top-k


def transcribe_audio(audio_path):
    whisper_model = get_whisper_model()
    result = whisper_model.transcribe(audio_path)
    return result["text"].strip()


def extract_entity_from_text(text, kb):
    """
    Simple entity extraction: fuzzy keyword matching against KB location names.
    Falls back gracefully if no match is found.
    """
    text_lower = text.lower()
    best_match = None
    best_score = 0

    for record in kb:
        name_tokens = record["name"].lower().replace("-", " ").replace(",", " ").split()
        # Also check id tokens (e.g. "main_library" -> "main", "library")
        id_tokens = record["id"].replace("_", " ").split()
        all_tokens = set(name_tokens + id_tokens)

        score = sum(1 for tok in all_tokens if len(tok) > 2 and tok in text_lower)
        if score > best_score:
            best_score = score
            best_match = record["id"]

    if best_score > 0:
        return best_match
    return None


def classify_intent(text):
    tokenizer, model = get_intent_classifier()
    if tokenizer is None or model is None:
        return None, None

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=32)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred_idx = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[0, pred_idx].item())

    intent_label = model.config.id2label[pred_idx]
    return intent_label, confidence


def process_text_query(text, kb):
    intent, intent_conf = classify_intent(text)
    entity = extract_entity_from_text(text, kb)
    return {
        "intent": intent,
        "intent_confidence": intent_conf,
        "entity": entity,
    }


# ----------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------

def render_result(result):
    if not result.get("matched"):
        st.warning(result.get("message", "No match found."))
        if "note" in result:
            st.caption(result["note"])
        return

    loc = result["location"]
    st.success(f"📍 **{loc['name']}**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Category:** {loc['category'].replace('_', ' ').title()}")
        st.markdown(f"**Opening Hours:** {loc['opening_hours']}")
    with col2:
        st.markdown(f"**Map Reference:** {loc['map_reference']}")
        st.markdown(f"**Match confidence:** {result.get('confidence', 0):.2f} ({result.get('source')})")

    st.markdown("**Description:**")
    st.write(loc["description"])

    if loc.get("events"):
        st.markdown("**Upcoming Events:**")
        for event in loc["events"]:
            st.markdown(f"- {event['title']} — {event['date']} at {event['time']}")
    else:
        st.markdown("**Upcoming Events:** None scheduled.")

    if "note" in result:
        st.caption(f"ℹ️ {result['note']}")


# ----------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Campus Orientation Assistant", page_icon="🎓", layout="wide")
    st.title("🎓 Campus Orientation Assistant")
    st.caption("University of Edinburgh - Multimodal Campus Chatbot")

    kb = get_kb()
    router = get_fusion_router()

    tab_image, tab_voice, tab_text = st.tabs(["📷 Image Upload", "🎙️ Voice Query", "⌨️ Text Query"])

    # --- Image Upload Tab ---
    with tab_image:
        st.subheader("Upload a photo of a campus building or sign")
        uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="image_upload")

        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded image", width=300)

            with st.spinner("Analyzing image with CLIP..."):
                tmp_dir = tempfile.gettempdir()
                tmp_path = os.path.join(tmp_dir, uploaded_image.name)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_image.getbuffer())

                top1, top3 = classify_image(tmp_path)

            st.markdown("**Top-3 candidates:**")
            for r in top3:
                st.write(f"- {r['name']} (score: {r['score']:.3f})")

            result = router.route(image_result=top1)
            render_result(result)

    # --- Voice Query Tab ---
    with tab_voice:
        st.subheader("Upload a voice recording of your question")
        uploaded_audio = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"], key="audio_upload")

        if uploaded_audio is not None:
            st.audio(uploaded_audio)

            with st.spinner("Transcribing with Whisper..."):
                tmp_dir = tempfile.gettempdir()
                tmp_path = os.path.join(tmp_dir, uploaded_audio.name)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_audio.getbuffer())

                transcript = transcribe_audio(tmp_path)

            st.markdown(f"**Transcript:** _{transcript}_")

            text_result = process_text_query(transcript, kb)
            st.markdown(
                f"**Detected intent:** {text_result['intent']} "
                f"(confidence: {text_result['intent_confidence']:.2f}) | "
                f"**Detected location:** {text_result['entity'] or 'none'}"
            )

            result = router.route(text_result=text_result)
            render_result(result)

    # --- Text Query Tab ---
    with tab_text:
        st.subheader("Type your question")
        user_text = st.text_input("e.g. 'Where is the main library?' or 'Is the cafeteria open on Sundays?'")

        if st.button("Submit query") and user_text.strip():
            text_result = process_text_query(user_text, kb)
            st.markdown(
                f"**Detected intent:** {text_result['intent']} "
                f"(confidence: {text_result['intent_confidence']:.2f}) | "
                f"**Detected location:** {text_result['entity'] or 'none'}"
            )

            result = router.route(text_result=text_result)
            render_result(result)

    st.divider()
    st.caption(
        "Multimodal Campus Assistant. "
        )


if __name__ == "__main__":
    main()