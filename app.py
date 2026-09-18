"""Streamlit UI. Run:  AMNMT_CHECKPOINT=path/to/best.pt streamlit run app.py

Thin client over `amnmt.inference.Translator`; no model logic lives here.
"""

from __future__ import annotations

import os

import streamlit as st

from amnmt.inference.translator import Translator

st.set_page_config(page_title="English ↔ Amharic", page_icon="🌍", layout="centered")


@st.cache_resource(show_spinner="Loading model…")
def load(checkpoint: str, tokenizer: str | None, device: str | None) -> Translator:
    return Translator.from_checkpoint(checkpoint, tokenizer or None, device or None)


with st.sidebar:
    st.header("Model")
    checkpoint = st.text_input("Checkpoint", os.environ.get("AMNMT_CHECKPOINT", ""))
    tokenizer = st.text_input("Tokenizer (optional)", os.environ.get("AMNMT_TOKENIZER", ""))
    device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    st.header("Decoding")
    beam = st.slider("Beam size", 1, 8, 4)
    length_penalty = st.slider("Length penalty", 0.0, 2.0, 1.0, 0.1)

st.title("English ↔ Amharic translation")

direction = st.radio(
    "Direction",
    ["en-am", "am-en"],
    horizontal=True,
    format_func=lambda d: {"en-am": "English → Amharic", "am-en": "Amharic → English"}[d],
)
placeholder = (
    "The children are playing in the garden."
    if direction == "en-am"
    else "ልጆቹ በአትክልት ስፍራ ውስጥ እየተጫወቱ ነው።"
)
text = st.text_area("Input (one sentence per line)", height=160, placeholder=placeholder)

if st.button("Translate", type="primary", disabled=not checkpoint):
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        st.warning("Enter at least one sentence.")
    else:
        try:
            tr = load(checkpoint, tokenizer, None if device == "auto" else device)
        except Exception as e:
            st.error(f"Could not load model: {e}")
            st.stop()
        with st.spinner("Translating…"):
            outputs = tr.translate(lines, direction, beam_size=beam, length_penalty=length_penalty)  # type: ignore[arg-type]
        for src, hyp in zip(lines, outputs, strict=True):
            st.markdown(f"**{src}**")
            st.markdown(f"→ {hyp}")
            st.divider()
        st.caption(f"{tr.model.num_parameters():,} parameters · {tr.device} · beam {beam}")
elif not checkpoint:
    st.info("Set a checkpoint path in the sidebar (or the AMNMT_CHECKPOINT environment variable).")
