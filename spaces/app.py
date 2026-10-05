"""Hugging Face Space: Streamlit UI over the published model. Weights come from the Hub model repo
(MODEL_ID) on first start and are cached by the Space afterwards.
"""

from __future__ import annotations

import os
import re
import time

import streamlit as st

from amnmt.inference.translator import Translator

MODEL_ID = os.environ.get("AMNMT_MODEL_ID", "Bekafi01/amnmt-en-am-base")
_ETHIOPIC = re.compile(r"[\u1200-\u137f]")

EXAMPLES = {
    "en-am": [
        "The children are playing in the garden.",
        "The meeting has been postponed until next Tuesday.",
        "Please wash your hands before eating.",
        "Scientists have discovered a new species of frog in the rainforest.",
    ],
    "am-en": [
        "ልጆቹ በአትክልት ስፍራ ውስጥ እየተጫወቱ ነው።",
        "ስብሰባው እስከ ሚቀጥለው ማክሰኞ ተራዝሟል።",
        "እባክዎን ከመብላትዎ በፊት እጅዎን ይታጠቡ።",
        "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።",
    ],
}

st.set_page_config(page_title="English ↔ Amharic", page_icon="🇪🇹", layout="centered")


@st.cache_resource(show_spinner="Downloading model (first start only)…")
def load() -> Translator:
    return Translator.from_checkpoint(f"hf://{MODEL_ID}", device="cpu")


st.title("English ↔ Amharic translation")
st.caption(
    "From-scratch 60M-parameter Transformer, one model for both directions. "
    "FLORES-200: en→am 14.5 BLEU · am→en 24.4 BLEU. "
    f"[Code](https://github.com/Bekafi01/english-amharic-transformer-nmt) · "
    f"[Weights](https://huggingface.co/{MODEL_ID})"
)

with st.sidebar:
    st.header("Decoding")
    beam = st.slider("Beam size", 1, 8, 4, help="1 = greedy. 4 is the evaluated setting.")
    length_penalty = st.slider("Length penalty", 0.0, 2.0, 1.0, 0.1)
    st.caption("CPU inference; ~0.3 s per sentence at beam 4.")

direction = st.radio(
    "Direction",
    ["en-am", "am-en"],
    horizontal=True,
    format_func=lambda d: {"en-am": "English → Amharic", "am-en": "Amharic → English"}[d],
)

if "text" not in st.session_state:
    st.session_state.text = ""
cols = st.columns(len(EXAMPLES[direction]))
for col, ex in zip(cols, EXAMPLES[direction], strict=True):
    if col.button(ex[:22] + "…", help=ex, use_container_width=True):
        st.session_state.text = ex

text = st.text_area("Input (one sentence per line)", key="text", height=140)

if st.button("Translate", type="primary"):
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        st.warning("Enter at least one sentence.")
    else:
        has_ethiopic = bool(_ETHIOPIC.search(lines[0]))
        if has_ethiopic != (direction == "am-en"):
            st.info("The input script does not match the selected direction — translating anyway.")
        tr = load()
        t0 = time.perf_counter()
        with st.spinner("Translating…"):
            outputs = tr.translate(lines, direction, beam_size=beam, length_penalty=length_penalty)  # type: ignore[arg-type]
        for src, hyp in zip(lines, outputs, strict=True):
            st.markdown(f"**{src}**")
            st.markdown(f"→ {hyp}")
            st.divider()
        st.caption(f"{len(lines)} sentence(s) in {time.perf_counter() - t0:.1f} s · beam {beam}")

with st.expander("About this model"):
    st.markdown(
        """
- **Architecture**: Pre-LN encoder–decoder Transformer (6+6 layers, d=512, 8 heads), 3-way tied
  32k joint BPE embedding, written from `torch.nn` primitives and verified against `nn.Transformer`.
- **Data**: MT560, OPUS-100, Tanzil, CCAligned and the top-quartile (by LASER score) of the NLLB
  mined bitext — 4.4M pairs after Ethiopic normalization, filtering and deduplication. FLORES-200
  dev/devtest were excluded from training.
- **Training**: 54k steps (7 epochs) with label smoothing, inverse-sqrt LR, fp16 — about 15 hours
  on a single Colab T4, resumed across four sessions.
- **Known limits**: Amharic output uses folded homophone spellings (ሀ for ሐ/ኀ, ሰ for ሠ, አ for ዐ,
  ጸ for ፀ); long or highly technical sentences degrade; inputs are truncated at ~120 tokens.
- **License**: code MIT; weights for research / non-commercial use (NLLB bitext is CC-BY-NC-4.0).
"""
    )
