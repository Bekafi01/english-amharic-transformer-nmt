"""Hosted demo (Streamlit Community Cloud): a minimal translator UI over the published model.
Weights come from the Hub model repo (MODEL_ID) on each cold start. Set AMNMT_CHECKPOINT to a
local path to run the same UI locally. Colours live in .streamlit/config.toml.
"""

from __future__ import annotations

import html
import os
import re
import time

import streamlit as st

from amnmt.inference.translator import Translator

MODEL_ID = os.environ.get("AMNMT_MODEL_ID", "Bekafi/amnmt-en-am-base")
CHECKPOINT = os.environ.get("AMNMT_CHECKPOINT", f"hf://{MODEL_ID}")
REPO = "https://github.com/Bekafi01/english-amharic-transformer-nmt"
_ETHIOPIC = re.compile(r"[\u1200-\u137f]")

LANG = {"en": "English", "am": "Amharic"}
EXAMPLES = {
    "en-am": [
        "The children are playing in the garden.",
        "The meeting has been postponed until next Tuesday.",
        "Please wash your hands before eating.",
        "Coffee was first cultivated in the highlands of Ethiopia.",
    ],
    "am-en": [
        "ልጆቹ በአትክልት ስፍራ ውስጥ እየተጫወቱ ነው።",
        "ስብሰባው እስከ ሚቀጥለው ማክሰኞ ተራዝሟል።",
        "እባክዎን ከመብላትዎ በፊት እጅዎን ይታጠቡ።",
        "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።",
    ],
}

st.set_page_config(
    page_title="English ↔ Amharic",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={"Get help": REPO, "Report a bug": f"{REPO}/issues", "About": "amnmt"},
)

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Noto+Sans+Ethiopic:wght@400;500&display=swap" rel="stylesheet">
<style>
:root { --blue: #2563EB; --ink: #0F172A; --muted: #64748B; --line: #E2E8F0; --panel: #F3F6FB; }
html, body, [class*="css"] { font-family: Inter, "Noto Sans Ethiopic", system-ui, sans-serif; }
.block-container { max-width: 880px; padding-top: 3rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }
.t-title { font-size: 2.2rem; font-weight: 600; color: var(--ink); letter-spacing: -0.02em; margin: 0; }
.t-title span { color: var(--blue); }
.t-sub { color: var(--muted); font-size: .95rem; margin: .35rem 0 1.6rem 0; }
.t-sub a { color: var(--blue); text-decoration: none; }
.t-label { font-size: .75rem; font-weight: 600; color: var(--muted); letter-spacing: .08em; text-transform: uppercase; margin-bottom: .4rem; }
.stTextArea textarea { font-size: 1.1rem !important; line-height: 1.7 !important; border: 1px solid var(--line) !important; border-radius: 12px !important; background: #fff !important; color: var(--ink) !important; }
.stTextArea textarea:focus { border-color: var(--blue) !important; box-shadow: 0 0 0 3px rgba(37,99,235,.15) !important; }
.t-out { border-radius: 12px; background: var(--panel); min-height: 150px; padding: .95rem 1.1rem; font-size: 1.1rem; line-height: 1.7; color: var(--ink); font-family: "Noto Sans Ethiopic", Inter, sans-serif; }
.t-out .l + .l { margin-top: .5rem; padding-top: .5rem; border-top: 1px solid var(--line); }
.t-out .empty { color: var(--muted); }
.stButton > button[kind="primary"] { background: var(--blue); border: 0; border-radius: 10px; font-weight: 600; padding: .6rem 1.4rem; }
.stButton > button[kind="primary"]:hover { background: #1D4ED8; }
.stButton > button[kind="secondary"] { border: 1px solid var(--line); border-radius: 999px; color: var(--ink); background: #fff; font-size: .85rem; padding: .25rem .8rem; }
.stButton > button[kind="secondary"]:hover { border-color: var(--blue); color: var(--blue); }
.stButton > button[kind="tertiary"] { color: var(--blue); font-size: .85rem; padding: 0; }
div[data-testid="stRadio"] > div { gap: 0; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; width: max-content; }
div[data-testid="stRadio"] label { padding: .45rem 1rem; margin: 0; border-right: 1px solid var(--line); }
div[data-testid="stRadio"] label:last-child { border-right: 0; }
div[data-testid="stRadio"] label:has(input:checked) { background: var(--panel); }
div[data-testid="stRadio"] label > div:first-child { display: none; }
.t-meta { color: var(--muted); font-size: .82rem; margin-top: .5rem; }
.t-foot { color: var(--muted); font-size: .82rem; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--line); }
.t-foot a { color: var(--blue); text-decoration: none; }
div[data-testid="stExpander"] details { border: 1px solid var(--line); border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model…")
def load() -> Translator:
    return Translator.from_checkpoint(CHECKPOINT, device="cpu")


# ------------------------------------------------------------------ state
ss = st.session_state
ss.setdefault("direction", "en-am")
ss.setdefault("text", "")
ss.setdefault("result", None)  # (direction, [(src, hyp)], seconds)


def _swap() -> None:
    ss.direction = "am-en" if ss.direction == "en-am" else "en-am"
    ss.text = "\n".join(h for _, h in ss.result[1]) if ss.result else ""
    ss.result = None


def _use(example: str) -> None:
    ss.text = example
    ss.result = None


# ------------------------------------------------------------------ header
st.markdown(
    f"""
<p class="t-title">English <span>↔</span> Amharic</p>
<p class="t-sub">A from-scratch Transformer trained on a single Colab T4 ·
<a href="{REPO}">code</a> · <a href="https://huggingface.co/{MODEL_ID}">weights</a></p>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ direction
c1, c2 = st.columns([1, 1], vertical_alignment="center")
with c1:
    choice = st.radio(
        "Direction",
        ["en-am", "am-en"],
        index=0 if ss.direction == "en-am" else 1,
        horizontal=True,
        format_func=lambda d: f"{LANG[d[:2]]} → {LANG[d[3:]]}",
        label_visibility="collapsed",
    )
    if choice != ss.direction:
        ss.direction, ss.text, ss.result = choice, "", None
with c2:
    st.button("⇄  Swap languages", type="tertiary", on_click=_swap)

src_lang, tgt_lang = ss.direction.split("-")

# ------------------------------------------------------------------ source
st.markdown(f'<div class="t-label">{LANG[src_lang]}</div>', unsafe_allow_html=True)
text = st.text_area(
    "source",
    key="text",
    height=150,
    label_visibility="collapsed",
    placeholder="Type or paste text. One sentence per line.",
)

ex_cols = st.columns(len(EXAMPLES[ss.direction]))
for col, ex in zip(ex_cols, EXAMPLES[ss.direction], strict=True):
    col.button(
        ex if len(ex) <= 24 else ex[:23] + "…",
        key=f"ex-{ss.direction}-{ex[:10]}",
        help=ex,
        on_click=_use,
        args=(ex,),
        use_container_width=True,
    )

b1, b2 = st.columns([1, 3])
with b1:
    go = st.button("Translate", type="primary", use_container_width=True)
with b2:
    with st.popover("Options"):
        beam = st.select_slider("Beam size", options=[1, 2, 4, 6, 8], value=4)
        length_penalty = st.slider("Length penalty", 0.0, 2.0, 1.0, 0.1)

if go:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        st.warning("Enter at least one sentence.")
    else:
        if bool(_ETHIOPIC.search(lines[0])) != (src_lang == "am"):
            st.info("The input script doesn't match the selected direction — translating anyway.")
        tr = load()
        t0 = time.perf_counter()
        with st.spinner("Translating…"):
            hyps = tr.translate(lines, ss.direction, beam_size=beam, length_penalty=length_penalty)  # type: ignore[arg-type]
        ss.result = (ss.direction, list(zip(lines, hyps, strict=True)), time.perf_counter() - t0)

# ------------------------------------------------------------------ target
st.markdown(
    f'<div class="t-label" style="margin-top:1.4rem">{LANG[tgt_lang]}</div>', unsafe_allow_html=True
)
if ss.result and ss.result[0] == ss.direction:
    body = "".join(f'<div class="l">{html.escape(h) or "&nbsp;"}</div>' for _, h in ss.result[1])
    st.markdown(f'<div class="t-out">{body}</div>', unsafe_allow_html=True)
    n = len(ss.result[1])
    st.markdown(
        f'<div class="t-meta">{n} sentence{"s" if n != 1 else ""} · {ss.result[2]:.1f} s on CPU · beam {beam}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="t-out"><span class="empty">Translation appears here.</span></div>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ footer
st.markdown(
    f"""
<div class="t-foot">
  60.5M parameters · one model for both directions · FLORES-200 devtest:
  en→am <b>14.5</b> BLEU / 37.7 chrF++ · am→en <b>24.4</b> BLEU / 49.3 chrF++ ·
  <a href="{REPO}#design">how it works</a>
</div>
""",
    unsafe_allow_html=True,
)
with st.expander("Notes and limits"):
    st.markdown(
        """
- Trained on 4.4M public sentence pairs (MT560, OPUS-100, Tanzil, CCAligned, top-quartile NLLB)
  for 54k steps, ≈ 15 GPU-hours. FLORES-200 dev/devtest were excluded from training.
- Amharic output uses folded homophone spellings (ሀ for ሐ/ኀ, ሰ for ሠ, አ for ዐ, ጸ for ፀ).
- Long or technical sentences degrade; inputs are truncated at ~120 tokens.
- Code MIT; weights research / non-commercial (the NLLB bitext is CC-BY-NC-4.0).
"""
    )
