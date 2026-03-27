import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io
import math

st.set_page_config(
    page_title="Gemini Image Editor",
    page_icon="✨",
    layout="wide"
)

# ── Accounts ────────────────────────────────────────────────────────────────────
ACCOUNTS = {
    "barabasi": {"label": "Barabasi", "secret_key": "GEMINI_KEY_BARABASI"},
    "abel":     {"label": "Abel",     "secret_key": "GEMINI_KEY_ABEL"},
}
ACCOUNT_KEYS = list(ACCOUNTS.keys())

# ── Model registry ─────────────────────────────────────────────────────────────
MODELS = {
    "gemini-3.1-flash-image-preview": {
        "label": "Gemini 3.1 Flash Image Preview ⭐",
        "description": "Newest preview — best quality (Feb 2026)",
        "input_text_per_1m":  0.10,
        "input_image_per_1m": 0.10,
        "output_per_1m":      0.40,
        "price_hint":         "$0.045 – $0.15 / image",
    },
    "gemini-3-pro-image-preview": {
        "label": "Gemini 3 Pro Image Preview",
        "description": "Most capable, highest quality",
        "input_text_per_1m":  0.50,
        "input_image_per_1m": 0.50,
        "output_per_1m":      2.00,
        "price_hint":         "$0.13 – $0.24 / image",
    },
    "gemini-2.5-flash-image": {
        "label": "Gemini 2.5 Flash Image (GA)",
        "description": "Production-stable, best value",
        "input_text_per_1m":  0.075,
        "input_image_per_1m": 0.075,
        "output_per_1m":      0.30,
        "price_hint":         "~$0.039 / image",
    },
    "gemini-2.0-flash-preview-image-generation": {
        "label": "Gemini 2.0 Flash (Legacy)",
        "description": "Older preview — most affordable",
        "input_text_per_1m":  0.075,
        "input_image_per_1m": 0.075,
        "output_per_1m":      0.30,
        "price_hint":         "~$0.030 / image",
    },
}

MODEL_KEYS  = list(MODELS.keys())
DEFAULT_IDX = 0  # gemini-3.1-flash-image-preview


# ── Session state init ─────────────────────────────────────────────────────────
# Cost totals persist across "Start New" — only reset on full page reload
for acct in ACCOUNT_KEYS:
    if f"total_spent_{acct}" not in st.session_state:
        st.session_state[f"total_spent_{acct}"] = 0.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def estimate_image_input_tokens(width: int, height: int) -> int:
    """Gemini tiles images at 768 px; each tile = 258 tokens + 85 base."""
    tiles = math.ceil(width / 768) * math.ceil(height / 768)
    return tiles * 258 + 85


def compute_cost(model_key: str, input_text_tok: int, input_img_tok: int, output_tok: int) -> dict:
    cfg = MODELS[model_key]
    text_cost   = (input_text_tok  / 1_000_000) * cfg["input_text_per_1m"]
    img_cost    = (input_img_tok   / 1_000_000) * cfg["input_image_per_1m"]
    output_cost = (output_tok      / 1_000_000) * cfg["output_per_1m"]
    return {
        "input_text_tok":  input_text_tok,
        "input_img_tok":   input_img_tok,
        "output_tok":      output_tok,
        "text_cost":       text_cost,
        "img_cost":        img_cost,
        "output_cost":     output_cost,
        "total":           text_cost + img_cost + output_cost,
    }


def call_gemini(api_key: str, model_key: str, images: list, prompt: str):
    """
    Send one or more images + prompt to Gemini.
    images: list of (bytes, mime_type) tuples, in order.
    Returns (result_image_bytes, result_text, usage_metadata).
    """
    client = genai.Client(api_key=api_key)

    parts = []
    if len(images) > 1:
        hint = ", ".join(
            f"Image {i+1} is the {'first' if i==0 else 'second' if i==1 else 'third' if i==2 else str(i+1)+'th'} image"
            for i in range(len(images))
        )
        parts.append(types.Part.from_text(text=f"[{hint}]"))

    for img_bytes, mime_type in images:
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))

    parts.append(types.Part.from_text(text=prompt))

    response = client.models.generate_content(
        model=model_key,
        contents=parts,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )

    result_bytes = None
    result_text  = None
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            result_bytes = part.inline_data.data
        elif part.text:
            result_text = part.text

    return result_bytes, result_text, response.usage_metadata


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("✨ Gemini Image Editor")
    st.caption("A thin wrapper around Google Gemini's native image editing.")
    st.divider()

    if st.button("🗑️ Start New", use_container_width=True):
        # Preserve cost totals, clear everything else
        saved = {f"total_spent_{a}": st.session_state[f"total_spent_{a}"] for a in ACCOUNT_KEYS}
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.update(saved)
        st.rerun()

    st.divider()

    # ── Account selector ───────────────────────────────────────────────────────
    st.subheader("Account")
    selected_account_idx = st.selectbox(
        "Choose account",
        options=range(len(ACCOUNT_KEYS)),
        format_func=lambda i: ACCOUNTS[ACCOUNT_KEYS[i]]["label"],
        index=0,
        label_visibility="collapsed",
    )
    selected_account = ACCOUNT_KEYS[selected_account_idx]

    # Per-account spend totals
    st.caption(
        f"Barabasi total: **${st.session_state['total_spent_barabasi']:.2f}**  \n"
        f"Abel total: **${st.session_state['total_spent_abel']:.2f}**"
    )

    st.divider()

    # ── Model selector ─────────────────────────────────────────────────────────
    st.subheader("Model")
    selected_idx = st.selectbox(
        "Choose model",
        options=range(len(MODEL_KEYS)),
        format_func=lambda i: MODELS[MODEL_KEYS[i]]["label"],
        index=DEFAULT_IDX,
        label_visibility="collapsed",
    )
    selected_model = MODEL_KEYS[selected_idx]
    cfg = MODELS[selected_model]

    st.caption(f"_{cfg['description']}_")
    st.caption(f"💰 Approx. {cfg['price_hint']}")


# ── Main layout ────────────────────────────────────────────────────────────────

st.header("Gemini Image Editor")

left, right = st.columns(2, gap="large")

with left:
    st.subheader("Input")

    uploaded_files = st.file_uploader(
        "Upload image(s)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="Upload one image, or multiple to reference by number in your prompt (e.g. 'use the person from Image 1 and the building from Image 2').",
    )

    if uploaded_files:
        if len(uploaded_files) == 1:
            st.image(uploaded_files[0], caption="Image 1", use_container_width=True)
        else:
            cols = st.columns(len(uploaded_files))
            for i, (col, f) in enumerate(zip(cols, uploaded_files)):
                col.image(f, caption=f"Image {i+1}", use_container_width=True)

    with st.form("edit_form"):
        prompt = st.text_area(
            "What should change?",
            placeholder=(
                "Single image:  Make the sky look like a sunset\n"
                "Multi-image:   Use the person from Image 1 and the building from Image 2, place them on a beach"
            ),
            height=130,
        )
        submit = st.form_submit_button(
            "✨ Edit Image",
            type="primary",
            use_container_width=True,
        )

with right:
    st.subheader("Result")
    result_slot = st.empty()
    cost_slot   = st.empty()

# ── Processing ─────────────────────────────────────────────────────────────────

if submit and uploaded_files and prompt.strip():
    api_key = st.secrets[ACCOUNTS[selected_account]["secret_key"]]

    images = []
    total_img_tokens = 0
    for f in uploaded_files:
        f.seek(0)
        raw = f.read()
        mime = f.type or "image/jpeg"
        images.append((raw, mime))
        pil = Image.open(io.BytesIO(raw))
        total_img_tokens += estimate_image_input_tokens(*pil.size)

    with st.spinner(f"Editing with **{cfg['label']}** …"):
        try:
            result_bytes, result_text, usage = call_gemini(
                api_key, selected_model, images, prompt.strip()
            )
        except Exception as exc:
            right.error(f"**API error:** {exc}")
            st.stop()

    if result_bytes:
        result_img = Image.open(io.BytesIO(result_bytes))

        with result_slot.container():
            st.image(result_img, caption="Result", use_container_width=True)

            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            st.download_button(
                "⬇️ Download result (PNG)",
                data=buf.getvalue(),
                file_name="edited_result.png",
                mime="image/png",
                use_container_width=True,
            )

        # ── Cost breakdown ────────────────────────────────────────────────────
        total_input    = getattr(usage, "prompt_token_count", 0) or 0
        input_text_tok = max(0, total_input - total_img_tokens)
        output_tok     = getattr(usage, "candidates_token_count", 1290) or 1290

        costs = compute_cost(selected_model, input_text_tok, total_img_tokens, output_tok)

        # Accumulate into the active account's running total
        st.session_state[f"total_spent_{selected_account}"] += costs["total"]

        with cost_slot.container():
            st.divider()
            st.caption(f"**💰 Cost breakdown** — charged to **{ACCOUNTS[selected_account]['label']}**")

            c1, c2, c3, c4 = st.columns(4)
            c1.caption(f"**Input · text**  \n{costs['input_text_tok']:,} tok  \n**${costs['text_cost']:.2f}**")
            c2.caption(f"**Input · image(s)**  \n{costs['input_img_tok']:,} tok  \n**${costs['img_cost']:.2f}**")
            c3.caption(f"**Output · image**  \n{costs['output_tok']:,} tok  \n**${costs['output_cost']:.2f}**")
            c4.caption(f"**This edit**  \n  \n**${costs['total']:.2f}**")
            st.caption(
                "⚠️ Prices are approximate. "
                "Check [Google AI pricing](https://ai.google.dev/pricing) for the latest rates. "
                f"Model: `{selected_model}`"
            )

    else:
        with result_slot.container():
            if result_text:
                st.info(f"The model returned text instead of an image:\n\n> {result_text}")
                st.caption("Try rephrasing your prompt or switching to a different model.")
            else:
                st.error("No image was returned. Try a different prompt or model.")
