import io
import json
import math
from hashlib import sha1
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates

APP_DIR = Path(__file__).parent
COSTS_FILE = APP_DIR / "costs.json"
RESULTS_DIR = APP_DIR / "workspace"
RESULT_IMAGE_FILE = RESULTS_DIR / "latest_result.png"
PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"

ACCOUNTS = {
    "barabasi": {"label": "Barabasi", "secret_key": "GEMINI_KEY_BARABASI"},
    "abel": {"label": "Abel", "secret_key": "GEMINI_KEY_ABEL"},
}
ACCOUNT_KEYS = list(ACCOUNTS.keys())

MODELS = {
    "gemini-3.1-flash-image-preview": {
        "label": "Gemini 3.1 Flash Image Preview",
        "description": "Fast image generation preview with Gemini 3.1 pricing",
        "input_text_per_1m": 0.50,
        "input_image_per_1m": 0.50,
        "output_per_1m": 60.00,
        "price_hint": "About 0.045 USD for 512x512.\nAbout 0.067 USD for 1024x1024.\nAbout 0.101 USD for 2048x2048.\nAbout 0.151 USD for 4096x4096.",
        "default_output_tokens": 1120,
    },
    "gemini-3-pro-image-preview": {
        "label": "Gemini 3 Pro Image Preview",
        "description": "Current highest-quality image preview",
        "input_text_per_1m": 2.00,
        "input_image_per_1m": 2.00,
        "output_per_1m": 120.00,
        "price_hint": "Input images cost about 0.0011 USD each.\nOutput is about 0.134 USD for 1024x1024 to 2048x2048.\nOutput is about 0.24 USD for 4096x4096.",
        "default_output_tokens": 1120,
    },
    "gemini-2.5-flash-image": {
        "label": "Gemini 2.5 Flash Image (GA)",
        "description": "Current production image model",
        "input_text_per_1m": 0.30,
        "input_image_per_1m": 0.30,
        "output_per_1m": 30.00,
        "price_hint": "About 0.039 USD per generated image up to 1024x1024.",
        "default_output_tokens": 1290,
    },
}
MODEL_KEYS = list(MODELS.keys())
DEFAULT_IDX = 0


def load_costs() -> dict:
    """Read persisted totals from disk, or return zeroes if the file does not exist."""
    if COSTS_FILE.exists():
        try:
            return json.loads(COSTS_FILE.read_text())
        except Exception:
            pass
    return {account_key: 0.0 for account_key in ACCOUNT_KEYS}


def save_costs(totals: dict) -> None:
    """Persist the running totals to disk."""
    COSTS_FILE.write_text(json.dumps(totals, indent=2))


def clear_workspace_result() -> None:
    """Delete the last generated workspace image if it exists."""
    try:
        RESULT_IMAGE_FILE.unlink()
    except FileNotFoundError:
        pass


def save_workspace_result(image_bytes: bytes) -> str:
    """Persist the latest generated image inside the repo workspace."""
    RESULTS_DIR.mkdir(exist_ok=True)
    RESULT_IMAGE_FILE.write_bytes(image_bytes)
    return RESULT_IMAGE_FILE.relative_to(APP_DIR).as_posix()


def estimate_image_input_tokens(model_key: str, width: int, height: int) -> int:
    """Approximate Gemini image input tokens using the current docs."""
    if model_key == "gemini-3-pro-image-preview":
        return 560

    if width <= 384 and height <= 384:
        return 258

    crop_unit = max(1, math.floor(min(width, height) / 1.5))
    return math.ceil(width / crop_unit) * math.ceil(height / crop_unit) * 258


def estimate_output_tokens(model_key: str, width: int, height: int, usage_output_tok: int) -> int:
    """Use API usage when available, otherwise fall back to documented image token sizes."""
    if usage_output_tok > 0:
        return usage_output_tok

    max_dimension = max(width, height)

    if model_key == "gemini-3.1-flash-image-preview":
        if max_dimension <= 512:
            return 747
        if max_dimension <= 1024:
            return 1120
        if max_dimension <= 2048:
            return 1680
        return 2520

    if model_key == "gemini-3-pro-image-preview":
        return 2000 if max_dimension > 2048 else 1120

    return MODELS[model_key].get("default_output_tokens", 1290)


def compute_cost(model_key: str, input_text_tok: int, input_img_tok: int, output_tok: int) -> dict:
    cfg = MODELS[model_key]
    if not cfg.get("pricing_available", True):
        return {}

    text_cost = (input_text_tok / 1_000_000) * cfg["input_text_per_1m"]
    img_cost = (input_img_tok / 1_000_000) * cfg["input_image_per_1m"]
    output_cost = (output_tok / 1_000_000) * cfg["output_per_1m"]
    return {
        "input_text_tok": input_text_tok,
        "input_img_tok": input_img_tok,
        "output_tok": output_tok,
        "text_cost": text_cost,
        "img_cost": img_cost,
        "output_cost": output_cost,
        "total": text_cost + img_cost + output_cost,
    }


def ordinal(n: int) -> str:
    """Return a human-friendly ordinal like 1st, 2nd, 3rd."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def build_image_id(image_bytes: bytes, image_index: int) -> str:
    """Create a stable image identifier from the uploaded bytes."""
    return f"image-{image_index}-{sha1(image_bytes).hexdigest()[:12]}"


def prepare_images(image_sources: list[dict]) -> list[dict]:
    """Load image sources once so previews, clicks, and submission use the same data."""
    prepared = []
    for image_index, image_source in enumerate(image_sources, start=1):
        raw = image_source["raw"]
        pil_image = Image.open(io.BytesIO(raw))
        pil_image.load()
        prepared.append(
            {
                "id": build_image_id(raw, image_index),
                "index": image_index,
                "name": image_source.get("name") or f"image_{image_index}",
                "mime_type": image_source.get("mime_type") or "image/jpeg",
                "raw": raw,
                "pil_image": pil_image,
                "size": pil_image.size,
            }
        )
    return prepared


def get_uploaded_image_sources(uploaded_files) -> list[dict]:
    """Normalize uploaded files into the app's internal image-source format."""
    return [
        {
            "name": uploaded_file.name or f"image_{image_index}",
            "mime_type": uploaded_file.type or "image/jpeg",
            "raw": uploaded_file.getvalue(),
        }
        for image_index, uploaded_file in enumerate(uploaded_files, start=1)
    ]


def stage_latest_result_as_input() -> None:
    """Make the latest generated image the next input image for a fresh task."""
    latest_result = st.session_state.get("latest_result")
    if not latest_result:
        return

    st.session_state["staged_input_images"] = [
        {
            "name": "edited_result.png",
            "mime_type": "image/png",
            "raw": latest_result["png_bytes"],
        }
    ]
    st.session_state["image_points"] = {}
    st.session_state["image_point_clicks"] = {}
    st.session_state["reset_edit_prompt"] = True
    st.session_state["uploader_version"] = st.session_state.get("uploader_version", 0) + 1


def sync_point_state(active_image_ids: list[str]) -> None:
    """Keep per-image point state aligned with the images currently uploaded."""
    point_store = st.session_state.setdefault("image_points", {})
    click_store = st.session_state.setdefault("image_point_clicks", {})
    active_ids = set(active_image_ids)

    for stale_image_id in list(point_store.keys()):
        if stale_image_id not in active_ids:
            del point_store[stale_image_id]

    for stale_image_id in list(click_store.keys()):
        if stale_image_id not in active_ids:
            del click_store[stale_image_id]


def translate_click_to_image_point(click_value: dict | None, image_size: tuple[int, int]) -> dict | None:
    """Convert a click from preview space into the original uploaded image coordinates."""
    if not click_value or "x" not in click_value or "y" not in click_value:
        return None

    preview_width = click_value.get("width") or 0
    preview_height = click_value.get("height") or 0
    if preview_width <= 0 or preview_height <= 0:
        return None

    image_width, image_height = image_size
    x_ratio = min(max(click_value["x"] / preview_width, 0.0), 1.0)
    y_ratio = min(max(click_value["y"] / preview_height, 0.0), 1.0)
    x = round(x_ratio * max(image_width - 1, 0))
    y = round(y_ratio * max(image_height - 1, 0))

    return {
        "x": x,
        "y": y,
        "x_normalized": round(x_ratio, 4),
        "y_normalized": round(y_ratio, 4),
    }


def register_point_click(image_id: str, click_value: dict | None, image_size: tuple[int, int]) -> bool:
    """Persist a new point only once per click event."""
    if not click_value:
        return False

    unix_time = click_value.get("unix_time")
    if unix_time is None:
        return False

    click_store = st.session_state.setdefault("image_point_clicks", {})
    if click_store.get(image_id) == unix_time:
        return False

    point = translate_click_to_image_point(click_value, image_size)
    if not point:
        return False

    st.session_state.setdefault("image_points", {}).setdefault(image_id, []).append(point)
    click_store[image_id] = unix_time
    return True


def render_point_preview(source_image: Image.Image, points: list[dict]) -> Image.Image:
    """Draw numbered reference markers on top of the uploaded image for clicking."""
    preview = source_image.convert("RGBA").copy()
    if not points:
        return preview

    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    radius = max(10, min(20, round(min(preview.size) * 0.02)))
    outline_width = max(2, radius // 4)

    for point_number, point in enumerate(points, start=1):
        x = point["x"]
        y = point["y"]
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(231, 76, 60, 220),
            outline=(255, 255, 255, 255),
            width=outline_width,
        )

        label = str(point_number)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_x = x - ((right - left) / 2)
        text_y = y - ((bottom - top) / 2)
        draw.text((text_x, text_y), label, fill=(255, 255, 255, 255), font=font)

    return preview


def build_point_reference_text(prepared_images: list[dict]) -> str | None:
    """Serialize the clicked points into a structured block for Gemini."""
    point_store = st.session_state.get("image_points", {})
    point_map = []

    for image in prepared_images:
        points = point_store.get(image["id"], [])
        if not points:
            continue

        point_map.append(
            {
                "image_number": image["index"],
                "image_ordinal": ordinal(image["index"]),
                "file_name": image["name"],
                "width": image["size"][0],
                "height": image["size"][1],
                "points": [
                    {
                        "point_number": point_number,
                        "point_ordinal": ordinal(point_number),
                        "x": point["x"],
                        "y": point["y"],
                        "x_normalized": point["x_normalized"],
                        "y_normalized": point["y_normalized"],
                    }
                    for point_number, point in enumerate(points, start=1)
                ],
            }
        )

    if not point_map:
        return None

    return (
        "[Reference point map]\n"
        "The user may refer to these clicked locations as 'point N on image M', "
        "'Nth position on the Mth image', or similar phrasing.\n"
        "Use this JSON as the exact coordinate map for the original uploaded images:\n"
        f"{json.dumps(point_map, indent=2)}"
    )


def call_gemini(api_key: str, model_key: str, images: list, prompt: str, point_reference_text: str | None = None):
    """
    Send one or more images plus a prompt to Gemini.
    images: list of (bytes, mime_type) tuples, in order.
    Returns (result_image_bytes, result_text, usage_metadata).
    """
    client = genai.Client(api_key=api_key)

    parts = []
    if len(images) > 1:
        hint = ", ".join(
            f"Image {i + 1} is the {ordinal(i + 1)} image"
            for i in range(len(images))
        )
        parts.append(types.Part.from_text(text=f"[{hint}]"))

    for img_bytes, mime_type in images:
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))

    if point_reference_text:
        parts.append(types.Part.from_text(text=point_reference_text))

    parts.append(types.Part.from_text(text=prompt))

    response = client.models.generate_content(
        model=model_key,
        contents=parts,
        config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
    )

    candidates = getattr(response, "candidates", None) or []
    result_bytes = None
    text_parts = []

    for candidate in candidates:
        content = getattr(candidate, "content", None)
        candidate_parts = getattr(content, "parts", None) or []
        for part in candidate_parts:
            inline_data = getattr(part, "inline_data", None)
            inline_bytes = getattr(inline_data, "data", None) if inline_data is not None else None
            if inline_bytes:
                result_bytes = inline_bytes

            part_text = getattr(part, "text", None)
            if part_text:
                text_parts.append(part_text)

    result_text = "\n".join(text_parts).strip() or None
    if result_bytes or result_text:
        return result_bytes, result_text, getattr(response, "usage_metadata", None)

    fallback_text = getattr(response, "text", None)
    if fallback_text:
        return None, fallback_text, getattr(response, "usage_metadata", None)

    finish_reasons = [
        str(getattr(candidate, "finish_reason", "unknown"))
        for candidate in candidates
        if getattr(candidate, "finish_reason", None) is not None
    ]
    reason_suffix = f" Finish reason: {', '.join(finish_reasons)}." if finish_reasons else ""
    raise RuntimeError(f"Gemini returned no image or text content.{reason_suffix}")


def render_saved_result(result_slot, cost_slot, show_empty_hint: bool = True) -> None:
    """Render the latest successful result so it survives reruns."""
    latest_result = st.session_state.get("latest_result")

    if not latest_result:
        if show_empty_hint:
            with result_slot.container():
                st.caption("The latest generated image stays here until the next successful edit or `Start New`.")
        return

    with result_slot.container():
        st.image(latest_result["png_bytes"], caption="Result", use_container_width=True)
        download_col, reuse_col = st.columns(2)
        download_col.download_button(
            "Download result (PNG)",
            data=latest_result["png_bytes"],
            file_name="edited_result.png",
            mime="image/png",
            use_container_width=True,
            key="download_latest_result",
        )
        if reuse_col.button(
            "Use Result As New Input",
            use_container_width=True,
            key="reuse_latest_result",
        ):
            stage_latest_result_as_input()
            st.rerun()
        st.caption(
            f"Saved in the workspace at `{latest_result['workspace_path']}` "
            "until the next successful edit or `Start New`."
        )

    costs = latest_result.get("costs")
    if not costs:
        pricing_note = latest_result.get("pricing_note")
        if pricing_note:
            with cost_slot.container():
                st.divider()
                st.caption(pricing_note)
        return

    with cost_slot.container():
        st.divider()
        st.caption(
            f"**Cost breakdown** - charged to **{ACCOUNTS[latest_result['account_key']]['label']}**"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.caption(f"**Input - text**  \n{costs['input_text_tok']:,} tok  \n**${costs['text_cost']:.2f}**")
        c2.caption(f"**Input - image(s)**  \n{costs['input_img_tok']:,} tok  \n**${costs['img_cost']:.2f}**")
        c3.caption(f"**Output - image**  \n{costs['output_tok']:,} tok  \n**${costs['output_cost']:.2f}**")
        c4.caption(f"**This edit**  \n  \n**${costs['total']:.2f}**")
        st.caption(
            "Prices are approximate. "
            f"Check [Google AI pricing]({PRICING_URL}) for the latest rates. "
            f"Model: `{latest_result['model_key']}`"
        )


st.set_page_config(
    page_title="Gemini Image Editor",
    page_icon="*",
    layout="wide",
)

# Seed from disk once per browser session so totals survive app restarts.
if "costs_loaded" not in st.session_state:
    persisted = load_costs()
    for account_key in ACCOUNT_KEYS:
        st.session_state[f"total_spent_{account_key}"] = persisted.get(account_key, 0.0)
    st.session_state["costs_loaded"] = True

if "latest_result" not in st.session_state:
    clear_workspace_result()
    st.session_state["latest_result"] = None

if "selected_account_idx" not in st.session_state or st.session_state["selected_account_idx"] >= len(ACCOUNT_KEYS):
    st.session_state["selected_account_idx"] = 0

if "selected_model_idx" not in st.session_state or st.session_state["selected_model_idx"] >= len(MODEL_KEYS):
    st.session_state["selected_model_idx"] = DEFAULT_IDX

if "uploader_version" not in st.session_state:
    st.session_state["uploader_version"] = 0

if "staged_input_images" not in st.session_state:
    st.session_state["staged_input_images"] = []

if "edit_prompt" not in st.session_state:
    st.session_state["edit_prompt"] = ""

if st.session_state.pop("reset_edit_prompt", False):
    st.session_state["edit_prompt"] = ""

with st.sidebar:
    st.title("Gemini Image Editor")
    st.caption("A thin wrapper around Google Gemini's native image editing.")
    st.divider()

    if st.button("Start New", use_container_width=True):
        clear_workspace_result()
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        persisted = load_costs()
        for account_key in ACCOUNT_KEYS:
            st.session_state[f"total_spent_{account_key}"] = persisted.get(account_key, 0.0)
        st.session_state["costs_loaded"] = True
        st.session_state["latest_result"] = None
        st.session_state["selected_account_idx"] = 0
        st.session_state["selected_model_idx"] = DEFAULT_IDX
        st.rerun()

    st.divider()

    st.subheader("Account")
    selected_account_idx = st.selectbox(
        "Choose account",
        options=range(len(ACCOUNT_KEYS)),
        format_func=lambda i: ACCOUNTS[ACCOUNT_KEYS[i]]["label"],
        label_visibility="collapsed",
        key="selected_account_idx",
    )
    selected_account = ACCOUNT_KEYS[selected_account_idx]

    st.caption(f"Barabasi total: ${st.session_state['total_spent_barabasi']:.2f}")
    st.caption(f"Abel total: ${st.session_state['total_spent_abel']:.2f}")

    st.divider()

    st.subheader("Model")
    selected_model_idx = st.selectbox(
        "Choose model",
        options=range(len(MODEL_KEYS)),
        format_func=lambda i: MODELS[MODEL_KEYS[i]]["label"],
        label_visibility="collapsed",
        key="selected_model_idx",
    )
    selected_model = MODEL_KEYS[selected_model_idx]
    cfg = MODELS[selected_model]

    st.write(cfg["description"])
    st.text(cfg["price_hint"])
    if not cfg.get("pricing_available", True):
        st.write("Current Google pricing marks image generation for this legacy model as unavailable.")

st.header("Gemini Image Editor")

left, right = st.columns(2, gap="large")
prepared_images = []
point_reference_text = None

with left:
    st.subheader("Input")

    uploaded_files = st.file_uploader(
        "Upload image(s)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key=f"uploaded_files_{st.session_state['uploader_version']}",
        help=(
            "Upload one image, or multiple to reference by number in your prompt "
            "(for example: 'use the person from Image 1 and the building from Image 2')."
        ),
    )

    image_sources = []
    if uploaded_files:
        image_sources = get_uploaded_image_sources(uploaded_files)
        st.session_state["staged_input_images"] = []
    elif st.session_state["staged_input_images"]:
        image_sources = st.session_state["staged_input_images"]
        st.caption("Using the latest generated result as the current input image.")

    prepared_images = prepare_images(image_sources) if image_sources else []
    sync_point_state([image["id"] for image in prepared_images])

    if prepared_images:
        st.caption(
            "Click on each image to drop numbered reference points. "
            "Those exact positions will be sent to Gemini with your prompt."
        )

        image_views = st.tabs([f"Image {image['index']}" for image in prepared_images])
        for image, image_view in zip(prepared_images, image_views):
            with image_view:
                points = st.session_state["image_points"].get(image["id"], [])
                click_value = streamlit_image_coordinates(
                    render_point_preview(image["pil_image"], points),
                    key=f"image_clicks_{image['id']}",
                    use_column_width="always",
                    cursor="crosshair",
                )
                if register_point_click(image["id"], click_value, image["size"]):
                    st.rerun()

                points = st.session_state["image_points"].get(image["id"], [])
                st.caption(f"Original size: {image['size'][0]} x {image['size'][1]}")
                if points:
                    for point_number, point in enumerate(points, start=1):
                        st.caption(
                            f"Point {point_number}: x={point['x']}, y={point['y']} "
                            f"({point['x_normalized']:.3f}, {point['y_normalized']:.3f})"
                        )
                else:
                    st.caption("No points added yet.")

                undo_col, clear_col = st.columns(2)
                if undo_col.button(
                    "Undo last point",
                    key=f"undo_point_{image['id']}",
                    use_container_width=True,
                    disabled=not points,
                ):
                    st.session_state["image_points"][image["id"]] = points[:-1]
                    st.rerun()

                if clear_col.button(
                    "Clear points",
                    key=f"clear_points_{image['id']}",
                    use_container_width=True,
                    disabled=not points,
                ):
                    st.session_state["image_points"][image["id"]] = []
                    st.rerun()

        point_reference_text = build_point_reference_text(prepared_images)
        if point_reference_text:
            st.caption(
                "Prompt example: `Change the animal at point 2 on image 3.` "
                "The coordinate map below is added automatically."
            )
            with st.expander("Point map sent to Gemini"):
                st.code(point_reference_text, language="text")

    with st.form("edit_form"):
        prompt = st.text_area(
            "What should change?",
            placeholder=(
                "Single image:  Make the sky look like a sunset\n"
                "With points:   Change the animal at point 2 on image 3\n"
                "Multi-image:   Use the person from Image 1 and the building from Image 2, place them on a beach"
            ),
            height=130,
            key="edit_prompt",
        )
        submit = st.form_submit_button(
            "Edit Image",
            type="primary",
            use_container_width=True,
        )

with right:
    st.subheader("Result")
    notice_slot = st.empty()
    result_slot = st.empty()
    cost_slot = st.empty()

notice_type = None
notice_message = None
notice_caption = None

if submit:
    if not prepared_images:
        notice_type = "warning"
        notice_message = "Upload at least one image before editing."
    elif not prompt.strip():
        notice_type = "warning"
        notice_message = "Enter a prompt before editing."
    else:
        api_key = st.secrets[ACCOUNTS[selected_account]["secret_key"]]

        images = [(image["raw"], image["mime_type"]) for image in prepared_images]
        total_img_tokens = sum(
            estimate_image_input_tokens(selected_model, image["size"][0], image["size"][1])
            for image in prepared_images
        )

        try:
            with st.spinner(f"Editing with **{cfg['label']}** ..."):
                result_bytes, result_text, usage = call_gemini(
                    api_key,
                    selected_model,
                    images,
                    prompt.strip(),
                    point_reference_text=point_reference_text,
                )
        except Exception as exc:
            exc_str = str(exc)
            if "503" in exc_str or "UNAVAILABLE" in exc_str or "Deadline" in exc_str:
                notice_type = "error"
                notice_message = "Google's servers took too long to respond and the request timed out."
                notice_caption = (
                    "**What you can try:**\n"
                    "- **Try again** — this is usually a temporary overload on Google's side\n"
                    "- **Use a smaller image** — large images take longer to process and are more likely to time out\n"
                    "- **Switch to Gemini 2.5 Flash Image** in the sidebar — it's the stable production model and handles load better than the preview models\n"
                    "- **Try a different model** if one preview model keeps failing"
                )
            else:
                notice_type = "error"
                notice_message = f"API error: {exc}"
        else:
            if result_bytes:
                result_img = Image.open(io.BytesIO(result_bytes))
                result_img.load()

                buffer = io.BytesIO()
                result_img.save(buffer, format="PNG")
                png_bytes = buffer.getvalue()
                workspace_path = save_workspace_result(png_bytes)

                total_input = getattr(usage, "prompt_token_count", 0) or 0
                input_text_tok = max(0, total_input - total_img_tokens)
                output_tok = estimate_output_tokens(
                    selected_model,
                    result_img.size[0],
                    result_img.size[1],
                    getattr(usage, "candidates_token_count", 0) or 0,
                )
                costs = compute_cost(selected_model, input_text_tok, total_img_tokens, output_tok)
                pricing_note = None
                if not costs:
                    pricing_note = (
                        "Current Google pricing marks image generation for this model as unavailable, "
                        "so no estimated charge is shown."
                    )

                st.session_state["latest_result"] = {
                    "png_bytes": png_bytes,
                    "workspace_path": workspace_path,
                    "model_key": selected_model,
                    "account_key": selected_account,
                    "costs": costs,
                    "pricing_note": pricing_note,
                }

                if costs:
                    st.session_state[f"total_spent_{selected_account}"] += costs["total"]
                    save_costs(
                        {account_key: st.session_state[f"total_spent_{account_key}"] for account_key in ACCOUNT_KEYS}
                    )
            elif result_text:
                notice_type = "info"
                notice_message = f"The model returned text instead of an image:\n\n> {result_text}"
                notice_caption = "Try rephrasing your prompt or switching to a different model."
            else:
                notice_type = "error"
                notice_message = "No image was returned. Try a different prompt or model."

if notice_type:
    with notice_slot.container():
        if notice_type == "info":
            st.info(notice_message)
        elif notice_type == "warning":
            st.warning(notice_message)
        else:
            st.error(notice_message)

        if notice_caption:
            st.caption(notice_caption)

render_saved_result(result_slot, cost_slot, show_empty_hint=not bool(notice_type))
