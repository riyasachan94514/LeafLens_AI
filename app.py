
from pathlib import Path
import sys

import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
CHECKPOINT = ROOT / "results" / "resnet18_baseline" / "best.pt"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models import build_model
from data import eval_transform


st.set_page_config(
    page_title="LeafLens AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- Styling --------------------

st.markdown(
    """
    <style>
    /* Main dark background */
    .stApp {
        background: #0b1410;
        color: #e6f1e8;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 20, 16, 0.95);
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Default readable text */
    .stApp p,
    .stApp label,
    .stApp li,
    .stApp h2,
    .stApp h3,
    .stApp h4 {
        color: #dce9df;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(120deg, #102b20, #18583a);
        border: 1px solid #286846;
        padding: 2.2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: 2.5rem;
        margin: 0.5rem 0;
    }

    .hero p {
        color: #d2e8d8 !important;
    }

    .eyebrow {
        color: #8ee6a7 !important;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
    }

    /* Information cards */
    .stat {
        background: #14251b;
        border: 1px solid #294b35;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }

    .stat-label {
        color: #a9c5b0 !important;
        font-size: 0.8rem;
    }

    .stat-value {
        color: #9aefb0 !important;
        font-size: 1.35rem;
        font-weight: 700;
    }

    /* Upload area */
    div[data-testid="stFileUploader"] {
        background: #111f17;
        border: 1px solid #294b35;
        border-radius: 14px;
        padding: 1rem;
    }

    div[data-testid="stFileUploader"] section {
        background: #17271e;
        border-color: #345740;
    }

    div[data-testid="stFileUploader"] button {
        color: #e6f1e8;
        border-color: #4d7959;
    }

    /* Dark inputs */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] {
        background: #14251b;
        color: #e6f1e8;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: #14251b;
        border: 1px solid #294b35;
        border-radius: 12px;
        padding: 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: #b6cdbb !important;
    }

    [data-testid="stMetricValue"] {
        color: #9aefb0 !important;
    }

    /* Buttons */
    .stButton button {
        background: #226b40;
        color: #ffffff;
        border: 1px solid #3d8c58;
        border-radius: 10px;
    }

    .stButton button:hover {
        background: #2c8550;
        color: #ffffff;
        border-color: #7ad995;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        background: #17271e;
        border-radius: 12px;
    }

    /* Progress indicators */
    [data-testid="stProgressBar"] > div {
        background: #263b2d;
    }

    [data-testid="stProgressBar"] > div > div {
        background: #65d987;
    }

    /* Dividers and secondary text */
    hr {
        border-color: #294135;
    }

    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: #9cb5a2 !important;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# -------------------- Model loading --------------------
@st.cache_resource
def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    class_names = checkpoint["class_names"]

    model = build_model(
        "resnet18",
        len(class_names),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, class_names, device


# -------------------- Header --------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">PLANT DISEASE RESEARCH LAB</div>
        <h1>🌿 LeafLens AI</h1>
        <p>
            Explore plant leaf classification using a trained ResNet18
            neural network. Upload an image to see the model's prediction.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not CHECKPOINT.is_file():
    st.error(f"Checkpoint not found: {CHECKPOINT}")
    st.stop()

try:
    model, class_names, device = load_model()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()


# -------------------- Model information --------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="stat">
            <div class="stat-label">MODEL ARCHITECTURE</div>
            <div class="stat-value">ResNet18</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="stat">
            <div class="stat-label">SUPPORTED CLASSES</div>
            <div class="stat-value">{len(class_names)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="stat">
            <div class="stat-label">INFERENCE DEVICE</div>
            <div class="stat-value">{str(device).upper()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.write("")

# -------------------- Upload and prediction --------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("📤 Upload a leaf image")
    st.write(
        "Choose a clear plant leaf photo. Supported formats: "
        "JPG, JPEG and PNG."
    )

    uploaded_file = st.file_uploader(
        "Choose image",
        type=["jpg", "jpeg", "png"],
        key="leaf_upload",
    )

    st.caption(
        "Tip: begin with an image from your PlantVillage test dataset."
    )

    if uploaded_file is not None:
        try:
            preview = Image.open(uploaded_file).convert("RGB")
            st.image(
                preview,
                caption="Uploaded leaf image",
                use_container_width=True,
            )
        except Exception:
            st.error("This image could not be opened. Try another file.")


with right:
    st.subheader("🔬 Prediction results")

    if uploaded_file is None:
        st.info("Upload an image to see the prediction here.")
        st.markdown("**How it works**")
        st.write("1. Upload a leaf photo.")
        st.write("2. The trained model analyzes the image.")
        st.write("3. Review the top three predicted classes.")

    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            input_tensor = eval_transform()(image).unsqueeze(0).to(device)

            with torch.inference_mode():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)[0]
                scores, indices = torch.topk(
                    probabilities,
                    k=min(3, len(class_names)),
                )

            predicted_class = class_names[int(indices[0].item())]
            confidence = float(scores[0].item())

            st.success(f"**Predicted class:** {predicted_class}")
            st.metric("Model confidence", f"{confidence:.1%}")

            st.caption(
                "Confidence is a model score, not a guarantee of correctness."
            )

            st.markdown("#### Top 3 predictions")

            for score, index in zip(scores.tolist(), indices.tolist()):
                st.write(f"**{class_names[index]}** — {score:.1%}")
                st.progress(float(score))

        except Exception as exc:
            st.error(f"Could not process the image: {exc}")


# -------------------- Research disclaimer --------------------
st.divider()

st.info(
    "Research prototype: trained on PlantVillage images. Predictions may "
    "be unreliable on real-world field photos and unfamiliar conditions."
)

st.caption(
    "LeafLens AI · Plant disease classification research demo · "
    "Existing trained checkpoint; no retraining."
)
