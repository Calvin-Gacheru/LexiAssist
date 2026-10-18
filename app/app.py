
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

from model import load_checkpoint
from preprocess import build_transform, display_image
from explain import gradcam_overlay

st.set_page_config(page_title="LexiAssist", page_icon="✍️", layout="centered")

@st.cache_resource
def get_model():
    return load_checkpoint(Path(__file__).parent / "lexiassist.pth")

model, meta = get_model()
threshold = float(meta.get("threshold", 0.5))

st.title("LexiAssist")
st.caption("A handwriting-based screening indicator for possible dyslexia-related patterns")

st.warning(
    "LexiAssist is an educational screening aid, **not a medical or diagnostic tool**. "
    "Dyslexia can only be diagnosed by a qualified specialist. Many people with dyslexia "
    "write neatly, and many people without it do not."
)
st.info("Privacy: your image is processed in memory only and is never saved or shared.")

consent = st.checkbox("I understand this is not a diagnosis and I have permission to upload this handwriting.")
file = st.file_uploader("Upload a clear photo or scan of handwriting", type=["jpg", "jpeg", "png"])

if file is not None and not consent:
    st.write("Please tick the box above to continue.")

if file is not None and consent:
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    gray = np.asarray(img.convert("L"))

    if min(img.size) < 150:
        st.warning("The image is quite small. A larger, sharper photo will give a more reliable result.")
    if gray.std() < 8:
        st.error("This image looks blank or almost uniform. Please upload a photo of handwriting.")
        st.stop()

    st.image(img, caption="Uploaded image", width="stretch")

    x = build_transform(meta["img_size"], train=False)(img).unsqueeze(0)
    with torch.no_grad():
        prob = torch.softmax(model(x), 1)[0, 1].item()

    st.subheader("Result")
    st.metric("Estimated likelihood of dyslexia-related handwriting patterns", f"{prob:.0%}")
    st.progress(min(max(prob, 0.0), 1.0))

    if prob >= threshold:
        st.warning(
            "This handwriting shows **possible indicators** that are sometimes associated with dyslexia. "
            "This is not a diagnosis. If you have concerns about reading or writing, consider speaking "
            "with a teacher, GP or an educational psychologist."
        )
    else:
        st.success(
            "This handwriting shows **few of the indicators** the model looks for. This does not rule out "
            "dyslexia. If you have concerns, a specialist assessment is still the right next step."
        )

    st.subheader("What did the model look at?")
    overlay = gradcam_overlay(model, x, display_image(img, meta["img_size"]), target_class=1)
    c1, c2 = st.columns(2)
    c1.image(display_image(img, meta["img_size"]), caption="Processed input", width="stretch")
    c2.image(overlay, caption="Attention heatmap (warmer = more influence)", width="stretch")
    st.caption("If the heatmap highlights the paper edges or shadows instead of the writing, treat the result with extra caution.")

with st.expander("About this model and its limits"):
    st.write(
        f"Trained on: {meta.get('trained_on', 'n/a')}. The training data is small and comes from a "
        "limited group of children, so results may not generalise to other ages, languages, pens or "
        "photo conditions. Decision threshold: " + f"{threshold:.2f}" + "."
    )
