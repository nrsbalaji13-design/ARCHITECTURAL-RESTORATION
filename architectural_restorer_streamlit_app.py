"""
Architectural Restorer — Streamlit App
--------------------------------------
Purpose
  - Upload a photo of a damaged/broken object or place.
  - The app will (1) attempt to auto-detect damage (via simple edge/crack detection), (2) allow manual mask touch-up with an in-browser drawing canvas, (3) run image inpainting to propose restored images, and (4) classify likely architectural styles for the scene and offer alternate restoration "ideas" (different style variants).

What this package provides
  - A single-file Streamlit app that you can run locally: `streamlit run architectural_restorer_streamlit_app.py`
  - Uses a combination of lightweight automatic damage-detection heuristics, user-editable masks (drawable canvas), CLIP-based style-classification, and Stable Diffusion inpainting (via diffusers).

Requirements
  - Python 3.10+ recommended.
  - A machine with a recent GPU (NVIDIA + CUDA) is strongly recommended for reasonable speed when using the inpainting model. CPU will work but will be slow.

Pip install (example):
  python -m pip install --upgrade pip
  pip install streamlit pillow numpy opencv-python torch torchvision torchvision --upgrade
  pip install transformers diffusers accelerate safetensors ftfy einops
  pip install streamlit-drawable-canvas
  pip install git+https://github.com/facebookresearch/segment-anything.git@main

  NOTE: If you can't install the SAM package / it gives problems, the app will still run using only the automatic heuristic + manual mask.

Models used (downloaded automatically by Hugging Face diffusers & transformers):
  - Stable Diffusion inpainting (example model id: "runwayml/stable-diffusion-inpainting") — you can change to another compatible inpainting model.
  - CLIP (via transformers) for image-text similarity.

How it works (high level)
  1. Upload image.
  2. App runs a simple damage-detector (Canny edges + morphological ops + heuristics) to propose a mask of damaged regions.
  3. You can refine the mask using the drawing canvas (paint over areas to include/exclude from restoration).
  4. Choose one or more restoration styles (e.g., "Original", "Gothic", "Modern Minimalist", "Mughal-inspired", etc.), or let the app auto-suggest top-N styles using CLIP classification.
  5. For each chosen style, the app generates 1..k inpainted variants by calling Stable Diffusion inpainting with style-specific prompts.
  6. The app shows results for comparison and lets you download images.

Limitations & Notes
  - Quality depends on the inpainting model and compute available.
  - Fine architectural geometry (columns, load-bearing elements) may need manual architectural expertise for real-world restoration — this tool is a design/idea generator and not a structural engineering solution.
  - Respect copyright and privacy for uploaded images.

-----
# Below is the app code. Save as `architectural_restorer_streamlit_app.py` and run with: streamlit run architectural_restorer_streamlit_app.py

"""

import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import io
import os
import cv2
import torch
from torchvision.transforms import functional as TF

# Optional imports (try/except so app runs even if some heavy libs fail)
try:
    from transformers import CLIPProcessor, CLIPModel
    has_clip = True
except Exception:
    has_clip = False

try:
    from diffusers import StableDiffusionInpaintPipeline
    has_diffusers = True
except Exception:
    has_diffusers = False

try:
    from streamlit_drawable_canvas import st_canvas
    has_canvas = True
except Exception:
    has_canvas = False

st.set_page_config(page_title="Architectural Restorer", layout="wide")
st.title("Architectural Restorer — restore, reimagine, classify")

# Sidebar: model settings
st.sidebar.header("Settings")
use_gpu = st.sidebar.checkbox("Use GPU if available", value=True)
model_choice = st.sidebar.selectbox("Inpainting model (diffusers)", ["runwayml/stable-diffusion-inpainting", "stabilityai/stable-diffusion-2-inpainting"], index=0)
num_variants = st.sidebar.slider("Variants per style", 1, 4, 2)

device = "cuda" if (torch.cuda.is_available() and use_gpu) else "cpu"

# Load CLIP if available
@st.cache_resource
def load_clip():
    if not has_clip:
        return None, None
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    return model, proc

clip_model, clip_proc = load_clip()

# Load inpainting pipeline lazily
@st.cache_resource
def load_inpaint_pipeline(model_id):
    if not has_diffusers:
        return None
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    if device=="cuda":
        pipe = pipe.to(device)
    return pipe

inpaint_pipe = None

# Architectural style list (textual labels & example descriptors)
ARCH_STYLES = {
    "Original / Neutral": "Restore the missing part to match the original appearance, neutral materials and textures.",
    "Gothic": "Gothic architecture: pointed arches, ribbed vaults, ornate stone tracery, vertical emphasis.",
    "Baroque": "Baroque: dramatic forms, rich ornamentation, curved shapes, elaborate details.",
    "Renaissance": "Renaissance architecture: symmetry, classical columns, pediments, harmonious proportions.",
    "Modernist": "Modern architecture: minimal lines, glass and steel, simple geometric forms.",
    "Brutalist": "Brutalist style: raw concrete surfaces, monolithic blocks, strong sculptural shapes.",
    "Victorian": "Victorian style: decorative trims, bay windows, textured walls, historic ornamentation.",
    "Mughal / Indo-Islamic": "Mughal architecture: domes, arches, intricate jali patterns, ornamental tilework.",
    "Dravidian / South Indian": "Dravidian temple architecture: tiered vimana, carved pillars, stone motifs.",
    "Contemporary Minimalist": "Contemporary minimalist: clean surfaces, neutral palette, subtle textures." 
}

st.markdown("Upload a photo of the damaged scene (exterior, interior, object). The app will propose a damage mask — refine it if needed — then generate restoration variants and classify likely architectural styles.")

# Upload
uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"], accept_multiple_files=False)

if uploaded is None:
    st.info("Upload an image to begin — try a photo of a broken wall, damaged carving, or missing tilework.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
orig_w, orig_h = img.size
st.sidebar.write(f"Image size: {orig_w} x {orig_h}")

# Auto-detect damaged regions (simple heuristic using edges and morphology)
@st.cache_data
def detect_damage_pil(pil_img):
    # Convert to grayscale & use Canny
    arr = np.array(pil_img.convert('L'))
    # Resize for speed if large
    scale = 1.0
    h, w = arr.shape
    if max(h,w) > 1600:
        scale = 1600 / max(h,w)
        arr = cv2.resize(arr, (int(w*scale), int(h*scale)))
    edges = cv2.Canny(arr, 50, 150)
    # Dilate edges to make mask
    kernel = np.ones((9,9), np.uint8)
    dil = cv2.dilate(edges, kernel, iterations=2)
    # Blur and threshold
    blur = cv2.GaussianBlur(dil, (15,15), 0)
    _, th = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
    # Upscale back if scaled
    if scale != 1.0:
        th = cv2.resize(th, (w,h), interpolation=cv2.INTER_NEAREST)
    # Convert to PIL mask
    mask = Image.fromarray(th).convert("L")
    return mask

mask_proposal = detect_damage_pil(img)

# Show original and proposed mask side-by-side
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original")
    st.image(img, use_column_width=True)
with col2:
    st.subheader("Proposed damage mask (editable)")
    st.image(mask_proposal, use_column_width=True)

# Let user edit mask using drawable canvas if available
st.markdown("### Edit mask (optional)")
mask = None
if has_canvas:
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=20,
        stroke_color="#ffffff",
        background_image=mask_proposal,
        height=orig_h//2 if orig_h>800 else orig_h,
        width=orig_w//2 if orig_w>800 else orig_w,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None:
        # canvas returns RGBA numpy array
        arr = canvas_result.image_data.astype('uint8')
        # Convert to grayscale mask
        gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
        _, m = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask = Image.fromarray(m).convert("L")
else:
    st.write("Drawable canvas not available — using proposed mask. (To enable painting, pip install streamlit-drawable-canvas)")

if mask is None:
    mask = mask_proposal

# Allow user to invert or clear mask
col_a, col_b, col_c = st.columns(3)
with col_a:
    if st.button("Invert mask"):
        mask = ImageOps.invert(mask)
with col_b:
    if st.button("Clear mask"):
        blank = Image.new('L', mask.size, 0)
        mask = blank
with col_c:
    if st.button("Reset to proposal"):
        mask = mask_proposal

st.image(mask, caption="Final mask", use_column_width=False)

# Style suggestions using CLIP
st.markdown("### Architectural style suggestions")
num_suggest = st.slider("Number of suggested styles", 1, 6, 3)

@st.cache_data
def clip_suggest_styles(image_pil, topk=3):
    if clip_model is None or clip_proc is None:
        return list(ARCH_STYLES.keys())[:topk]
    inputs = clip_proc(text=list(ARCH_STYLES.values()), images=image_pil, return_tensors="pt", padding=True)
    # Move tensors to device
    for k, v in inputs.items():
        if hasattr(v, 'to'):
            inputs[k] = v.to(device)
    with torch.no_grad():
        image_embeds = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
        text_embeds = clip_model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    # normalize
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    sims = (image_embeds @ text_embeds.T).squeeze(0).cpu().tolist()
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    top = [list(ARCH_STYLES.keys())[i] for i, _ in ranked[:topk]]
    return top

suggested = clip_suggest_styles(img, topk=num_suggest)
st.write("Top style suggestions:", suggested)

# Let user pick styles to generate
st.markdown("### Choose styles to generate")
style_choices = st.multiselect("Styles", options=list(ARCH_STYLES.keys()), default=suggested)
if not style_choices:
    st.warning("Select at least one style to generate outputs.")
    st.stop()

# Load inpainting pipeline on demand
with st.spinner("Loading inpainting model (may take time on first run)..."):
    if has_diffusers and inpaint_pipe is None:
        try:
            inpaint_pipe = load_inpaint_pipeline(model_choice)
        except Exception as e:
            st.error(f"Failed to load inpainting pipeline: {e}")
            inpaint_pipe = None

# Generate variants
def pil_to_np(img_pil):
    return np.array(img_pil)

def run_inpaint(pipeline, image_pil, mask_pil, prompt, num_inference_steps=30, guidance_scale=7.5, num_images=1):
    if pipeline is None:
        return []
    # model expects images in RGB PIL; mask: white=keep, black=fill? diffusers uses white area to inpaint
    results = []
    for n in range(num_images):
        out = pipeline(prompt=prompt, image=image_pil, mask_image=mask_pil, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images
        if isinstance(out, list):
            results.extend(out)
        else:
            results.append(out)
    return results

st.markdown("### Generate restorations")
all_outputs = {}
for style in style_choices:
    st.write(f"Generating for style: **{style}** — {ARCH_STYLES[style]}")
    prompts = []
    # base prompt variants
    base_desc = ARCH_STYLES[style]
    # Create a few different prompts to get variety
    for i in range(num_variants):
        if style == "Original / Neutral":
            p = f"Restore the missing part to match the original image. Seamless repair, consistent texture and color."
        else:
            p = f"Restore the missing part with {style} characteristics: {base_desc} Realistic, photorealistic, high detail." 
        # add small variation
        if i % 2 == 1:
            p += " Slightly aged, realistic wear "
        else:
            p += " Clean and well-maintained "
        prompts.append(p)

    outputs = []
    for p in prompts:
        if inpaint_pipe is None:
            st.info("Inpainting pipeline not available — skipping generation. Please install diffusers and a compatible model to enable generation.")
            break
        imgs = run_inpaint(inpaint_pipe, img, mask, prompt=p, num_images=1)
        outputs.extend(imgs)
    all_outputs[style] = outputs

# Display results
st.markdown("## Results")
for style, imgs in all_outputs.items():
    if not imgs:
        st.write(style + ": No outputs (model missing)")
        continue
    st.subheader(style)
    cols = st.columns(len(imgs))
    for i, im in enumerate(imgs):
        with cols[i]:
            st.image(im, use_column_width=True)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            b = buf.getvalue()
            st.download_button(label=f"Download {style} variant {i+1}", data=b, file_name=f"restoration_{style.replace(' ','_')}_{i+1}.png", mime="image/png")

st.success("Done — review the variants and download the ones you like.")

st.markdown("---\n### Notes & next steps:\n- If you want higher-fidelity architectural geometry reconstruction (cornices, column capitals), consider coupling this with a small 3D photogrammetry step or consulting a conservation architect.\n- To improve automatic damage detection, replace heuristic detector with a trained model for cracks/decay (e.g., U-Net trained for crack segmentation) or use Segment Anything (SAM) with automatic mask proposals.\n- You can swap the inpainting model in the sidebar to try different diffusion checkpoints.\n")

# End of app
