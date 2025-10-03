# Architectural Restorer – AI-Powered Restoration Tool

## Project Description
Architectural Restorer is an AI-powered web application that uses **computer vision** and **generative models** to restore and reimagine broken or damaged structures. 

- Upload an image of a damaged structure or object.
- The app automatically detects damaged areas and lets you refine the mask.
- AI reconstructs missing parts using **image inpainting (Stable Diffusion)**.
- Suggests architectural styles using **CLIP**.
- Generates multiple restoration ideas in different architectural styles.

##  Key Features
- Automatic + manual damage detection.
- AI-based inpainting to restore missing parts.
- Architectural style classification & suggestions.
- Reimagine in Gothic, Mughal, Victorian, Modernist, and more styles.
- Download restored images.

##  Deployment (Streamlit Cloud)
1. Fork or clone this repo to your GitHub account.
2. Ensure the following files are present:
   - `architectural_restorer_streamlit_app.py`
   - `requirements.txt`
3. Go to [Streamlit Cloud](https://share.streamlit.io/), log in with GitHub.
4. Click **New App** → Select this repo → Choose the `main` branch → Set **Main file path** to `architectural_restorer_streamlit_app.py`.
5. Deploy! Your app will be available at `https://yourappname.streamlit.app`.

##  Local Installation (Optional)
```bash
# Clone the repo
https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Run locally
streamlit run architectural_restorer_streamlit_app.py
```

##  Files in Repo
- `architectural_restorer_streamlit_app.py` → Main Streamlit app
- `requirements.txt` → List of dependencies
- `README.md` → Project description & instructions

##  Example Commit Messages
- `Initial commit: Added Streamlit app and requirements.txt`
- `Updated README with deployment instructions`
- `Improved damage detection mask`
- `Added new architectural styles (Baroque, Dravidian, Minimalist)`

##  Applications
- Heritage conservation
- Urban restoration
- Architectural design experiments
- Education & research

---
⚡ **In short:** This app acts like a *digital AI architect* that doesn’t just repair damaged structures in images but also reimagines them in multiple architectural styles.
