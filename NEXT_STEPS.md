# SixtyScan — Next Steps

## Changes Completed

### Security
| ID | File | Change |
|----|------|--------|
| C1 | `computer.py`, `mobile.py` | Added `weights_only=True` to `torch.load()` — prevents arbitrary code execution from a malicious checkpoint |
| C2 | `.devcontainer/devcontainer.json` | Removed `--server.enableCORS false --server.enableXsrfProtection false` — re-enables Streamlit's built-in CSRF protection |

### Correctness
| ID | File | Change |
|----|------|--------|
| H1 | `mobile.py` | Added `_clean_state_dict()` at module level; replaced the 3-line `load_model()` with the full robust version: DataParallel prefix stripping, compound-checkpoint detection (`{'state_dict': …}` format), and `strict=True → strict=False` fallback |
| H3 | `mobile.py` | Replaced probability-space averaging with logit-space averaging in `predict_from_model()` — now mathematically identical to `computer.py`. Removed `np.mean(all_probs)`; function now returns a single `float` |
| set_page_config | `computer.py`, `mobile.py` | Wrapped `st.set_page_config()` in `if __name__ == "__main__":` guard — eliminates `StreamlitAPIException` on the second page load when both files are imported via `app.py` |

### Reliability / Performance
| ID | File | Change |
|----|------|--------|
| H4 | `computer.py`, `mobile.py` | Moved `_clean_state_dict()` and `load_model()` to module level — `@st.cache_resource` now caches reliably using stable qualnames instead of closure-bound nested functions |
| M1 | `computer.py`, `mobile.py` | Wrapped `plt.close(fig)` in `try/finally` in `audio_to_mel_tensor()` — prevents matplotlib figure accumulation on exception paths |
| M2 | `computer.py`, `mobile.py` | Added file size validation (>50 MB) and duration validation (>30 s) in `audio_to_mel_tensor()` — shows Thai error messages before expensive librosa processing begins |

### UX / Performance
| Change | Files | Effect |
|--------|-------|--------|
| Removed `create_mel_spectrogram_display()` entirely | `computer.py`, `mobile.py` | Eliminates ~18 matplotlib renders per analysis run (9 audio × 2 figures each); analysis page no longer rerenders on every recording |
| Removed post-analysis spectrogram grid | `computer.py`, `mobile.py` | Results display immediately without a 9-figure rendering pass |
| Added `st.download_button` for every recording | `computer.py`, `mobile.py` | 3 call sites per file (7 vowels in loop + pataka + sentence) — 9 download buttons at runtime per path |

### Infrastructure
| Change | File | Effect |
|--------|------|--------|
| Created `.gitignore` | `.gitignore` | Covers `__pycache__/`, `*.pyc/pyo/pyd`, all model formats (`*.pth`, `*.pt`, `*.onnx`, `*.pkl`, `*.joblib`), `.DS_Store`, `.ipynb_checkpoints/`, `venv/`, `recordings/`, IDEs |
| Removed from git tracking | `best_model.pth`, `__pycache__/*.pyc`, `.DS_Store` | 5 files untracked via `git rm --cached`; local copies preserved |
| Added `pytz` to `requirements.txt` | `requirements.txt` | Declares the previously implicit transitive dependency |
| Removed unused packages | `requirements.txt` | Removed `scikit-learn==1.4.2`, `pandas==2.2.2`, `streamlit-webrtc==0.62.4` — eliminates ~250 MB of unnecessary install overhead on cold start |

---

## Remaining Issues

### Must Fix Before Production

| ID | Severity | File | Issue | Fix |
|----|----------|------|-------|-----|
| H2 | High | `computer.py`, `mobile.py` | ~95% code duplication — every bug fix must be applied twice; H1 was missed on mobile because of this | Extract shared logic into `utils.py`: audio processing, model loading, session state, validation. Each file becomes ~50 lines of layout-specific code |
| L3a | Low | `debugging.ipynb` | Still tracked in git — `*.ipynb` was not added to `.gitignore`, only `.ipynb_checkpoints/` | Run `git rm --cached debugging.ipynb` and add `*.ipynb` to `.gitignore` |
| L3b | Low | `.gitignore` | `*.ipynb` pattern missing — future notebooks will be tracked | Add `*.ipynb` line to `.gitignore` |
| L1 | Low | `model.py:7` | `models.resnet18(pretrained=True)` is deprecated since torchvision 0.13; prints `UserWarning` on every model load | Replace with `from torchvision.models import ResNet18_Weights` and `resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` |

### Should Fix Soon

| ID | Severity | File | Issue | Fix |
|----|----------|------|-------|-----|
| L5 | Low | `app.py` | Device detection falls through to `st.stop()` if all three detection methods fail — app shows spinner indefinitely | Add `device_type = 'desktop'` as the final else-branch fallback before `load_app_module()` |
| L2 | Low | `computer.py`, `mobile.py` | `atexit.register(cleanup_all_temp_files)` never fires between sessions — temp files accumulate in `/tmp` | Remove `atexit.register()`; add Streamlit `on_session_end` callback (Streamlit ≥ 1.29) |
| M3 | Medium | `deskstyle.css:158` | `min-width: 1000px !important` hardcoded — buttons overflow viewport on 1000–1200 px screens | Replace with `clamp(280px, 80vw, 1000px)` |
| M4 | Medium | `deskstyle.css`, `mobilestyle.css` | `[data-testid="stButton"]` and `[data-testid="stMarkdownContainer"]` selectors — Streamlit internal IDs, break silently on upgrades | Switch to `.stButton > button` (already partially used) and semantic class selectors |
| M5 | Medium | `deskstyle.css`, `mobilestyle.css` | `!important` on virtually every CSS property — cascade is unmanageable | Audit which properties genuinely need it; replace the rest with higher-specificity selectors like `.stApp .stButton > button` |
| M8 | Medium | `.streamlit/config.toml` | References `static/fonts/NotoSansThai-Regular.woff2` which does not exist — font silently missing | Either commit the `.woff2` file to `static/fonts/` or remove the `[[theme.fontFaces]]` block and rely solely on the Google Fonts CDN import in the CSS files |
| L4 | Low | `sampleaudio/yes/` | Directory contains only a stray file named `;` — PD-positive example audio missing | Add real PD-positive sample audio, or remove the directory and the `yes/` subdirectory reference |
| L6 | Low | `README.md` | Two lines — no setup instructions, no model info, no usage guide | Expand to cover installation, `streamlit run app.py`, model download behavior, desktop vs mobile paths |

---

## Suggested Implementation Order

### Phase 1 — Quick wins (< 1 hour total)
1. `git rm --cached debugging.ipynb` and add `*.ipynb` to `.gitignore`
2. Fix `model.py:7` — replace `pretrained=True` with `weights=ResNet18_Weights.IMAGENET1K_V1`
3. Fix `app.py` device detection fallback — add `device_type = 'desktop'` as final else
4. Remove `atexit.register()` calls from both files
5. Expand `README.md`

### Phase 2 — CSS / layout (2–4 hours)
6. Fix `deskstyle.css` hardcoded `1000px` button width
7. Audit and remove unnecessary `!important` declarations
8. Replace fragile `data-testid` selectors
9. Resolve the missing `NotoSansThai-Regular.woff2` font

### Phase 3 — Code deduplication (4–8 hours, moderate risk)
10. Extract `utils.py`: `_clean_state_dict`, `load_model`, `audio_to_mel_tensor`, `audio validation`, `convert_to_wav_if_needed`, `initialize_session_state`, `cleanup_*`, `add_temp_file`, `save_uploaded_file`, `predict_from_model`
11. Reduce `computer.py` and `mobile.py` to layout-only code (~50 lines each)
12. Run full test checklist after each extraction step — do not batch

### Phase 4 — Next.js migration (see section below)

---

## Test Checklist

### Smoke tests (run after every change)
- [ ] `python -m py_compile computer.py mobile.py app.py model.py` — no syntax errors
- [ ] `python -c "import computer; import mobile; print('imports OK')"` — no Streamlit calls at import time
- [ ] `streamlit run app.py` — app loads, spinner appears, JS redirect fires, desktop UI loads without `StreamlitAPIException`

### Recording workflow
- [ ] Record all 7 vowels — each shows a ⬇️ download button after recording
- [ ] Download a vowel WAV — file opens in audio player
- [ ] Record pataka — download button appears
- [ ] Record sentence — download button appears
- [ ] Re-record a vowel — previous download button updates to new file
- [ ] Click ลบข้อมูล (clear) — all download buttons disappear, all audio inputs reset

### Upload workflow
- [ ] Upload 7 vowel files via file uploader — download buttons appear
- [ ] Upload pataka and sentence — download buttons appear
- [ ] Mix recording and upload — both paths produce download buttons

### Validation
- [ ] Upload a WAV longer than 30 seconds — Thai error message appears, analysis blocked
- [ ] Upload a file > 50 MB — error message appears, analysis blocked
- [ ] Submit with fewer than 9 inputs — Thai warning appears, no analysis runs

### Analysis
- [ ] Submit all 9 recordings — result card appears with label, percentage, and advice
- [ ] Desktop and mobile paths with identical recordings — probability values match within ±0.1%
- [ ] Result label is "Non Parkinson" for % ≤ 50, "Parkinson" for % > 50

### Model loading
- [ ] Delete cached model file, run app — download spinner appears, model loads successfully
- [ ] Reload page without deleting model — no re-download, instant load (cache hit)
- [ ] Check Streamlit logs — no `UserWarning` from `pretrained=True` (after Phase 1 fix)

### Navigation
- [ ] Home → เริ่มใช้งาน → Analysis page loads
- [ ] Home → คู่มือ → Guide page loads with sample audio players
- [ ] Back button from Analysis returns to Home
- [ ] Back button from Guide returns to Home

### Mobile path
- [ ] Open `http://localhost:8501/?device=mobile` — mobile layout loads
- [ ] All recording and download features work identically to desktop

---

## Deployment Checklist

### Before deploying to Streamlit Cloud

- [ ] `requirements.txt` does not include `scikit-learn`, `pandas`, or `streamlit-webrtc`
- [ ] `packages.txt` contains `ffmpeg`
- [ ] `best_model.pth` and `best_resnet18.pth` are **not** in the git repository
- [ ] Google Drive URLs in `CONFIG['MODEL_URL']` (both `computer.py` and `mobile.py`) are publicly accessible — test with an incognito browser
- [ ] `gdown` version `5.2.0` handles the Drive URLs without the "Permission denied" error — verify with `gdown --version`
- [ ] `.streamlit/config.toml` font reference is resolved (either the `.woff2` file is committed, or the broken `[[theme.fontFaces]]` block is removed)
- [ ] `sampleaudio/no/` contains all 9 `.m4a` files
- [ ] All image assets present: `logo.png`, `insert.jpg`, `doctor.jpg`, `doctor2.jpg`, `reward.jpg`, `present.jpg`, `tamdai.png`
- [ ] `.devcontainer/devcontainer.json` does not contain `--server.enableCORS false` or `--server.enableXsrfProtection false`
- [ ] App cold-start time is under 120 seconds — test on Streamlit Cloud free tier after trimming requirements

### Environment
- [ ] Python 3.11 (matches `.devcontainer` image and compiled `.pyc` files)
- [ ] `torch==2.2.2` installs correctly on the target platform (CPU-only for Streamlit Cloud)
- [ ] Thai font renders correctly in deployed environment — test on a non-macOS machine

---

## Migration to Next.js

### Why migrate
Streamlit is a rapid-prototyping tool. It re-runs the entire Python script on every user interaction, which causes UI lag, limits layout control, and makes fine-grained state management difficult. A Next.js frontend with a Python API backend would give full control over UX, layout, performance, and deployment.

### Architecture target

```
┌─────────────────────────────┐     ┌──────────────────────────────┐
│  Next.js frontend (Vercel)  │────▶│  FastAPI backend (Railway /  │
│  React + Tailwind + App     │     │  Fly.io / GCP Cloud Run)     │
│  Router                     │◀────│  PyTorch inference + librosa │
└─────────────────────────────┘     └──────────────────────────────┘
```

### Component mapping

| Current (Streamlit) | Next.js equivalent |
|---------------------|--------------------|
| `st.session_state` | React `useState` / Zustand store |
| `st.session_state.page` routing | Next.js App Router (`/`, `/analysis`, `/guide`) |
| `st.audio_input()` | `react-media-recorder` or Web Audio API + `MediaRecorder` |
| `st.file_uploader()` | `<input type="file">` with drag-and-drop |
| `st.download_button()` | `URL.createObjectURL(blob)` + `<a download>` click |
| `st.audio()` playback | `<audio>` element with Blob URL |
| `st.spinner()` | Loading state + skeleton UI |
| `st.success()` / `st.error()` | Toast notifications (react-hot-toast or sonner) |
| `st.columns()` | CSS Grid / Flexbox |
| `@st.cache_resource` model | Python backend singleton with `lifespan` startup event |
| Thai font (Prompt) | `next/font/google` with `Prompt` |
| Inline HTML + `unsafe_allow_html` | Native JSX components |
| `deskstyle.css` / `mobilestyle.css` | Tailwind CSS with responsive prefixes (`md:`, `lg:`) |

### Backend API design (FastAPI)

```
POST /api/predict
  Body: multipart/form-data with 9 audio files
  Returns: { "pd_probability": float, "percent": int, "label": str, "level": str }

GET /api/health
  Returns: { "model_loaded": bool }
```

The `audio_to_mel_tensor()` function moves to the backend unchanged — no preprocessing changes required. The model weights load once at startup via FastAPI `lifespan`.

### Migration steps (suggested order)

1. Set up FastAPI backend — port `audio_to_mel_tensor`, `predict_from_model`, and `load_model` from `utils.py` (Phase 3 extraction above makes this straightforward)
2. Add `/api/predict` endpoint with `python-multipart` for file upload
3. Scaffold Next.js app with App Router and Tailwind
4. Implement Thai-language pages: Home, Guide, Analysis
5. Port recording workflow using `react-media-recorder`
6. Connect analysis page to `/api/predict` via `fetch`
7. Implement download via Blob URL
8. Port CSS design from `deskstyle.css` to Tailwind utility classes — device detection becomes unnecessary (CSS responsive breakpoints replace `computer.py` vs `mobile.py`)
9. Deploy frontend to Vercel, backend to a container host with persistent model cache

### Key decisions before starting

- **Model file hosting**: the 43 MB `best_model.pth` must be accessible at backend startup. Options: bundle in the Docker image, download from Google Drive at startup (current approach), or store in object storage (S3/GCS) and stream at startup.
- **Audio format**: browsers record in `audio/webm` (Chrome) or `audio/mp4` (Safari). The backend must handle both. `pydub` via `ffmpeg` already does this — keep the `convert_to_wav_if_needed` logic server-side.
- **CORS**: the FastAPI backend must allow the Vercel frontend domain.
- **GDPR**: if deployed publicly, the Google Fonts CDN calls (Prompt font) send user IPs to Google. Self-host the font to comply.
