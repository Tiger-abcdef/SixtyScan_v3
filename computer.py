import streamlit as st
import os
import base64
import tempfile
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import io
import tempfile
from datetime import datetime
import pytz
import atexit
from pathlib import Path

# =============================
# Configuration
# =============================
CONFIG = {
    'MODEL_PATH': "best_model.pth",
    'CSS_FILE': "deskstyle.css",
    'LOGO_PATHS': ["logo.png", "./logo.png", "assets/logo.png", "images/logo.png"],
    'IMAGE_PATHS': ["insert.jpg", "./insert.jpg", "assets/insert.jpg", "images/insert.jpg"],
    'THAI_TIMEZONE': 'Asia/Bangkok',
    'DOCTOR_PATHS' : ["doctor.jpg", "./doctor.jpg", "assets/doctor.jpg", "images/doctor.jpg"],
    'REWARD_PATHS' : ["reward.jpg", "./reward.jpg", "assets/reward.jpg", "images/reward.jpg"],
    'PRESENT_PATHS' : ["present.jpg", "./present.jpg", "assets/present.jpg", "images/present.jpg"],
    'DOCTOR2_PATHS' : ["doctor2.jpg", "./doctor2.jpg", "assets/doctor2.jpg", "images/doctor2.jpg"],
    # Control which class index is PD in your training (0 or 1)
    'PD_INDEX': 1,
}

if __name__ == "__main__":
    st.set_page_config(
        page_title="SixtyScan - Parkinson Detection",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

# Import your model class
try:
    from model import ResNet18Classifier
except ImportError:
    st.error("Could not import ResNet18Classifier from model.py. Make sure the file exists.")
    st.stop()

# =============================
# Model Loading
# =============================
def _clean_state_dict(state_dict: dict) -> dict:
    """Strip 'module.' prefixes from DataParallel checkpoints."""
    return {(k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state_dict.items()}

@st.cache_resource
def load_model():
    """Load the ResNet18 model from the bundled checkpoint."""
    try:
        if not os.path.exists(CONFIG['MODEL_PATH']):
            st.error(f"Model file '{CONFIG['MODEL_PATH']}' not found. Ensure it is present in the repository root.")
            return None
        model = ResNet18Classifier()
        ckpt = torch.load(CONFIG['MODEL_PATH'], map_location=torch.device("cpu"), weights_only=True)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state = _clean_state_dict(ckpt['state_dict'])
        elif isinstance(ckpt, dict):
            state = _clean_state_dict(ckpt)
        else:
            state = ckpt
        try:
            model.load_state_dict(state, strict=True)
        except Exception as e:
            st.warning(f"Strict load failed ({e}); trying non-strict load.")
            model.load_state_dict(state, strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# =============================
# Utility Functions
# =============================
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'page': 'home',
        'vowel_files': [],
        'pataka_file': None,
        'sentence_file': None,
        'clear_clicked': False,
        'temp_files': [],  # Track all temp files for cleanup
        'result': None,    # Last analysis result; cleared only by ลบข้อมูล
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def cleanup_temp_files(file_list):
    """Clean up specific temporary files"""
    for file_path in file_list:
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            st.warning(f"Failed to delete temp file {file_path}: {str(e)}")

def cleanup_all_temp_files():
    """Clean up all temporary files stored in session state"""
    if 'temp_files' in st.session_state:
        cleanup_temp_files(st.session_state.temp_files)
        st.session_state.temp_files = []

def add_temp_file(file_path):
    """Add a file to the temp files tracking list"""
    if 'temp_files' not in st.session_state:
        st.session_state.temp_files = []
    st.session_state.temp_files.append(file_path)

def run_desktop_app():
    """Main function to run the desktop version"""
    # Initialize Session State
    initialize_session_state()

    # Register cleanup function
    atexit.register(cleanup_all_temp_files)

    # =============================
    # Page-specific Functions
    # =============================
    def load_css():
        """Load external CSS file"""
        css_file = Path(CONFIG['CSS_FILE'])
        if css_file.exists():
            with open(css_file, 'r', encoding='utf-8') as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        else:
            st.warning(f"CSS file '{CONFIG['CSS_FILE']}' not found. Using minimal styling.")
            # Fallback minimal CSS
            st.markdown("""
                <style>
                @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700;800;900&display=swap');
                * { font-family: 'Prompt', sans-serif !important; }
                .stApp { background: linear-gradient(135deg, #f8f4ff 0%, #e8f4fd 100%) !important; }
                </style>
            """, unsafe_allow_html=True)

    @st.cache_data
    def load_image_file(image_paths, alt_text="Image"):
        """Generic function to load image files with fallback options"""
        for path in image_paths:
            try:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        return base64.b64encode(f.read()).decode()
            except Exception as e:
                st.warning(f"Failed to load {path}: {str(e)}")
                continue
        return None

    def get_thai_time():
        """Get current Thai time formatted for display"""
        try:
            thai_tz = pytz.timezone(CONFIG['THAI_TIMEZONE'])
            now = datetime.now(thai_tz)
            return now.strftime("%d/%m/%Y %H:%M:%S")
        except Exception as e:
            st.error(f"Error getting Thai time: {str(e)}")
            return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    def save_uploaded_file(uploaded_file):
        """Save uploaded file to temporary location"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.read())
                add_temp_file(tmp.name)
                return tmp.name
        except Exception as e:
            st.error(f"Error saving uploaded file: {str(e)}")
            return None

    # =============================
    # Model and Analysis Functions
    # =============================
    def convert_to_wav_if_needed(file_path):
        """Convert audio file to WAV format if necessary"""
        if file_path is None:
            return None
        try:
            from pydub import AudioSegment
            if not file_path.lower().endswith(".wav"):
                audio = AudioSegment.from_file(file_path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    audio.export(tmp.name, format="wav")
                    add_temp_file(tmp.name)
                    return tmp.name
            return file_path
        except ImportError:
            # Fallback: librosa can read many formats without converting
            return file_path
        except Exception as e:
            st.error(f"Error converting audio file: {str(e)}")
            return None

    def audio_to_mel_tensor(file_path):
        """Convert audio file to mel spectrogram tensor (training-aligned)."""
        if file_path is None:
            return None
        try:
            wav_file = convert_to_wav_if_needed(file_path)
            if not wav_file:
                return None

            if os.path.getsize(wav_file) > 50 * 1024 * 1024:
                st.error("ไฟล์เสียงใหญ่เกินไป (สูงสุด 50MB)")
                return None

            # Match training: do NOT force-resample; use mono
            y, sr = librosa.load(wav_file, sr=None, mono=True)

            duration = len(y) / sr
            if duration > 30:
                st.error(f"เสียงยาวเกินไป ({duration:.1f}s) กรุณาบันทึกไม่เกิน 30 วินาที")
                return None

            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Render to a 224x224 image exactly like training-style specshow
            fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
            ax.axis('off')
            librosa.display.specshow(mel_db, sr=sr, ax=ax)  # no cmap/fmax: keep as default
            buf = io.BytesIO()
            try:
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            finally:
                plt.close(fig)
            buf.seek(0)

            image = Image.open(buf).convert('RGB')

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225]),
            ])
            return transform(image).unsqueeze(0)  # (1, 3, 224, 224)
        except Exception as e:
            st.error(f"Error creating mel tensor: {str(e)}")
            return None

    def predict_from_model(vowel_paths, pataka_path, sentence_path, model):
        """Predict PD probability by averaging logits (more stable than averaging probs)."""
        try:
            tensors = []

            # Process vowel files
            for path in vowel_paths:
                t = audio_to_mel_tensor(path)
                if t is None:
                    st.error(f"Failed to process vowel file: {path}")
                    return None
                tensors.append(t)

            # Process pataka and sentence — skip None paths (bypass mode or missing recordings)
            for path in [p for p in [pataka_path, sentence_path] if p is not None]:
                t = audio_to_mel_tensor(path)
                if t is None:
                    st.error(f"Failed to process file: {path}")
                    return None
                tensors.append(t)

            if not tensors:
                return None

            with torch.no_grad():
                logits_list = []
                for t in tensors:
                    out = model(t)            # [1, 2]
                    logits_list.append(out)

                # Combine logits: [N,2] -> mean over N -> [2]
                logits_all = torch.cat(logits_list, dim=0)      # [N, 2]
                mean_logits = logits_all.mean(dim=0)            # [2]
                probs = torch.softmax(mean_logits, dim=0)       # [2]

                pd_idx = CONFIG.get('PD_INDEX', 1)
                final_prob_pd = float(probs[pd_idx].item())
                return final_prob_pd

        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            return None

    # =============================
    # Result PNG builder
    # =============================
    def build_result_png(label, level, percent, border_color, box_color):
        """Render the result card as a PNG and return the raw bytes."""
        import io as _io
        import datetime as _dt
        import numpy as _np
        import matplotlib.pyplot as _plt
        import matplotlib.patches as _patches
        from matplotlib.colors import LinearSegmentedColormap as _LSC

        try:
            risk_en = level[level.index('(') + 1: level.index(')')]
        except ValueError:
            risk_en = level

        diagnosis_en = "No Parkinson's Detected" if label == "Non Parkinson" else "Parkinson's Detected"

        if percent <= 50:
            advice = [
                "No symptoms: annual check-up (optional)",
                "Mild symptoms: check-up twice per year",
                "Warning signs: check-up 2-4 times per year",
            ]
        elif percent <= 75:
            advice = [
                "Consult a neurologist",
                "Keep a daily symptom journal",
                "If on medication: record any side effects",
            ]
        else:
            advice = [
                "See a neurologist as soon as possible",
                "Keep a daily symptom journal",
                "If on medication: monitor closely",
            ]

        fig = _plt.figure(figsize=(7, 9.5), facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 9.5)
        ax.axis('off')

        ax.add_patch(_patches.FancyBboxPatch(
            (0.25, 0.25), 6.5, 9.0, boxstyle="round,pad=0.1",
            facecolor=box_color, edgecolor=border_color, linewidth=4))
        ax.add_patch(_patches.Rectangle(
            (0.25, 0.25), 0.2, 9.0, facecolor=border_color, linewidth=0))

        ax.text(3.5, 8.85, 'SixtyScan', ha='center', va='center',
                fontsize=24, fontweight='bold', color='#4A148C')
        ax.text(3.5, 8.4, 'Voice-based Parkinson Screening Result',
                ha='center', va='center', fontsize=11, color='#777', style='italic')
        ax.plot([0.6, 6.4], [8.1, 8.1], color=border_color, linewidth=1.2)

        ax.text(3.5, 7.55, label, ha='center', va='center',
                fontsize=40, fontweight='bold', color=border_color)
        ax.text(3.5, 6.85, f'Risk Level:  {risk_en}',
                ha='center', va='center', fontsize=14, color='#444')
        ax.text(3.5, 6.3, f'PD Probability:  {percent}%',
                ha='center', va='center', fontsize=18, fontweight='bold', color='#111')

        bx, by, bw, bh = 0.7, 5.75, 5.6, 0.35
        cmap = _LSC.from_list('risk', ['#4caf50', '#ff9800', '#f44336'])
        ax.imshow(_np.linspace(0, 1, 256).reshape(1, -1), aspect='auto', cmap=cmap,
                  extent=[bx, bx + bw, by, by + bh], zorder=2)
        ax.add_patch(_patches.FancyBboxPatch(
            (bx, by), bw, bh, boxstyle="round,pad=0.04",
            facecolor='none', edgecolor='#bbb', linewidth=1, zorder=3))
        mx = bx + (percent / 100.0) * bw
        ax.add_patch(_patches.Rectangle(
            (mx - 0.045, by - 0.1), 0.09, bh + 0.2,
            facecolor='#222', edgecolor='white', linewidth=1.2, zorder=4))
        ax.text(bx - 0.05, by + bh / 2, '0%',   ha='right', va='center', fontsize=8, color='#888')
        ax.text(bx + bw + 0.05, by + bh / 2, '100%', ha='left',  va='center', fontsize=8, color='#888')

        ax.text(3.5, 5.25, f'Diagnosis:  {diagnosis_en}',
                ha='center', va='center', fontsize=14, color='#333')
        ax.plot([0.6, 6.4], [4.95, 4.95], color='#ddd', linewidth=0.8)

        ax.text(3.5, 4.65, 'Recommendations', ha='center', va='center',
                fontsize=13, fontweight='bold', color='#333')
        for i, line in enumerate(advice):
            ax.text(1.0, 4.15 - i * 0.62, f'•  {line}',
                    ha='left', va='center', fontsize=11, color='#444')

        ax.plot([0.6, 6.4], [0.9, 0.9], color='#ddd', linewidth=0.8)
        try:
            ts = _dt.datetime.now(
                __import__('pytz').timezone('Asia/Bangkok')
            ).strftime('%d %b %Y  %H:%M ICT')
        except Exception:
            ts = _dt.datetime.now().strftime('%d %b %Y  %H:%M')
        ax.text(3.5, 0.65, f'Generated: {ts}', ha='center', va='center',
                fontsize=9, color='#aaa')
        ax.text(3.5, 0.38,
                'Screening tool only. Consult a physician for medical diagnosis.',
                ha='center', va='center', fontsize=8, color='#bbb', style='italic')

        buf = _io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        _plt.close(fig)
        buf.seek(0)
        return buf.read()

    # =============================
    # Page Functions
    # =============================
    def get_header_html():
        """Get the header HTML without rendering it"""
        logo_b64 = load_image_file(CONFIG['LOGO_PATHS'], "SixtyScan Logo")
        current_time = get_thai_time()

        logo_html = ""
        if logo_b64:
            logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="header-logo" style="height: 56px; width: auto; margin-right: 24px;" alt="SixtyScan Logo">'

        return f"""
            <div class="header-container">
                <div class="logo-section">
                    {logo_html}
                    <div class="logo-text">SixtyScan</div>
                    <div class="header-divider"></div>
                    <div class="tagline">นวัตกรรมคัดกรองโรคพาร์กินสันจากเสียง</div>
                </div>
                <div class="datetime-display">{current_time}</div>
            </div>
        """

    def show_home_page():
        """Display the home page - FIXED VERSION"""
        load_css()

        woman_image_b64 = load_image_file(CONFIG['IMAGE_PATHS'], "Woman using phone")

        # SOLUTION: Combine header and main content in ONE st.markdown call
        combined_html = f"""
            {get_header_html()}
            <div class="main-content">
                <div class="content-wrapper">
                    <div class="text-section">
                        <h1 class="main-title">
                            ตรวจเช็คโรคพาร์กินสัน<br>ทันทีด้วย <span class="highlight">SixtyScan</span>
                        </h1>
                    </div>
                    <div class="image-section">
                        {f'<img src="data:image/jpg;base64,{woman_image_b64}" alt="Woman using phone" class="main-image">' if woman_image_b64 else '''
                        <div class="image-placeholder">
                            <div class="placeholder-content">
                                <div class="placeholder-icon">📱</div>
                                <div class="placeholder-text">
                                    insert.jpg<br>not found
                                </div>
                            </div>
                        </div>
                        '''}
                    </div>
                </div>
            </div>
        """

        # Render the combined HTML (no gap between header and content!)
        st.markdown(combined_html, unsafe_allow_html=True)

        # Add buttons positioned within the text section area
        st.markdown('<div class="homepage-buttons-wrapper">', unsafe_allow_html=True)

        # First button - เริ่มใช้งาน (Start Analysis)
        if st.button("**เริ่มใช้งาน**", key="start_analysis"):
            st.session_state.page = 'analysis'
            st.rerun()

        # Second button - คู่มือ (Guide)
        if st.button("**คู่มือ**", key="guide_manual"):
            st.session_state.page = 'guide'
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # ============ About Us Section ============ (Desktop optimized for 1920x1080)
        st.markdown("""
            <div style="margin:40px auto; max-width:1200px; padding:40px; background:linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%); border-radius:25px; box-shadow:0 8px 32px rgba(74, 20, 140, 0.1); border:1px solid rgba(74, 20, 140, 0.05);">
                <div style="text-align:center; margin-bottom:30px;">
                    <h2 style="color:#4A148C; font-family:'Prompt',sans-serif; font-size:36px; font-weight:700; margin:0; padding-top: 15px; text-align:center;">
                        เกี่ยวกับเรา
                    </h2>
                    <div style="width: 80px; height: 4px; background: linear-gradient(135deg, #4A148C, #7B1FA2); margin: 12px auto; border-radius: 2px;"></div>
                </div>
                <div style="max-width:900px; margin:0 auto; padding:0 20px;">
                    <p style="font-size:20px; line-height:1.8; text-align:center; font-family:'Prompt',sans-serif; margin-bottom:24px; color:#2c2c2c;">
                        แรงบันดาลใจของ <strong style="color:#4A148C;">SixtyScan.life</strong> เริ่มจากคนใกล้ตัวที่บ้านของเรา ที่เป็นผู้ป่วยโรคพาร์กินสัน 
                        ได้เห็นถึงความยากลำบากของท่านและผู้ที่เกี่ยวข้องทุกคน จึงเกิดคำถามว่า 
                        <em>"ถ้าช่วยผู้คนเข้าถึงการรักษาได้เร็ว จะช่วยสังคมได้มาก"</em>
                    </p>
                    <p style="font-size:20px; line-height:1.8; text-align:center; font-family:'Prompt',sans-serif; margin-bottom:30px; color:#2c2c2c;">
                        ด้วยความตั้งใจนั้น จึงนำความคิดไปปรึกษาคุณครู จนได้รวมทีมกัน 
                        ใช้เทคโนโลยีพัฒนาเป็น <strong style="color:#4A148C;">SixtyScan.life</strong>
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Doctor section - simplified version first
        try:
            doctor_image_b64 = load_image_file(CONFIG['DOCTOR_PATHS'], "doctor")
            doctor2_image_b64 = load_image_file(CONFIG['DOCTOR2_PATHS'], "doctor2")
        
            if doctor_image_b64 and doctor2_image_b64:
                st.markdown(f"""
                <div style="max-width:1200px; margin:30px auto; padding:0 40px;">
                    <div style="display:flex; flex-wrap:wrap; gap:60px; align-items:center; justify-content:center;">
                        <div style="text-align:center; flex:1; min-width:300px; max-width:400px;">
                            <img src="data:image/jpg;base64,{doctor_image_b64}" alt="นพ.ณัฐฏ์ กล้าผจญ" style="width:100%; max-width:350px; border-radius:15px; box-shadow:0 6px 20px rgba(0,0,0,0.15);">
                            <p style="font-size:18px; color:#4A148C; font-family:'Prompt',sans-serif; line-height:1.5; margin-top:16px; font-weight:600;">
                                นพ.ณัฐฏ์ กล้าผจญ
                            </p>
                        </div>
                        <div style="text-align:center; flex:1; min-width:300px; max-width:400px;">
                            <img src="data:image/jpg;base64,{doctor2_image_b64}" alt="ผศ.นพ.สุรัตน์ ตันประเวช" style="width:100%; max-width:350px; border-radius:15px; box-shadow:0 6px 20px rgba(0,0,0,0.15);">
                            <p style="font-size:18px; color:#4A148C; font-family:'Prompt',sans-serif; line-height:1.5; margin-top:16px; font-weight:600;">
                                ผศ.นพ.สุรัตน์ ตันประเวช
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif doctor_image_b64:
                st.markdown(f"""
                <div style="max-width:1200px; margin:30px auto; padding:0 40px; text-align:center;">
                    <img src="data:image/jpg;base64,{doctor_image_b64}" alt="นพ.ณัฐฏ์ กล้าผจญ" style="max-width:400px; width:100%; border-radius:15px; box-shadow:0 6px 20px rgba(0,0,0,0.15); margin-bottom:16px;">
                    <p style="font-size:18px; color:#4A148C; font-family:'Prompt',sans-serif; font-weight:600;">นพ.ณัฐฏ์ กล้าผจญ</p>
                </div>
                """, unsafe_allow_html=True)
            elif doctor2_image_b64:
                st.markdown(f"""
                <div style="max-width:1200px; margin:30px auto; padding:0 40px; text-align:center;">
                    <img src="data:image/jpg;base64,{doctor2_image_b64}" alt="ผศ.นพ.สุรัตน์ ตันประเวช" style="max-width:400px; width:100%; border-radius:15px; box-shadow:0 6px 20px rgba(0,0,0,0.15); margin-bottom:16px;">
                    <p style="font-size:18px; color:#4A148C; font-family:'Prompt',sans-serif; font-weight:600;">ผศ.นพ.สุรัตน์ ตันประเวช</p>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.warning(f"Could not load doctor image: {e}")

        # Continuation section
        st.markdown("""
        <div style="max-width:1200px; margin:20px auto; padding:40px; background:linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%); border-radius:25px; box-shadow:0 8px 32px rgba(74, 20, 140, 0.1); border:1px solid rgba(74, 20, 140, 0.05);">
            <div style="max-width:900px; margin:0 auto;">
                <p style="font-size:18px; line-height:1.8; text-align:center; font-family:'Prompt',sans-serif; margin-bottom:0; color:#2c2c2c;">
                    จากแนวคิดนี้ เราได้รับรางวัลจาก <strong style="color:#4A148C;">AI Builder 2025</strong> 
                    และปัจจุบันพวกเรามีโอกาสทำงานร่วมกับแพทย์ผู้เชี่ยวชาญด้านประสาทวิทยา<br><br>
                    ได้แก่ <strong>นพ.ณัฐฏ์ กล้าผจญ</strong> และ<br><strong>ผศ.นพ.สุรัตน์ ตันประเวช</strong><br>
                    จาก <strong style="color:#4A148C;">MED CMU Health Innovation Center (MedCHIC) มหาวิทยาลัยเชียงใหม่</strong>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Contact section
        st.markdown("""
        <div style="max-width:800px; margin:40px auto; padding:40px; background:#e3f2fd; border-radius:20px;">
            <h2 style="text-align:center; color:#1565C0; font-family:'Prompt',sans-serif; margin-bottom:30px; font-size:28px;">ติดต่อเรา</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Address section
        st.markdown("""
        <div style="max-width:600px; margin:20px auto; padding:25px; background:white; border-radius:15px; text-align:center; box-shadow:0 4px 10px rgba(0,0,0,0.1);">
            <h3 style="color:#1565C0; font-family:'Prompt',sans-serif; margin-bottom:15px;">📍 ที่อยู่</h3>
            <p style="font-size:16px; font-family:'Prompt',sans-serif; color:#2c2c2c; line-height:1.6; margin:0;">
                121/11 อาคารอีคิวสแควร์<br>
                ถนนเชียงใหม่-ฮอด ตำบลป่าแดด<br>
                อำเภอเมืองเชียงใหม่<br>
                จังหวัดเชียงใหม่ 50100
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Phone section
        st.markdown("""
        <div style="max-width:400px; margin:20px auto; padding:25px; background:white; border-radius:15px; text-align:center; box-shadow:0 4px 10px rgba(0,0,0,0.1);">
            <h3 style="color:#1565C0; font-family:'Prompt',sans-serif; margin-bottom:15px;">📞 โทรศัพท์</h3>
            <p style="font-size:22px; font-weight:600; color:#2e7d32; font-family:'Prompt',sans-serif; margin:0;">
                064-9506228
            </p>
        </div>
        """, unsafe_allow_html=True)

    def show_guide_page():
        """Display the guide/manual page with proper styling - FIXED VERSION"""
        load_css()

    # FIXED: Combine header with back button and title
        guide_html = f"""
            {get_header_html()}
            <div class="guide-container">
                <h1 class="guide-title">คู่มือการใช้งาน SixtyScan</h1>
            </div>
        """

        st.markdown(guide_html, unsafe_allow_html=True)

    # Back button
        if st.button("**← กลับหน้าแรก**", key="back_to_home_from_guide"):
            st.session_state.page = 'home'
            st.rerun()

    # Guide content - Fixed version with proper HTML structure
        st.markdown("""
            <div style="max-width: 1000px; margin: 0 auto; padding: 0 40px;">
                <div style="background: white; padding: 40px; border-radius: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.08); margin-bottom: 32px;">
                    <h2 style="color: #4A148C; font-size: 36px; margin-bottom: 24px; margin-top: 0; font-family: 'Prompt', sans-serif;">การเตรียมตัวก่อนการตรวจ</h2>
                    <div style="font-size: 22px; line-height: 1.7; font-family: 'Prompt', sans-serif; margin-top: 0; padding-left: 24px;">
                        <div style="margin-bottom: 8px;"><strong>1.</strong> พักผ่อนเพียงพอก่อนการตรวจ</div>
                        <div style="margin-bottom: 8px;"><strong>2.</strong> หาสถานที่เงียบ ปราศจากเสียงรบกวน</div>
                        <div style="margin-bottom: 8px;"><strong>3.</strong> นั่งหรือยืนในท่าที่สบาย</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="max-width: 1000px; margin: 0 auto; padding: 0 40px;">
                <div style="background: white; padding: 40px; border-radius: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.08); margin-bottom: 32px;">
                    <h2 style="color: #4A148C; font-size: 36px; margin-bottom: 24px; margin-top: 0; font-family: 'Prompt', sans-serif;">ขั้นตอนการตรวจ</h2>
                    <ul style="font-size: 22px; line-height: 1.7; font-family: 'Prompt', sans-serif; margin-top: 0; padding-left: 24px;">
                        <li style="margin-bottom: 16px;"><strong>การออกเสียงสระ:</strong> ออกเสียงสระแต่ละตัว 5-8 วินาที ให้ชัดเจนและคงที่</li>
                        <li style="margin-bottom: 16px;"><strong>การออกเสียงพยางค์:</strong> ออกเสียง "พา-ทา-คา" ซ้ำๆ ประมาณ 6 วินาที</li>
                        <li style="margin-bottom: 16px;"><strong>การอ่านประโยค:</strong> อ่านประโยคที่กำหนดให้อย่างเป็นธรรมชาติและชัดเจน</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="max-width: 1000px; margin: 0 auto; padding: 0 40px;">
                <div style="background: white; padding: 40px; border-radius: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.08);">
                    <h2 style="color: #4A148C; font-size: 36px; margin-bottom: 24px; margin-top: 0; font-family: 'Prompt', sans-serif;">ข้อควรระวัง</h2>
                    <ul style="font-size: 22px; line-height: 1.7; color: #d32f2f; font-family: 'Prompt', sans-serif; margin-top: 0; padding-left: 24px;">
                        <li style="margin-bottom: 12px;"><strong style="font-weight: 600;">ระบบนี้เป็นเพียงการตรวจคัดกรองเบื้องต้น</strong></li>
                        <li style="margin-bottom: 12px;"><strong style="font-weight: 600;">ไม่สามารถทดแทนการวินิจฉัยโดยแพทย์ได้</strong></li>
                        <li style="margin-bottom: 12px;"><strong style="font-weight: 600;">หากมีข้อสงสัยควรปรึกษาแพทย์เฉพาะทาง</strong></li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # FIXED: Move the "คำแนะนำเพิ่มเติม" section BEFORE the sample audio section
        st.markdown("""
            <div style="max-width: 1000px; margin: 0 auto; padding: 0 40px;">
                <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border-radius: 20px; padding: 30px; margin: 30px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.1); border-left: 6px solid #1976d2;">
                    <h3 style="color: #1565c0; margin-bottom: 20px; font-family: 'Prompt', sans-serif; font-size: 24px; font-weight: 600; text-align: center;">💡 คำแนะนำเพิ่มเติม</h3>
                    <ul style="font-size: 22px; font-family: 'Prompt', sans-serif; line-height: 1.8; color: #2e7d32; margin: 0; padding-left: 24px;">
                        <li style="margin-bottom: 12px;">ฟังตัวอย่างเสียงก่อนเริ่มการตรวจเพื่อเข้าใจรูปแบบการออกเสียงที่ถูกต้อง</li>
                        <li style="margin-bottom: 12px;">พยายามออกเสียงให้เหมือนกับตัวอย่างให้มากที่สุด</li>
                        <li style="margin-bottom: 12px;">หากไม่แน่ใจ สามารถฟังตัวอย่างซ้ำได้หลายครั้ง</li>
                        <li style="margin-bottom: 12px;">ตัวอย่างเสียงเหล่านี้เป็นเสียงจากผู้ที่ไม่เป็นพาร์กินสัน</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Sample audio section
        st.markdown("""
            <div style="max-width: 1000px; margin: 0 auto; padding: 0 40px;">
                <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 20px; padding: 30px; margin: 30px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
                    <h3 style="color: #495057; margin-bottom: 25px; font-family: 'Prompt', sans-serif; font-size: 24px; font-weight: 600; text-align: center;">🎵 ตัวอย่างเสียงที่ถูกต้อง</h3>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Sample audio files in order according to analysis page
        sample_audio_files = [
            ("อา", "sampleaudio/no/อา 1(1) pd.m4a"),
            ("อี", "sampleaudio/no/E 1(1) pd.m4a"),
            ("อือ", "sampleaudio/no/อือ 1(1) pd.m4a"),
            ("อู", "sampleaudio/no/อู 1(1) pd.m4a"),
            ("ไอ", "sampleaudio/no/ไอ 1(1) pd.m4a"),
            ("อำ", "sampleaudio/no/อำ 1(1) pd.m4a"),
            ("เอา", "sampleaudio/no/เอา 1(1) pd.m4a"),
            ("พยางค์ (พา-ทา-คา)", "sampleaudio/no/Pa-ta-ka 1(1) pd.m4a"),
            ("ประโยค", "sampleaudio/no/Sentence 1(1) pd.m4a")
        ]

    # Create columns for audio display
        audio_cols = st.columns(3)

        for i, (title, file_path) in enumerate(sample_audio_files):
            with audio_cols[i % 3]:
                try:
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()
                            st.markdown(f"""
                                <div style="background: white; border-radius: 15px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.1); border-left: 4px solid #6A1B9A;">
                                    <h4 style="color: #4A148C; margin-bottom: 15px; font-family: 'Prompt', sans-serif; font-size: 18px; font-weight: 600; text-align: center;">{title}</h4>
                                </div>
                            """, unsafe_allow_html=True)
                            st.audio(audio_bytes, format="audio/m4a")
                    else:
                        st.markdown(f"""
                            <div style="background: #fff3cd; border-radius: 15px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.1); border-left: 4px solid #ffc107;">
                                <h4 style="color: #856404; margin-bottom: 15px; font-family: 'Prompt', sans-serif; font-size: 18px; font-weight: 600; text-align: center;">{title}</h4>
                                <p style="color: #856404; text-align: center; font-size: 14px;">ไฟล์เสียงไม่พบ</p>
                            </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"""
                        <div style="background: #f8d7da; border-radius: 15px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.1); border-left: 4px solid #dc3545;">
                            <h4 style="color: #721c24; margin-bottom: 15px; font-family: 'Prompt', sans-serif; font-size: 18px; font-weight: 600; text-align: center;">{title}</h4>
                            <p style="color: #721c24; text-align: center; font-size: 14px;">เกิดข้อผิดพลาดในการโหลดไฟล์</p>
                        </div>
                    """, unsafe_allow_html=True)

    # FIXED: Add a final summary section at the bottom for better visibility
        st.markdown("""
            <div style="max-width: 1000px; margin: 0 auto; padding: 0 40px;">
                <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffcc02 20%, #fff3e0 100%); border-radius: 20px; padding: 30px; margin: 40px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.1); border-left: 6px solid #f57c00; text-align: center;">
                    <h3 style="color: #e65100; margin-bottom: 20px; font-family: 'Prompt', sans-serif; font-size: 28px; font-weight: 700;">⚡ พร้อมเริ่มการตรวจแล้ว!</h3>
                    <p style="font-size: 20px; font-family: 'Prompt', sans-serif; line-height: 1.6; color: #bf360c; margin: 0;">
                        เมื่อท่านเข้าใจขั้นตอนและได้ฟังตัวอย่างเสียงแล้ว<br>
                        <strong>กลับไปที่หน้าแรกเพื่อเริ่มใช้งานระบบตรวจคัดกรอง SixtyScan</strong>
                    </p>
                </div>
            </div>

            <!-- Fallback for mobile devices -->
            <style>
                @media (max-width: 768px) {
                    div[style*="display:grid; grid-template-columns:1fr 1fr"] {
                        display: flex !important;
                        flex-direction: column !important;
                    }
                }
            </style>
        </div>
        """, unsafe_allow_html=True)
    
    def show_analysis_page():
        """Display the analysis page"""
        load_css()

        # FIXED: Combine header with analysis content
        analysis_html = f"""
            {get_header_html()}
        """

        st.markdown(analysis_html, unsafe_allow_html=True)

        # Back button
        if st.button("**← กลับหน้าแรก**", key="back_to_home"):
            st.session_state.page = 'home'
            st.rerun()

        # Load model
        model = load_model()
        if not model:
            st.error("Cannot proceed without model. Please verify best_model.pth is present in the repository root.")
            return

        # Clear button logic
        if 'clear_button_clicked' in st.session_state and st.session_state.clear_button_clicked:
            cleanup_all_temp_files()
            st.session_state.vowel_files = []
            st.session_state.pataka_file = None
            st.session_state.sentence_file = None
            st.session_state.clear_clicked = True
            st.session_state.clear_button_clicked = False
            st.session_state.result = None
            st.success("ลบข้อมูลทั้งหมดเรียบร้อยแล้ว", icon="🗑️")
            st.rerun()

        # Vowel recordings
        vowel_card_html = """
        <div class='card'>
            <h2>1. สระ</h2>
            <p class='instructions'>กรุณาออกเสียงแต่ละสระ 5-8 วินาทีอย่างชัดเจน โดยกดปุ่มบันทึกทีละสระ</p>
        </div>
        """
        st.markdown(vowel_card_html, unsafe_allow_html=True)

        vowel_sounds = ["อา", "อี", "อือ", "อู", "ไอ", "อำ", "เอา"]

        for i, sound in enumerate(vowel_sounds):
            st.markdown(f"<p class='pronounce'>ออกเสียง <b>\"{sound}\"</b></p>", unsafe_allow_html=True)

            if not st.session_state.clear_clicked:
                audio_bytes = st.audio_input(f"🎤 บันทึกเสียง {sound}", key=f"vowel_{i}")
                if audio_bytes:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_bytes.read())
                        add_temp_file(tmp.name)
                        while len(st.session_state.vowel_files) <= i:
                            st.session_state.vowel_files.append(None)
                        if st.session_state.vowel_files[i] and os.path.exists(st.session_state.vowel_files[i]):
                            os.unlink(st.session_state.vowel_files[i])
                        st.session_state.vowel_files[i] = tmp.name
                    st.success(f"บันทึกเสียง \"{sound}\" สำเร็จ", icon="✅")
            else:
                st.audio_input(f"🎤 บันทึกเสียง {sound}", key=f"vowel_{i}_new")

            if i < len(st.session_state.vowel_files) and st.session_state.vowel_files[i]:
                fp = st.session_state.vowel_files[i]
                if os.path.exists(fp):
                    with open(fp, 'rb') as f:
                        st.download_button(
                            label=f"⬇️ ดาวน์โหลด {sound}.wav",
                            data=f.read(),
                            file_name=f"vowel_{sound}.wav",
                            mime="audio/wav",
                            key=f"dl_vowel_{i}"
                        )

        # File uploader for vowels
        uploaded_vowels = st.file_uploader("อัปโหลดไฟล์เสียงสระ (7 ไฟล์)", type=["wav", "mp3", "m4a"], accept_multiple_files=True)
        if uploaded_vowels and len([f for f in st.session_state.vowel_files if f is not None]) < 7:
            cleanup_all_temp_files()
            st.session_state.vowel_files = []
            for file in uploaded_vowels[:7]:
                saved_path = save_uploaded_file(file)
                if saved_path:
                    st.session_state.vowel_files.append(saved_path)

        # Pataka recording
        pataka_card_html = """
        <div class='card'>
            <h2>2. พยางค์</h2>
            <p class='instructions'>กรุณาออกเสียงคำว่า <b>"พา - ทา - คา"</b> ให้จบภายใน 6 วินาที</p>
        </div>
        """
        st.markdown(pataka_card_html, unsafe_allow_html=True)

        if not st.session_state.clear_clicked:
            pataka_bytes = st.audio_input("🎤 บันทึกเสียงพยางค์", key="pataka")
            if pataka_bytes:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(pataka_bytes.read())
                    add_temp_file(tmp.name)
                    if st.session_state.pataka_file and os.path.exists(st.session_state.pataka_file):
                        os.unlink(st.session_state.pataka_file)
                    st.session_state.pataka_file = tmp.name
                st.success("บันทึกพยางค์สำเร็จ", icon="✅")
        else:
            pataka_bytes = st.audio_input("🎤 บันทึกเสียงพยางค์", key="pataka_new")

        if st.session_state.pataka_file and os.path.exists(st.session_state.pataka_file):
            with open(st.session_state.pataka_file, 'rb') as f:
                st.download_button(
                    label="⬇️ ดาวน์โหลด pataka.wav",
                    data=f.read(),
                    file_name="pataka.wav",
                    mime="audio/wav",
                    key="dl_pataka"
                )

        # File uploader for pataka
        uploaded_pataka = st.file_uploader("อัปโหลดไฟล์เสียงพยางค์", type=["wav", "mp3", "m4a"], accept_multiple_files=False)
        if uploaded_pataka and not st.session_state.pataka_file:
            saved_path = save_uploaded_file(uploaded_pataka)
            if saved_path:
                st.session_state.pataka_file = saved_path

        # Sentence recording
        sentence_card_html = """
        <div class='card'>
            <h2>3. ประโยค</h2>
            <p class='sentence-instruction'>กรุณาอ่านประโยค <b>"วันนี้อากาศแจ่มใสนกร้องเสียงดังเป็นจังหวะ"</b></p>
        </div>
        """
        st.markdown(sentence_card_html, unsafe_allow_html=True)

        if not st.session_state.clear_clicked:
            sentence_bytes = st.audio_input("🎤 บันทึกการอ่านประโยค", key="sentence")
            if sentence_bytes:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(sentence_bytes.read())
                    add_temp_file(tmp.name)
                    if st.session_state.sentence_file and os.path.exists(st.session_state.sentence_file):
                        os.unlink(st.session_state.sentence_file)
                    st.session_state.sentence_file = tmp.name
                st.success("บันทึกประโยคสำเร็จ", icon="✅")
        else:
            sentence_bytes = st.audio_input("🎤 บันทึกการอ่านประโยค", key="sentence_new")

        if st.session_state.sentence_file and os.path.exists(st.session_state.sentence_file):
            with open(st.session_state.sentence_file, 'rb') as f:
                st.download_button(
                    label="⬇️ ดาวน์โหลด sentence.wav",
                    data=f.read(),
                    file_name="sentence.wav",
                    mime="audio/wav",
                    key="dl_sentence"
                )

        # File uploader for sentence
        uploaded_sentence = st.file_uploader("อัปโหลดไฟล์เสียงประโยค", type=["wav", "mp3", "m4a"], accept_multiple_files=False)
        if uploaded_sentence and not st.session_state.sentence_file:
            saved_path = save_uploaded_file(uploaded_sentence)
            if saved_path:
                st.session_state.sentence_file = saved_path

        # Action buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            predict_btn = st.button("**🔍 วิเคราะห์**", key="predict", type="primary", use_container_width=True)
        with col2:
            if st.button("**🗑️ ลบข้อมูล**", key="clear", type="secondary", use_container_width=True):
                st.session_state.clear_button_clicked = True
                st.rerun()

        # Reset clear_clicked flag
        if st.session_state.clear_clicked:
            st.session_state.clear_clicked = False

        # Prediction logic
        if predict_btn:
            valid_vowel_files = [f for f in st.session_state.vowel_files if f is not None]

            # BYPASS: validation temporarily disabled — restore by removing the `if True:` line and uncommenting the original
            # if len(valid_vowel_files) == 7 and st.session_state.pataka_file and st.session_state.sentence_file:
            if True:  # BYPASS
                with st.spinner("กำลังวิเคราะห์..."):
                    try:
                        final_prob = predict_from_model(
                            valid_vowel_files,
                            st.session_state.pataka_file,
                            st.session_state.sentence_file,
                            model
                        )

                        if final_prob is None:
                            st.error("การวิเคราะห์ล้มเหลว กรุณาลองใหม่อีกครั้ง")
                            return

                        percent = int(final_prob * 100)

                        # Determine risk level and advice
                        if percent <= 50:
                            level = "ระดับต่ำ (Low)"
                            label = "Non Parkinson"
                            diagnosis = "ไม่เป็นพาร์กินสัน"
                            box_color = "#e8f5e9"
                            border_color = "#4caf50"
                            advice_html = """
                            <ul style='font-size:26px; font-family: "Prompt", sans-serif; line-height: 1.6;'>
                                <li>ถ้าไม่มีอาการ: ควรตรวจปีละครั้ง(ไม่บังคับ)</li>
                                <li>ถ้ามีอาการเล็กน้อย: ตรวจปีละ 2 ครั้ง</li>
                                <li>ถ้ามีอาการเตือน: ตรวจ 2–4 ครั้งต่อปี</li>
                            </ul>
                            """
                        elif percent <= 75:
                            level = "ปานกลาง (Moderate)"
                            label = "Parkinson"
                            diagnosis = "เป็นพาร์กินสัน"
                            box_color = "#fff8e1"
                            border_color = "#ff9800"
                            advice_html = """
                            <ul style='font-size:26px; font-family: "Prompt", sans-serif; line-height: 1.6;'>
                                <li>พบแพทย์เฉพาะทางระบบประสาท</li>
                                <li>บันทึกอาการประจำวัน</li>
                                <li>หากได้รับยา: บันทึกผลข้างเคียง</li>
                            </ul>
                            """
                        else:
                            level = "สูง (High)"
                            label = "Parkinson"
                            diagnosis = "เป็นพาร์กินสัน"
                            box_color = "#ffebee"
                            border_color = "#f44336"
                            advice_html = """
                            <ul style='font-size:26px; font-family: "Prompt", sans-serif; line-height: 1.6;'>
                                <li>พบแพทย์เฉพาะทางโดยเร็วที่สุด</li>
                                <li>บันทึกอาการทุกวัน</li>
                                <li>หากได้รับยา: ติดตามผลอย่างละเอียด</li>
                            </ul>
                            """

                        # Store result in session state so the display survives reruns
                        # (e.g. when the download button is pressed). Cleared only by ลบข้อมูล.
                        st.session_state.result = {
                            'label': label,
                            'level': level,
                            'percent': percent,
                            'diagnosis': diagnosis,
                            'box_color': box_color,
                            'border_color': border_color,
                            'advice_html': advice_html,
                            'png': build_result_png(label, level, percent, border_color, box_color),
                        }
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}")
            # else:  # BYPASS: warning suppressed
            #     st.warning("กรุณาอัดเสียงหรืออัปโหลดให้ครบทั้ง 7 สระ พยางค์ และประโยค", icon="⚠")

        # Render stored result — persists across reruns until ลบข้อมูล is pressed
        if st.session_state.get('result'):
            r = st.session_state.result
            results_html = f"""
                <div style='background-color:{r['box_color']}; padding: 40px; border-radius: 20px; font-size: 28px; color: #000000; font-family: "Prompt", sans-serif; border-left: 8px solid {r['border_color']}; box-shadow: 0 8px 32px rgba(0,0,0,0.08); margin: 30px 0;'>
                    <div style='text-align: center; font-size: 48px; font-weight: 700; margin-bottom: 30px; color: {r['border_color']};'>{r['label']}</div>
                    <p style='margin-bottom: 20px;'><b>ระดับความน่าจะเป็น:</b> {r['level']}</p>
                    <p style='margin-bottom: 20px;'><b>ความน่าจะเป็นของพาร์กินสัน:</b> {r['percent']}%</p>
                    <div style='height: 40px; background: linear-gradient(to right, #4caf50, #ff9800, #f44336); border-radius: 20px; margin-bottom: 25px; position: relative; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);'>
                        <div style='position: absolute; left: {r['percent']}%; top: -5px; bottom: -5px; width: 6px; background-color: #333; border-radius: 3px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'></div>
                    </div>
                    <p style='margin-bottom: 20px;'><b>ผลการวิเคราะห์:</b> {r['diagnosis']}</p>
                    <p style='margin-bottom: 15px; font-size: 30px; font-weight: 600;'><b>คำแนะนำ</b></p>
                    {r['advice_html']}
                </div>
            """
            st.markdown(results_html, unsafe_allow_html=True)
            st.download_button(
                label="⬇️ ดาวน์โหลดผลการตรวจ (PNG)",
                data=r['png'],
                file_name="sixtyscan_result.png",
                mime="image/png",
                use_container_width=True,
            )

    # =============================
    # Main App Logic
    # =============================
    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'guide':
        show_guide_page()
    elif st.session_state.page == 'analysis':
        show_analysis_page()

# Run the app
if __name__ == "__main__":
    run_desktop_app()
