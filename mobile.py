import streamlit as st
import base64
import os
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
import gdown
from datetime import datetime
import pytz
import atexit
from pathlib import Path

# Configuration for mobile
CONFIG = {
    'MODEL_PATH': "best_resnet18.pth",
    'MODEL_URL': "https://drive.google.com/uc?id=1_oHE9B-2PgSqpTQCC9HrG7yO0rsnZtqs",
    'CSS_FILE': "mobilestyle.css",
    'LOGO_PATHS': ["logo.png", "./logo.png", "assets/logo.png", "images/logo.png"],
    'IMAGE_PATHS': ["insert.jpg", "./insert.jpg", "assets/insert.jpg", "images/insert.jpg"],
    'DOCTOR_PATHS' : ["doctor.jpg", "./doctor.jpg", "assets/doctor.jpg", "images/doctor.jpg"],
    'REWARD_PATHS' : ["reward.jpg", "./reward.jpg", "assets/reward.jpg", "images/reward.jpg"],
    'PRESENT_PATHS' : ["present.jpg", "./present.jpg", "assets/present.jpg", "images/present.jpg"],
    'DOCTOR2_PATHS' : ["doctor2.jpg", "./doctor2.jpg", "assets/doctor2.jpg", "images/doctor2.jpg"],
    'THAI_TIMEZONE': 'Asia/Bangkok'
}

# ===== Classification index mapping (must match your training & computer.py) =====
# If training order is [Non-PD, PD], set = 1. If [PD, Non-PD], set = 0.
CLASS_INDEX_PD = 1  

def pd_probability(logits: torch.Tensor) -> float:
    """
    Return PD probability using the configured CLASS_INDEX_PD.
    Ensures results come directly from the model without index mismatch.
    """
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError(f"Expected logits shape [N, 2], got {tuple(logits.shape)}")
    probs = F.softmax(logits, dim=1)[0]
    return float(probs[CLASS_INDEX_PD].item())


# Mobile-optimized page config for iPhone 13 (2532x1170 resolution, 460 ppi)
if __name__ == "__main__":
    st.set_page_config(
        page_title="SixtyScan Mobile - Parkinson Detection",
        page_icon="🎤",
        layout="centered",  # Changed from "wide" to "centered" for mobile
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
    """Load the ResNet18 model with robust checkpoint handling."""
    try:
        if not os.path.exists(CONFIG['MODEL_PATH']):
            with st.spinner("Downloading model..."):
                gdown.download(CONFIG['MODEL_URL'], CONFIG['MODEL_PATH'], quiet=False)
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
        'temp_files': []  # Track all temp files for cleanup
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

def run_mobile_app():
    """Main function to run the mobile version"""
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
            # Fallback minimal CSS for mobile
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
            st.error("pydub library is required. Please install it with: pip install pydub")
            return None
        except Exception as e:
            st.error(f"Error converting audio file: {str(e)}")
            return None

    def audio_to_mel_tensor(file_path):
        """Convert audio file to mel spectrogram tensor"""
        try:
            # Convert to WAV if necessary
            wav_file = convert_to_wav_if_needed(file_path)
            if not wav_file:
                return None

            if os.path.getsize(wav_file) > 50 * 1024 * 1024:
                st.error("ไฟล์เสียงใหญ่เกินไป (สูงสุด 50MB)")
                return None

            # === CHANGED: match training SR behavior (no forced resample) ===
            y, sr = librosa.load(wav_file, sr=None, mono=True)

            duration = len(y) / sr
            if duration > 30:
                st.error(f"เสียงยาวเกินไป ({duration:.1f}s) กรุณาบันทึกไม่เกิน 30 วินาที")
                return None

            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)

            fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
            ax.axis('off')
            librosa.display.specshow(mel_db, sr=sr, ax=ax)

            buf = io.BytesIO()
            try:
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            finally:
                plt.close(fig)

            buf.seek(0)
            image = Image.open(buf).convert('RGB')

            # === CHANGED: add ImageNet normalization (like training) ===
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std =[0.229, 0.224, 0.225]),
            ])

            return transform(image).unsqueeze(0)
        except Exception as e:
            st.error(f"Error creating mel tensor: {str(e)}")
            return None

    def predict_from_model(vowel_paths, pataka_path, sentence_path, model):
        """Predict PD probability by averaging logits (matches desktop, more stable than prob-space)."""
        try:
            tensors = []

            # Process vowel files
            for path in vowel_paths:
                t = audio_to_mel_tensor(path)
                if t is None:
                    st.error(f"Failed to process vowel file: {path}")
                    return None
                tensors.append(t)

            # Process pataka and sentence
            for path in [pataka_path, sentence_path]:
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

                # Average in logit space, then softmax (matches desktop)
                logits_all = torch.cat(logits_list, dim=0)      # [N, 2]
                mean_logits = logits_all.mean(dim=0)             # [2]
                probs = torch.softmax(mean_logits, dim=0)        # [2]
                return float(probs[CLASS_INDEX_PD].item())

        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            return None

    # =============================
    # Page Functions
    # =============================
    def get_header_html():
        """Get the mobile-optimized header HTML"""
        logo_b64 = load_image_file(CONFIG['LOGO_PATHS'], "SixtyScan Logo")
        
        logo_html = ""
        if logo_b64:
            logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="mobile-header-logo" alt="SixtyScan Logo">'
        
        return f"""
            <div class="mobile-header-container">
                <div class="mobile-logo-section">
                    {logo_html}
                    <div class="mobile-logo-text">SixtyScan</div>
                </div>
                <div class="mobile-tagline">นวัตกรรมคัดกรองโรคพาร์กินสันจากเสียง</div>
            </div>
        """

    def show_home_page():
        """Display the mobile-optimized home page"""
        load_css()

        woman_image_b64 = load_image_file(CONFIG['IMAGE_PATHS'], "Woman using phone")

        # Mobile-optimized layout - keeping your original CSS classes
        combined_html = f"""
            {get_header_html()}
            <div class="mobile-main-content">
                <div class="mobile-content-wrapper">
                    <div class="mobile-text-section">
                        <h1 class="mobile-main-title">
                            ตรวจเช็คโรค<br>พาร์กินสันทันที<br> ด้วย <span class="mobile-highlight">SixtyScan</span>
                        </h1>
                    </div>
                    <div class="mobile-image-section">
                        {f'<img src="data:image/jpg;base64,{woman_image_b64}" alt="Woman using phone" class="mobile-main-image">' if woman_image_b64 else '''
                        <div class="mobile-image-placeholder">
                            <div class="mobile-placeholder-content">
                                <div class="mobile-placeholder-icon">📱</div>
                                <div class="mobile-placeholder-text">
                                    insert.jpg<br>not found
                                </div>
                            </div>
                        </div>
                        '''}
                    </div>
                </div>
            </div>
        """

        st.markdown(combined_html, unsafe_allow_html=True)

        # Mobile-optimized buttons - keeping your CSS classes
        st.markdown('<div class="mobile-homepage-buttons-wrapper">', unsafe_allow_html=True)

        if st.button("**เริ่มใช้งาน**", key="mobile_start_analysis"):
            st.session_state.page = 'analysis'
            st.rerun()

        if st.button("**คู่มือ**", key="mobile_guide_manual"):
            st.session_state.page = 'guide'
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # ============ About Us Section ============ (Improved "เกี่ยวกับเรา" heading with better spacing)
        st.markdown("""
            <div style="margin-top:40px; padding:25px; background:linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%); border-radius:25px; box-shadow:0 8px 32px rgba(74, 20, 140, 0.1); border:1px solid rgba(74, 20, 140, 0.05);">
                <div style="text-align:center; margin-bottom:20px;">
                    <h2 style="color:#4A148C; font-family:'Prompt',sans-serif; font-size:28px; font-weight:700; margin:0; padding-top: 15px;">
                        เกี่ยวกับเรา
                    </h2>
                    <div style="width: 60px; height: 3px; background: linear-gradient(135deg, #4A148C, #7B1FA2); margin: 8px auto; border-radius: 2px;"></div>
                </div>
                <p style="font-size:16px; line-height:1.8; text-align:left; font-family:'Prompt',sans-serif; margin-bottom:18px; color:#2c2c2c;">
                    แรงบันดาลใจของ <strong style="color:#4A148C;">SixtyScan.life</strong> เริ่มจากคนใกล้ตัวที่บ้านของเรา ที่เป็นผู้ป่วยโรคพาร์กินสัน 
                    ได้เห็นถึงความยากลำบากของท่านและผู้ที่เกี่ยวข้องทุกคน จึงเกิดคำถามว่า 
                    <em>"ถ้าช่วยผู้คนเข้าถึงการรักษาได้เร็ว จะช่วยสังคมได้มาก"</em>
                </p>
                <p style="font-size:16px; line-height:1.8; text-align:left; font-family:'Prompt',sans-serif; margin-bottom:20px; color:#2c2c2c;">
                    ด้วยความตั้งใจนั้น จึงนำความคิดไปปรึกษาคุณครู จนได้รวมทีมกัน 
                    ใช้เทคโนโลยีพัฒนาเป็น <strong style="color:#4A148C;">SixtyScan.life</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Doctor section with individual captions - Updated to show both doctors
        try:
            doctor_image_b64 = load_image_file(CONFIG['DOCTOR_PATHS'], "doctor")
            doctor2_image_b64 = load_image_file(CONFIG['DOCTOR2_PATHS'], "doctor2")
            
            doctors_html = '<div style="text-align:center; margin:25px 0; padding:0 20px;">'
            
            if doctor_image_b64 and doctor2_image_b64:
                doctors_html += f"""
                    <div style="display:flex; flex-direction:column; gap:25px; align-items:center;">
                        <div style="text-align:center;">
                            <img src="data:image/jpg;base64,{doctor_image_b64}" alt="นพ.ณัฐฏ์ กล้าผจญ" style="max-width:85%; border-radius:15px; box-shadow:0 4px 16px rgba(0,0,0,0.1);">
                            <p style="font-size:16px; color:#4A148C; font-family:'Prompt',sans-serif; line-height:1.5; margin-top:12px; font-weight:600;">
                                นพ.ณัฐฏ์ กล้าผจญ
                            </p>
                        </div>
                        <div style="text-align:center;">
                            <img src="data:image/jpg;base64,{doctor2_image_b64}" alt="ผศ.นพ.สุรัตน์ ตันประเวช" style="max-width:85%; border-radius:15px; box-shadow:0 4px 16px rgba(0,0,0,0.1);">
                            <p style="font-size:16px; color:#4A148C; font-family:'Prompt',sans-serif; line-height:1.5; margin-top:12px; font-weight:600;">
                                ผศ.นพ.สุรัตน์ ตันประเวช
                            </p>
                        </div>
                    </div>
                """
            elif doctor_image_b64:
                doctors_html += f"""
                    <div style="text-align:center;">
                        <img src="data:image/jpg;base64,{doctor_image_b64}" alt="นพ.ณัฐฏ์ กล้าผจญ" style="max-width:85%; border-radius:15px; box-shadow:0 4px 16px rgba(0,0,0,0.1); margin-bottom:12px;">
                        <p style="font-size:16px; color:#4A148C; font-family:'Prompt',sans-serif; font-weight:600;">นพ.ณัฐฏ์ กล้าผจญ</p>
                    </div>
                """
            elif doctor2_image_b64:
                doctors_html += f"""
                    <div style="text-align:center;">
                        <img src="data:image/jpg;base64,{doctor2_image_b64}" alt="ผศ.นพ.สุรัตน์ ตันประเวช" style="max-width:85%; border-radius:15px; box-shadow:0 4px 16px rgba(0,0,0,0.1); margin-bottom:12px;">
                        <p style="font-size:16px; color:#4A148C; font-family:'Prompt',sans-serif; font-weight:600;">ผศ.นพ.สุรัตน์ ตันประเวช</p>
                    </div>
                """
            
            doctors_html += '</div>'
            st.markdown(doctors_html, unsafe_allow_html=True)
            
        except Exception as e:
            st.warning(f"Could not load doctor image: {e}")

        # Continuation of about section with improved spacing and left-aligned text
        st.markdown("""
            <div style="padding:25px; background:linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%); margin-top:10px; border-radius:25px; box-shadow:0 8px 32px rgba(74, 20, 140, 0.1); border:1px solid rgba(74, 20, 140, 0.05);">
                <p style="font-size:16px; line-height:1.8; text-align:left; font-family:'Prompt',sans-serif; margin-bottom:0; color:#2c2c2c;">
                    จากแนวคิดนี้ เราได้รับรางวัลจาก <strong style="color:#4A148C;">AI Builder 2025</strong> 
                    และปัจจุบันพวกเรามีโอกาสทำงานร่วมกับแพทย์ผู้เชี่ยวชาญด้านประสาทวิทยา<br><br>
                    ได้แก่ <strong>นพ.ณัฐฏ์ กล้าผจญ</strong> และ<br><strong>ผศ.นพ.สุรัตน์ ตันประเวช</strong><br>
                    จาก <strong style="color:#4A148C;">MED CMU Health Innovation Center (MedCHIC) มหาวิทยาลัยเชียงใหม่</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Award images with error handling
        try:
            reward_image_b64 = load_image_file(CONFIG['REWARD_PATHS'], "reward")
            present_image_b64 = load_image_file(CONFIG['PRESENT_PATHS'], "present")
            
            images_html = '<div style="display:flex; flex-direction:column; align-items:center; gap:15px; margin-top:20px; padding:0 20px;">'
            
            if reward_image_b64:
                images_html += f'<img src="data:image/jpg;base64,{reward_image_b64}" alt="AI Builder Award" style="max-width:90%; border-radius:15px; box-shadow:0 4px 16px rgba(0,0,0,0.1);">'
            
            if present_image_b64:
                images_html += f'<img src="data:image/jpg;base64,{present_image_b64}" alt="Award Presentation" style="max-width:90%; border-radius:15px; box-shadow:0 4px 16px rgba(0,0,0,0.1);">'
            
            images_html += '<p style="font-size:14px; color:#666; font-family:\'Prompt\',sans-serif; margin-top:10px; text-align:center;">ภาพจากการได้รับรางวัล AI Builder 2025</p></div>'
            
            st.markdown(images_html, unsafe_allow_html=True)
            
        except Exception as e:
            st.warning(f"Could not load award images: {e}")

        # ============ Contact Section ============ (Fixed address line breaks)
        st.markdown("""
            <div style="margin-top:25px; padding:25px; background:linear-gradient(135deg, #e3f2fd 0%, #ffffff 100%); border-radius:25px; box-shadow:0 8px 32px rgba(21, 101, 192, 0.1); border:1px solid rgba(21, 101, 192, 0.1);">
                <h2 style="text-align:center; color:#1565C0; font-family:'Prompt',sans-serif; margin-bottom:20px; font-size:24px; font-weight:600;"> ติดต่อเรา</h2>
                <div style="background:rgba(255,255,255,0.7); padding:20px; border-radius:15px; margin-bottom:15px;">
                    <p style="font-size:15px; line-height:1.6; font-family:'Prompt',sans-serif; text-align:center; margin-bottom:0; color:#2c2c2c;">
                        📍 121/11 อาคารอีคิวสแควร์<br>
                        ถนนเชียงใหม่-ฮอด ตำบลป่าแดด<br>
                        อำเภอเมืองเชียงใหม่<br>
                        จังหวัดเชียงใหม่ 50100
                    </p>
                </div>
                <div style="background:rgba(255,255,255,0.7); padding:15px; border-radius:15px; text-align:center;">
                    <p style="font-size:18px; font-weight:600; color:#2e7d32; font-family:'Prompt',sans-serif; margin:0;">
                        📞 064-9506228
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    def show_guide_page():
        """Display the guide/manual page with mobile-responsive styling"""
        load_css()

        # Mobile-responsive header with back button and title
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

        # Guide content - Mobile-optimized version
        st.markdown("""
            <div style="max-width: 1000px; margin: 0 auto; padding: 0 16px;">
                <div style="background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); margin-bottom: 20px;">
                    <h2 style="color: #4A148C; font-size: clamp(24px, 5vw, 36px); margin-bottom: 16px; margin-top: 0; font-family: 'Prompt', sans-serif; line-height: 1.2;">การเตรียมตัวก่อนการตรวจ</h2>
                    <div style="font-size: clamp(16px, 4vw, 22px); line-height: 1.6; font-family: 'Prompt', sans-serif; margin-top: 0; padding-left: 16px;">
                        <div style="margin-bottom: 12px;"><strong>1.</strong> พักผ่อนเพียงพอก่อนการตรวจ</div>
                        <div style="margin-bottom: 12px;"><strong>2.</strong> หาสถานที่เงียบ ปราศจากเสียงรบกวน</div>
                        <div style="margin-bottom: 12px;"><strong>3.</strong> นั่งหรือยืนในท่าที่สบาย</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="max-width: 1000px; margin: 0 auto; padding: 0 16px;">
                <div style="background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); margin-bottom: 20px;">
                    <h2 style="color: #4A148C; font-size: clamp(24px, 5vw, 36px); margin-bottom: 16px; margin-top: 0; font-family: 'Prompt', sans-serif; line-height: 1.2;">ขั้นตอนการตรวจ</h2>
                    <ul style="font-size: clamp(16px, 4vw, 22px); line-height: 1.6; font-family: 'Prompt', sans-serif; margin-top: 0; padding-left: 16px; list-style-position: outside;">
                        <li style="margin-bottom: 16px; padding-left: 4px;"><strong>การออกเสียงสระ:</strong> ออกเสียงสระแต่ละตัว 5-8 วินาที ให้ชัดเจนและคงที่</li>
                        <li style="margin-bottom: 16px; padding-left: 4px;"><strong>การออกเสียงพยางค์:</strong> ออกเสียง "พา-ทา-คา" ซ้ำๆ ประมาณ 6 วินาที</li>
                        <li style="margin-bottom: 16px; padding-left: 4px;"><strong>การอ่านประโยค:</strong> อ่านประโยคที่กำหนดให้อย่างเป็นธรรมชาติและชัดเจน</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style="max-width: 1000px; margin: 0 auto; padding: 0 16px;">
                <div style="background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); margin-bottom: 20px;">
                    <h2 style="color: #4A148C; font-size: clamp(24px, 5vw, 36px); margin-bottom: 16px; margin-top: 0; font-family: 'Prompt', sans-serif; line-height: 1.2;">ข้อควรระวัง</h2>
                    <ul style="font-size: clamp(16px, 4vw, 22px); line-height: 1.6; color: #d32f2f; font-family: 'Prompt', sans-serif; margin-top: 0; padding-left: 16px; list-style-position: outside;">
                        <li style="margin-bottom: 12px; padding-left: 4px;"><strong style="font-weight: 600;">ระบบนี้เป็นเพียงการตรวจคัดกรองเบื้องต้น</strong></li>
                        <li style="margin-bottom: 12px; padding-left: 4px;"><strong style="font-weight: 600;">ไม่สามารถทดแทนการวินิจฉัยโดยแพทย์ได้</strong></li>
                        <li style="margin-bottom: 12px; padding-left: 4px;"><strong style="font-weight: 600;">หากมีข้อสงสัยควรปรึกษาแพทย์เฉพาะทาง</strong></li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Additional tips section - Mobile optimized
        st.markdown("""
            <div style="max-width: 1000px; margin: 0 auto; padding: 0 16px;">
                <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border-radius: 15px; padding: 20px; margin: 20px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.1); border-left: 4px solid #1976d2;">
                    <h3 style="color: #1565c0; margin-bottom: 16px; font-family: 'Prompt', sans-serif; font-size: clamp(20px, 4.5vw, 24px); font-weight: 600; text-align: center; line-height: 1.2;">💡 คำแนะนำเพิ่มเติม</h3>
                    <ul style="font-size: clamp(16px, 4vw, 22px); font-family: 'Prompt', sans-serif; line-height: 1.7; color: #2e7d32; margin: 0; padding-left: 16px; list-style-position: outside;">
                        <li style="margin-bottom: 12px; padding-left: 4px;">ฟังตัวอย่างเสียงก่อนเริ่มการตรวจเพื่อเข้าใจรูปแบบการออกเสียงที่ถูกต้อง</li>
                        <li style="margin-bottom: 12px; padding-left: 4px;">พยายามออกเสียงให้เหมือนกับตัวอย่างให้มากที่สุด</li>
                        <li style="margin-bottom: 12px; padding-left: 4px;">หากไม่แน่ใจ สามารถฟังตัวอย่างซ้ำได้หลายครั้ง</li>
                        <li style="margin-bottom: 12px; padding-left: 4px;">ตัวอย่างเสียงเหล่านี้เป็นเสียงจากผู้ที่ไม่เป็นพาร์กินสัน</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Sample audio section - Mobile optimized
        st.markdown("""
            <div style="max-width: 1000px; margin: 0 auto; padding: 0 16px;">
                <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; padding: 20px; margin: 20px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.1);">
                    <h3 style="color: #495057; margin-bottom: 20px; font-family: 'Prompt', sans-serif; font-size: clamp(20px, 4.5vw, 24px); font-weight: 600; text-align: center; line-height: 1.2;">🎵 ตัวอย่างเสียงที่ถูกต้อง</h3>
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

        # Create responsive audio display - single column on mobile, multiple on desktop
        st.markdown("""
            <style>
            @media (max-width: 768px) {
                .audio-grid {
                    display: flex !important;
                    flex-direction: column !important;
                    gap: 12px;
                }
                .audio-item {
                    width: 100% !important;
                }
            }
            @media (min-width: 769px) {
                .audio-grid {
                    display: grid !important;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)) !important;
                    gap: 16px;
                }
            }
            </style>
            <div style="max-width: 1000px; margin: 0 auto; padding: 0 16px;">
                <div class="audio-grid">
        """, unsafe_allow_html=True)

        for i, (title, file_path) in enumerate(sample_audio_files):
            try:
                if os.path.exists(file_path):
                    with open(file_path, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                        st.markdown(f"""
                            <div class="audio-item" style="background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 3px solid #6A1B9A;">
                                <h4 style="color: #4A148C; margin-bottom: 12px; font-family: 'Prompt', sans-serif; font-size: clamp(16px, 3.5vw, 18px); font-weight: 600; text-align: center; line-height: 1.2;">{title}</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        st.audio(audio_bytes, format="audio/m4a")
                else:
                    st.markdown(f"""
                        <div class="audio-item" style="background: #ffebee; border-radius: 12px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 3px solid #f44336;">
                            <h4 style="color: #d32f2f; margin-bottom: 12px; font-family: 'Prompt', sans-serif; font-size: clamp(16px, 3.5vw, 18px); font-weight: 600; text-align: center; line-height: 1.2;">{title}</h4>
                            <p style="text-align: center; color: #666; font-size: 14px;">ไฟล์ไม่พบ: {file_path}</p>
                        </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading audio file {file_path}: {str(e)}")

        st.markdown("""
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style="max-width: 1000px; margin: 0 auto; padding: 0 40px;">
                <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffcc02 20%, #fff3e0 100%); border-radius: 20px; padding: 30px; margin: 40px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.1); border-left: 6px solid #f57c00; text-align: center;">
                    <h3 style="color: #e65100; margin-bottom: 20px; font-family: 'Prompt', sans-serif; font-size: 28px; font-weight: 700;">⚡ พร้อมเริ่มการตรวจแล้ว!</h3>
                    <p style="font-size: 19px; font-family: 'Prompt', sans-serif; line-height: 1.6; color: #bf360c; margin: 0;">
                        เมื่อท่านเข้าใจขั้นตอนและได้ฟังตัวอย่างเสียงแล้ว<br>
                        <strong>กลับไปที่หน้าแรกเพื่อเริ่มใช้งานระบบตรวจคัดกรอง SixtyScan</strong>
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    def show_analysis_page():
        """Display the mobile-optimized analysis page"""
        load_css()
        
        analysis_html = f"""
            {get_header_html()}
        """
        
        st.markdown(analysis_html, unsafe_allow_html=True)
        
        # Back button
        if st.button("**← กลับหน้าแรก**", key="mobile_back_to_home"):
            st.session_state.page = 'home'
            st.rerun()
        
        # Load model
        model = load_model()
        if not model:
            st.error("Cannot proceed without model. Please check your internet connection and try again.")
            return

        # Clear button logic
        if 'clear_button_clicked' in st.session_state and st.session_state.clear_button_clicked:
            cleanup_all_temp_files()
            st.session_state.vowel_files = []
            st.session_state.pataka_file = None
            st.session_state.sentence_file = None
            st.session_state.clear_clicked = True
            st.session_state.clear_button_clicked = False
            st.success("ลบข้อมูลทั้งหมดเรียบร้อยแล้ว", icon="🗑️")
            st.rerun()

        # Mobile-optimized vowel recordings
        vowel_card_html = """
        <div class='mobile-card'>
            <h2 class='mobile-card-title'>1. สระ</h2>
            <p class='mobile-instructions'>กรุณาออกเสียงแต่ละสระ 5-8 วินาทีอย่างชัดเจน โดยกดปุ่มบันทึกทีละสระ</p>
        </div>
        """
        st.markdown(vowel_card_html, unsafe_allow_html=True)

        vowel_sounds = ["อา", "อี", "อือ", "อู", "ไอ", "อำ", "เอา"]

        for i, sound in enumerate(vowel_sounds):
            st.markdown(f"<p class='mobile-pronounce'>ออกเสียง <b>\"{sound}\"</b></p>", unsafe_allow_html=True)
            
            if not st.session_state.clear_clicked:
                audio_bytes = st.audio_input(f"🎤 บันทึกเสียง {sound}", key=f"mobile_vowel_{i}")
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
                st.audio_input(f"🎤 บันทึกเสียง {sound}", key=f"mobile_vowel_{i}_new")
            
            if i < len(st.session_state.vowel_files) and st.session_state.vowel_files[i]:
                fp = st.session_state.vowel_files[i]
                if os.path.exists(fp):
                    with open(fp, 'rb') as f:
                        st.download_button(
                            label=f"⬇️ ดาวน์โหลด {sound}.wav",
                            data=f.read(),
                            file_name=f"vowel_{sound}.wav",
                            mime="audio/wav",
                            key=f"dl_mobile_vowel_{i}"
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

        # Mobile-optimized pataka recording
        pataka_card_html = """
        <div class='mobile-card'>
            <h2 class='mobile-card-title'>2. พยางค์</h2>
            <p class='mobile-instructions'>กรุณาออกเสียงคำว่า <b>"พา - ทา - คา"</b> ให้จบภายใน 6 วินาที</p>
        </div>
        """
        st.markdown(pataka_card_html, unsafe_allow_html=True)

        if not st.session_state.clear_clicked:
            pataka_bytes = st.audio_input("🎤 บันทึกเสียงพยางค์", key="mobile_pataka")
            if pataka_bytes:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(pataka_bytes.read())
                    add_temp_file(tmp.name)
                    if st.session_state.pataka_file and os.path.exists(st.session_state.pataka_file):
                        os.unlink(st.session_state.pataka_file)
                    st.session_state.pataka_file = tmp.name
                st.success("บันทึกพยางค์สำเร็จ", icon="✅")
        else:
            pataka_bytes = st.audio_input("🎤 บันทึกเสียงพยางค์", key="mobile_pataka_new")

        if st.session_state.pataka_file and os.path.exists(st.session_state.pataka_file):
            with open(st.session_state.pataka_file, 'rb') as f:
                st.download_button(
                    label="⬇️ ดาวน์โหลด pataka.wav",
                    data=f.read(),
                    file_name="pataka.wav",
                    mime="audio/wav",
                    key="dl_mobile_pataka"
                )

        # File uploader for pataka
        uploaded_pataka = st.file_uploader("อัปโหลดไฟล์เสียงพยางค์", type=["wav", "mp3", "m4a"], accept_multiple_files=False)
        if uploaded_pataka and not st.session_state.pataka_file:
            saved_path = save_uploaded_file(uploaded_pataka)
            if saved_path:
                st.session_state.pataka_file = saved_path

        # Mobile-optimized sentence recording
        sentence_card_html = """
        <div class='mobile-card'>
            <h2 class='mobile-card-title'>3. ประโยค</h2>
            <p class='mobile-sentence-instruction'>กรุณาอ่านประโยค <b>"วันนี้อากาศแจ่มใสนกร้องเสียงดังเป็นจังหวะ"</b></p>
        </div>
        """
        st.markdown(sentence_card_html, unsafe_allow_html=True)

        if not st.session_state.clear_clicked:
            sentence_bytes = st.audio_input("🎤 บันทึกการอ่านประโยค", key="mobile_sentence")
            if sentence_bytes:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(sentence_bytes.read())
                    add_temp_file(tmp.name)
                    if st.session_state.sentence_file and os.path.exists(st.session_state.sentence_file):
                        os.unlink(st.session_state.sentence_file)
                    st.session_state.sentence_file = tmp.name
                st.success("บันทึกประโยคสำเร็จ", icon="✅")
        else:
            sentence_bytes = st.audio_input("🎤 บันทึกการอ่านประโยค", key="mobile_sentence_new")

        if st.session_state.sentence_file and os.path.exists(st.session_state.sentence_file):
            with open(st.session_state.sentence_file, 'rb') as f:
                st.download_button(
                    label="⬇️ ดาวน์โหลด sentence.wav",
                    data=f.read(),
                    file_name="sentence.wav",
                    mime="audio/wav",
                    key="dl_mobile_sentence"
                )

        # File uploader for sentence
        uploaded_sentence = st.file_uploader("อัปโหลดไฟล์เสียงประโยค", type=["wav", "mp3", "m4a"], accept_multiple_files=False)
        if uploaded_sentence and not st.session_state.sentence_file:
            saved_path = save_uploaded_file(uploaded_sentence)
            if saved_path:
                st.session_state.sentence_file = saved_path

        # Mobile-optimized action buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            predict_btn = st.button("**🔍 วิเคราะห์**", key="mobile_predict", type="primary", use_container_width=True)
        with col2:
            if st.button("**🗑️ ลบข้อมูล**", key="mobile_clear", type="secondary", use_container_width=True):
                st.session_state.clear_button_clicked = True
                st.rerun()

        # Reset clear_clicked flag
        if st.session_state.clear_clicked:
            st.session_state.clear_clicked = False

        # Prediction logic
        if predict_btn:
            valid_vowel_files = [f for f in st.session_state.vowel_files if f is not None]
            
            if len(valid_vowel_files) == 7 and st.session_state.pataka_file and st.session_state.sentence_file:
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
                            <ul class='mobile-advice-list'>
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
                            <ul class='mobile-advice-list'>
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
                            <ul class='mobile-advice-list'>
                                <li>พบแพทย์เฉพาะทางโดยเร็วที่สุด</li>
                                <li>บันทึกอาการทุกวัน</li>
                                <li>หากได้รับยา: ติดตามผลอย่างละเอียด</li>
                            </ul>
                            """

                        # Mobile-optimized results display
                        results_html = f"""
                            <div class='mobile-results-container' style='background-color:{box_color}; border-left: 8px solid {border_color};'>
                                <div class='mobile-results-label' style='color: {border_color};'>{label}</div>
                                <p class='mobile-results-text'><b>ระดับความน่าจะเป็น:</b> {level}</p>
                                <p class='mobile-results-text'><b>ความน่าจะเป็นของพาร์กินสัน:</b> {percent}%</p>
                                <div class='mobile-progress-bar'>
                                    <div class='mobile-progress-indicator' style='left: {percent}%;'></div>
                                </div>
                                <p class='mobile-results-text'><b>ผลการวิเคราะห์:</b> {diagnosis}</p>
                                <p class='mobile-advice-title'><b>คำแนะนำ</b></p>
                                {advice_html}
                            </div>
                        """
                        st.markdown(results_html, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}")
            else:
                st.warning("กรุณาอัดเสียงหรืออัปโหลดให้ครบทั้ง 7 สระ พยางค์ และประโยค", icon="⚠")

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
    run_mobile_app()
