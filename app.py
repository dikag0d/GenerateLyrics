"""
Lyrics Generator - Streamlit Application
=========================================
A web application for transcribing audio files into song lyrics
using OpenAI's Whisper model.

Usage:
    streamlit run app.py
"""

import os
import tempfile
import time
from pathlib import Path
import logging

import streamlit as st

# Import our modules
from audio_processing import (
    validate_audio_file,
    preprocess_audio,
    get_audio_duration,
    check_spleeter_available,
    isolate_vocals
)
from transcription import (
    load_model,
    transcribe,
    get_available_models,
    get_model_info,
    get_device,
    estimate_transcription_time
)
from lyrics_formatter import (
    format_as_lyrics,
    generate_lyrics_file_content,
    estimate_song_structure
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="🎤 Lyrics Generator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# Custom Styling
# =============================================================================

st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .stTitle {
        color: #1DB954;
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #1DB954;
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Lyrics display area */
    .lyrics-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    
    /* Success message styling */
    .success-message {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f23 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(29, 185, 84, 0.4);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
    }
    
    /* Info boxes */
    .info-card {
        background: rgba(29, 185, 84, 0.1);
        border-left: 4px solid #1DB954;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_name' not in st.session_state:
        st.session_state.model_name = None
    if 'transcription_result' not in st.session_state:
        st.session_state.transcription_result = None
    if 'lyrics' not in st.session_state:
        st.session_state.lyrics = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False

init_session_state()


# =============================================================================
# Sidebar Configuration
# =============================================================================

def render_sidebar():
    """Render the sidebar with configuration options."""
    
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")
        
        # Model selection
        st.markdown("### 🤖 Whisper Model")
        model_options = get_available_models()
        
        selected_model = st.selectbox(
            "Choose model size:",
            model_options,
            index=1,  # Default to 'small'
            help="Larger models are more accurate but slower"
        )
        
        # Display model info
        model_info = get_model_info(selected_model)
        st.markdown(f"""
        <div class="info-card">
            <b>📦 Size:</b> {model_info['size']}<br>
            <b>⚡ Speed:</b> {model_info['speed']}<br>
            <b>🎯 Accuracy:</b> {model_info['accuracy']}<br>
            <b>💾 VRAM:</b> {model_info['vram']}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Language selection
        st.markdown("### 🌐 Language")
        
        language_options = {
            'Auto-detect': None,
            'Indonesian': 'id',
            'English': 'en',
            'Japanese': 'ja',
            'Korean': 'ko',
            'Chinese': 'zh',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Portuguese': 'pt',
            'Italian': 'it',
            'Dutch': 'nl',
            'Russian': 'ru',
            'Arabic': 'ar',
            'Hindi': 'hi',
            'Thai': 'th',
            'Vietnamese': 'vi',
            'Malay': 'ms'
        }
        
        selected_language = st.selectbox(
            "Song language:",
            list(language_options.keys()),
            index=0,
            help="Select language for better accuracy, or Auto-detect"
        )
        
        st.markdown("---")
        
        # Output options
        st.markdown("### 📝 Output Options")
        
        include_timestamps = st.checkbox(
            "Include timestamps",
            value=False,
            help="Add timestamps at the beginning of each line"
        )
        
        include_metadata = st.checkbox(
            "Include metadata header",
            value=True,
            help="Add header with language and other info"
        )
        
        st.markdown("---")
        
        # Vocal isolation option (if spleeter is available)
        st.markdown("### 🎚️ Audio Enhancement")
        
        spleeter_available = check_spleeter_available()
        
        if spleeter_available:
            use_vocal_isolation = st.checkbox(
                "Isolate vocals",
                value=False,
                help="Remove background music for better accuracy (slower)"
            )
        else:
            use_vocal_isolation = False
            st.info(
                "💡 Install `spleeter` for vocal isolation:\n"
                "`pip install spleeter`"
            )
        
        st.markdown("---")
        
        # Device info
        st.markdown("### 💻 System Info")
        device = get_device()
        device_emoji = "🎮" if device == "cuda" else "💻"
        st.markdown(f"{device_emoji} **Device:** {device.upper()}")
        
        return {
            'model_name': selected_model,
            'language': language_options[selected_language],
            'include_timestamps': include_timestamps,
            'include_metadata': include_metadata,
            'use_vocal_isolation': use_vocal_isolation
        }


# =============================================================================
# Main Application
# =============================================================================

def render_header():
    """Render the application header."""
    
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="
            background: linear-gradient(90deg, #1DB954, #1ed760, #1DB954);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 10px;
        ">🎤 Lyrics Generator</h1>
        <p style="color: #888; font-size: 1.2rem;">
            Transform your songs into clean lyrics with AI-powered transcription
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_file_uploader():
    """Render the file upload section."""
    
    st.markdown("### 📁 Upload Your Audio File")
    
    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=['mp3', 'wav', 'm4a'],
        help="Supported formats: MP3, WAV, M4A"
    )
    
    if uploaded_file:
        # Display file info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 File Name", uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 20 else uploaded_file.name)
        with col2:
            st.metric("📦 File Size", f"{file_size_mb:.2f} MB")
        with col3:
            st.metric("🎵 Format", Path(uploaded_file.name).suffix.upper())
    
    return uploaded_file


def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to a temporary location.
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        Path to the saved file
    """
    # Create temp directory if it doesn't exist
    temp_dir = os.path.join(tempfile.gettempdir(), 'lyrics_generator')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save file
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def process_audio(file_path: str, config: dict) -> dict:
    """
    Process audio file through the transcription pipeline.
    
    Args:
        file_path: Path to the audio file
        config: Configuration dictionary from sidebar
    
    Returns:
        Transcription result dictionary
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Validate audio file
        status_text.text("🔍 Validating audio file...")
        progress_bar.progress(5)
        
        is_valid, error_msg = validate_audio_file(file_path)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Step 2: Get audio duration for time estimation
        duration = get_audio_duration(file_path)
        status_text.text(f"📊 Audio duration: {duration:.1f} seconds")
        progress_bar.progress(10)
        time.sleep(0.5)
        
        # Step 3: Vocal isolation (if enabled)
        audio_path = file_path
        if config['use_vocal_isolation']:
            status_text.text("🎚️ Isolating vocals (this may take a while)...")
            progress_bar.progress(15)
            try:
                audio_path = isolate_vocals(file_path)
                status_text.text("✅ Vocals isolated successfully!")
                progress_bar.progress(30)
                time.sleep(0.5)
            except Exception as e:
                st.warning(f"Vocal isolation failed: {str(e)}. Using original audio.")
        
        # Step 4: Preprocess audio
        status_text.text("⚙️ Preprocessing audio...")
        progress_bar.progress(35)
        processed_path = preprocess_audio(audio_path)
        progress_bar.progress(40)
        
        # Step 5: Load Whisper model
        model_name = config['model_name']
        
        # Check if model is already loaded
        if st.session_state.model is None or st.session_state.model_name != model_name:
            status_text.text(f"🤖 Loading Whisper {model_name} model...")
            progress_bar.progress(45)
            st.session_state.model = load_model(model_name)
            st.session_state.model_name = model_name
        
        progress_bar.progress(55)
        
        # Step 6: Estimate transcription time
        device = get_device()
        estimated_time = estimate_transcription_time(duration, model_name, device)
        status_text.text(
            f"⏱️ Estimated time: ~{estimated_time:.0f} seconds"
        )
        time.sleep(1)
        
        # Step 7: Transcribe
        lang = config.get('language')
        lang_msg = f" (Language: {lang})" if lang else " (Auto-detect)"
        status_text.text(f"🎤 Transcribing audio{lang_msg}... Please wait...")
        progress_bar.progress(60)
        
        result = transcribe(processed_path, st.session_state.model, language=lang)
        
        progress_bar.progress(90)
        status_text.text("✨ Formatting lyrics...")
        time.sleep(0.5)
        
        # Step 8: Complete
        progress_bar.progress(100)
        status_text.text("✅ Transcription complete!")
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return result
        
    except MemoryError as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Memory Error: {str(e)}")
        st.info("💡 Try using a smaller model or a shorter audio file.")
        raise
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error: {str(e)}")
        raise


def display_lyrics(transcription_result: dict, config: dict):
    """
    Display the transcribed lyrics.
    
    Args:
        transcription_result: Whisper transcription result
        config: Configuration dictionary
    """
    segments = transcription_result.get('segments', [])
    language = transcription_result.get('language', 'unknown')
    
    # Format lyrics
    lyrics = format_as_lyrics(
        segments,
        include_timestamps=config['include_timestamps']
    )
    
    # Store in session state
    st.session_state.lyrics = lyrics
    st.session_state.transcription_result = transcription_result
    
    # Display stats
    st.markdown("### 📊 Transcription Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    structure = estimate_song_structure(segments)
    
    with col1:
        st.metric("🌐 Language", language.upper())
    with col2:
        st.metric("📝 Segments", len(segments))
    with col3:
        st.metric("🎵 Est. Verses", structure['estimated_verses'])
    with col4:
        duration_min = structure['total_duration'] / 60
        st.metric("⏱️ Duration", f"{duration_min:.1f} min")
    
    st.markdown("---")
    
    # Display lyrics
    st.markdown("### 🎤 Generated Lyrics")
    
    st.markdown("""
    <div class="lyrics-container">
    </div>
    """, unsafe_allow_html=True)
    
    # Text area for lyrics (editable)
    edited_lyrics = st.text_area(
        "You can edit the lyrics below:",
        value=lyrics,
        height=400,
        key="lyrics_editor"
    )
    
    # Update stored lyrics if edited
    if edited_lyrics != lyrics:
        st.session_state.lyrics = edited_lyrics


def render_download_section(config: dict):
    """Render the download button section."""
    
    if st.session_state.lyrics:
        st.markdown("### 💾 Export Lyrics")
        
        # Generate file content
        if st.session_state.transcription_result:
            file_content = generate_lyrics_file_content(
                st.session_state.transcription_result,
                include_timestamps=config['include_timestamps'],
                include_metadata=config['include_metadata']
            )
        else:
            file_content = st.session_state.lyrics
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download as lyrics.txt",
                data=file_content,
                file_name="lyrics.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # Also offer timestamped version if not already included
            if not config['include_timestamps'] and st.session_state.transcription_result:
                timestamped_content = generate_lyrics_file_content(
                    st.session_state.transcription_result,
                    include_timestamps=True,
                    include_metadata=config['include_metadata']
                )
                st.download_button(
                    label="📥 Download with timestamps",
                    data=timestamped_content,
                    file_name="lyrics_timestamped.txt",
                    mime="text/plain",
                    use_container_width=True
                )


def main():
    """Main application entry point."""
    
    # Render sidebar and get configuration
    config = render_sidebar()
    
    # Render header
    render_header()
    
    st.markdown("---")
    
    # File upload section
    uploaded_file = render_file_uploader()
    
    st.markdown("---")
    
    # Process button
    if uploaded_file:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button(
                "🚀 Generate Lyrics",
                use_container_width=True,
                disabled=st.session_state.processing
            ):
                st.session_state.processing = True
                
                try:
                    # Save uploaded file
                    file_path = save_uploaded_file(uploaded_file)
                    
                    # Process audio
                    result = process_audio(file_path, config)
                    
                    # Display results
                    display_lyrics(result, config)
                    
                    # Show success message
                    st.success("🎉 Lyrics generated successfully!")
                    
                except Exception as e:
                    logger.error(f"Processing failed: {str(e)}")
                
                finally:
                    st.session_state.processing = False
    
    # If we have existing lyrics, display them
    elif st.session_state.lyrics:
        st.markdown("### 🎤 Previous Lyrics")
        st.text_area(
            "Previously generated lyrics:",
            value=st.session_state.lyrics,
            height=400,
            disabled=True
        )
    
    # Download section
    if st.session_state.lyrics:
        st.markdown("---")
        render_download_section(config)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🎵 Powered by OpenAI Whisper | Built with Streamlit</p>
        <p style="font-size: 0.8rem;">
            Tip: For best results, use high-quality audio files with clear vocals
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
