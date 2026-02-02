"""
Audio Processing Module
=======================
Handles audio file loading, format conversion, preprocessing,
and optional vocal isolation for the lyrics generator app.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import logging

import librosa
import numpy as np
from pydub import AudioSegment
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Audio Loading and Conversion
# =============================================================================

def load_audio(file_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load an audio file using librosa.
    
    Args:
        file_path: Path to the audio file
        sr: Target sample rate (default 16000 for Whisper)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is corrupted or invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        # Load audio with librosa, resampling to target sample rate
        audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
        logger.info(f"Loaded audio: {len(audio)/sample_rate:.2f} seconds at {sample_rate}Hz")
        return audio, sample_rate
    except Exception as e:
        raise ValueError(f"Failed to load audio file: {str(e)}")


def convert_to_wav(input_path: str, output_dir: Optional[str] = None) -> str:
    """
    Convert audio file to WAV format for optimal Whisper processing.
    
    Args:
        input_path: Path to the input audio file (mp3, m4a, etc.)
        output_dir: Directory for the output WAV file (uses temp dir if None)
    
    Returns:
        Path to the converted WAV file
    
    Raises:
        ValueError: If conversion fails
    """
    try:
        # Determine input format from extension
        file_ext = Path(input_path).suffix.lower()
        
        # Load audio based on format
        if file_ext == '.mp3':
            audio = AudioSegment.from_mp3(input_path)
        elif file_ext == '.m4a':
            audio = AudioSegment.from_file(input_path, format='m4a')
        elif file_ext == '.wav':
            # Already WAV, just return the path
            return input_path
        else:
            # Try generic loading
            audio = AudioSegment.from_file(input_path)
        
        # Set up output path
        if output_dir is None:
            output_dir = tempfile.gettempdir()
        
        output_path = os.path.join(
            output_dir, 
            f"{Path(input_path).stem}_converted.wav"
        )
        
        # Export as WAV with Whisper-friendly settings
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format='wav')
        
        logger.info(f"Converted {input_path} to {output_path}")
        return output_path
        
    except Exception as e:
        raise ValueError(f"Failed to convert audio: {str(e)}")


def preprocess_audio(file_path: str, output_dir: Optional[str] = None) -> str:
    """
    Preprocess audio file for optimal transcription.
    
    Steps:
    1. Convert to WAV if needed
    2. Normalize audio levels
    3. Ensure mono channel
    4. Resample to 16kHz (Whisper's expected sample rate)
    
    Args:
        file_path: Path to the audio file
        output_dir: Directory for processed output
    
    Returns:
        Path to the preprocessed audio file
    """
    try:
        # First convert to WAV if needed
        wav_path = convert_to_wav(file_path, output_dir)
        
        # Load with librosa for processing
        audio, sr = librosa.load(wav_path, sr=16000, mono=True)
        
        # Normalize audio (prevent clipping while maximizing volume)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        # Set up output path
        if output_dir is None:
            output_dir = tempfile.gettempdir()
        
        output_path = os.path.join(
            output_dir,
            f"{Path(file_path).stem}_processed.wav"
        )
        
        # Save processed audio
        sf.write(output_path, audio, sr)
        
        logger.info(f"Preprocessed audio saved to {output_path}")
        return output_path
        
    except Exception as e:
        raise ValueError(f"Failed to preprocess audio: {str(e)}")


# =============================================================================
# Vocal Isolation (Optional - requires spleeter)
# =============================================================================

def check_spleeter_available() -> bool:
    """Check if spleeter is installed and available."""
    try:
        from spleeter.separator import Separator
        return True
    except ImportError:
        return False


def isolate_vocals(
    file_path: str, 
    output_dir: Optional[str] = None,
    keep_background: bool = False
) -> str:
    """
    Isolate vocals from background music using Spleeter.
    
    This is useful for songs with heavy instrumentation that might
    interfere with speech recognition.
    
    Args:
        file_path: Path to the audio file
        output_dir: Directory for the isolated vocals
        keep_background: If True, also save the accompaniment
    
    Returns:
        Path to the vocals-only audio file
    
    Raises:
        ImportError: If spleeter is not installed
        ValueError: If separation fails
    """
    if not check_spleeter_available():
        raise ImportError(
            "Spleeter is not installed. Install it with: "
            "pip install spleeter"
        )
    
    try:
        from spleeter.separator import Separator
        
        # Set up output directory
        if output_dir is None:
            output_dir = tempfile.gettempdir()
        
        # Initialize the separator (2stems = vocals + accompaniment)
        separator = Separator('spleeter:2stems')
        
        # Perform separation
        separator.separate_to_file(
            file_path, 
            output_dir,
            codec='wav',
            filename_format='{instrument}.{codec}'
        )
        
        # Get path to vocals file
        vocals_path = os.path.join(
            output_dir,
            Path(file_path).stem,
            'vocals.wav'
        )
        
        if not os.path.exists(vocals_path):
            raise ValueError("Vocal separation completed but output file not found")
        
        logger.info(f"Vocals isolated and saved to {vocals_path}")
        return vocals_path
        
    except ImportError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to isolate vocals: {str(e)}")


def get_audio_duration(file_path: str) -> float:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        file_path: Path to the audio file
    
    Returns:
        Duration in seconds
    """
    try:
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        logger.warning(f"Could not get duration: {str(e)}")
        return 0.0


def validate_audio_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate that an audio file is usable.
    
    Args:
        file_path: Path to the audio file
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file exists
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    # Check file size (warn if too large)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > 500:
        return False, f"File is too large ({file_size_mb:.1f}MB). Maximum recommended size is 500MB"
    
    # Try to load a small portion
    try:
        audio, sr = librosa.load(file_path, sr=None, duration=5)
        if len(audio) == 0:
            return False, "Audio file appears to be empty"
        return True, "Audio file is valid"
    except Exception as e:
        return False, f"Cannot read audio file: {str(e)}"
