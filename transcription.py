"""
Transcription Module
====================
Handles Whisper model loading and audio transcription
for the lyrics generator app.
"""

import os
import logging
from typing import Dict, List, Optional, Any
import warnings

# Suppress warnings during model loading
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import whisper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Configuration
# =============================================================================

# Available Whisper models and their characteristics
MODEL_INFO = {
    'base': {
        'name': 'base',
        'size': '~140MB',
        'speed': 'Fast',
        'accuracy': 'Good for clear audio',
        'vram': '~1GB'
    },
    'small': {
        'name': 'small',
        'size': '~460MB',
        'speed': 'Moderate',
        'accuracy': 'Better accuracy',
        'vram': '~2GB'
    },
    'medium': {
        'name': 'medium',
        'size': '~1.5GB',
        'speed': 'Slower',
        'accuracy': 'High accuracy',
        'vram': '~5GB'
    }
}


def get_available_models() -> List[str]:
    """Return list of available model names."""
    return list(MODEL_INFO.keys())


def get_model_info(model_name: str) -> Dict[str, str]:
    """Get information about a specific model."""
    return MODEL_INFO.get(model_name, MODEL_INFO['base'])


# =============================================================================
# Device Detection
# =============================================================================

def get_device() -> str:
    """
    Detect the best available device for running Whisper.
    
    Returns:
        'cuda' if GPU is available, otherwise 'cpu'
    """
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
    else:
        device = 'cpu'
        logger.info("Using CPU (GPU not available)")
    
    return device


def check_memory_available(model_name: str) -> bool:
    """
    Check if there's enough memory to load the specified model.
    
    Args:
        model_name: Name of the Whisper model
    
    Returns:
        True if memory seems sufficient
    """
    if torch.cuda.is_available():
        # Check GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        required_memory = {
            'base': 1,
            'small': 2,
            'medium': 5
        }
        required = required_memory.get(model_name, 1)
        
        if gpu_memory_gb < required:
            logger.warning(
                f"GPU memory ({gpu_memory_gb:.1f}GB) may be insufficient "
                f"for {model_name} model (recommended: {required}GB)"
            )
            return False
    
    return True


# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_name: str = 'base', device: Optional[str] = None) -> whisper.Whisper:
    """
    Load a Whisper model.
    
    Args:
        model_name: Size of the model ('base', 'small', 'medium')
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        Loaded Whisper model
    
    Raises:
        ValueError: If model name is invalid
        MemoryError: If not enough memory to load model
    """
    # Validate model name
    if model_name not in MODEL_INFO:
        raise ValueError(
            f"Invalid model: {model_name}. "
            f"Choose from: {list(MODEL_INFO.keys())}"
        )
    
    # Determine device
    if device is None:
        device = get_device()
    
    # Check memory
    if not check_memory_available(model_name):
        logger.warning("Proceeding despite memory concerns...")
    
    try:
        logger.info(f"Loading Whisper {model_name} model on {device}...")
        model = whisper.load_model(model_name, device=device)
        logger.info("Model loaded successfully!")
        return model
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise MemoryError(
                f"Not enough memory to load {model_name} model. "
                "Try a smaller model or close other applications."
            )
        raise
    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}")


# =============================================================================
# Transcription
# =============================================================================

def transcribe(
    audio_path: str,
    model: whisper.Whisper,
    language: Optional[str] = None,
    task: str = 'transcribe',
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Transcribe an audio file using Whisper.
    
    Args:
        audio_path: Path to the audio file
        model: Loaded Whisper model
        language: Language code (e.g., 'en', 'id') or None for auto-detect
        task: 'transcribe' or 'translate' (translate to English)
        verbose: If True, print progress
    
    Returns:
        Dictionary containing:
        - 'text': Full transcription text
        - 'segments': List of segments with timestamps
        - 'language': Detected language
    
    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If transcription fails
    """
    # Validate file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        logger.info(f"Starting transcription of {audio_path}...")
        
        # Set up transcription options
        options = {
            'task': task,
            'verbose': verbose,
            'fp16': torch.cuda.is_available()  # Use FP16 on GPU for speed
        }
        
        if language:
            options['language'] = language
        
        # Perform transcription
        result = model.transcribe(audio_path, **options)
        
        logger.info(
            f"Transcription complete! "
            f"Language: {result.get('language', 'unknown')}, "
            f"Segments: {len(result.get('segments', []))}"
        )
        
        return result
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise MemoryError(
                "Ran out of memory during transcription. "
                "Try a smaller model or shorter audio file."
            )
        raise ValueError(f"Transcription failed: {str(e)}")
    except Exception as e:
        raise ValueError(f"Transcription failed: {str(e)}")


def transcribe_with_progress(
    audio_path: str,
    model: whisper.Whisper,
    progress_callback: Optional[callable] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Transcribe with progress updates.
    
    Note: Whisper doesn't provide built-in progress callbacks,
    so this is a wrapper that provides pre/post callbacks.
    
    Args:
        audio_path: Path to the audio file
        model: Loaded Whisper model
        progress_callback: Function to call with progress updates
        **kwargs: Additional arguments passed to transcribe()
    
    Returns:
        Transcription result dictionary
    """
    if progress_callback:
        progress_callback(0.1, "Starting transcription...")
    
    try:
        if progress_callback:
            progress_callback(0.3, "Processing audio...")
        
        result = transcribe(audio_path, model, **kwargs)
        
        if progress_callback:
            progress_callback(0.9, "Finalizing...")
        
        return result
        
    except Exception as e:
        if progress_callback:
            progress_callback(1.0, f"Error: {str(e)}")
        raise


def get_transcription_segments(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract segments from transcription result.
    
    Args:
        result: Transcription result from Whisper
    
    Returns:
        List of segment dictionaries with 'start', 'end', 'text'
    """
    return result.get('segments', [])


def estimate_transcription_time(
    audio_duration_seconds: float,
    model_name: str,
    device: str
) -> float:
    """
    Estimate transcription time based on audio length and model.
    
    Args:
        audio_duration_seconds: Length of audio in seconds
        model_name: Whisper model name
        device: 'cuda' or 'cpu'
    
    Returns:
        Estimated time in seconds
    """
    # Rough multipliers based on empirical observations
    base_multipliers = {
        'base': 0.1,
        'small': 0.2,
        'medium': 0.5
    }
    
    multiplier = base_multipliers.get(model_name, 0.2)
    
    # CPU is significantly slower
    if device == 'cpu':
        multiplier *= 5
    
    return audio_duration_seconds * multiplier
