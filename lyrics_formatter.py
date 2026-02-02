"""
Lyrics Formatter Module
======================
Formats Whisper transcription output as clean, readable song lyrics.
"""

import re
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Pause duration thresholds (in seconds)
SHORT_PAUSE = 0.5   # Brief pause - might be phrase break
MEDIUM_PAUSE = 1.0  # Likely line break
LONG_PAUSE = 2.0    # Likely verse/section break

# Common line ending patterns
LINE_ENDINGS = [',', '!', '?', '.', '...', '、', '。']


# =============================================================================
# Timestamp Formatting
# =============================================================================

def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to MM:SS.ms format.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted timestamp string
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"[{minutes:02d}:{secs:05.2f}]"


def format_timestamp_simple(seconds: float) -> str:
    """
    Convert seconds to MM:SS format (no milliseconds).
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted timestamp string
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"[{minutes:02d}:{secs:02d}]"


# =============================================================================
# Text Processing
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean transcription text for lyrics display.
    
    Args:
        text: Raw transcription text
    
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Capitalize first letter if not already
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
    return text.strip()


def should_break_line(
    text: str, 
    next_text: str,
    pause_duration: float
) -> bool:
    """
    Determine if a line break should be inserted.
    
    Args:
        text: Current segment text
        next_text: Next segment text
        pause_duration: Pause between segments in seconds
    
    Returns:
        True if a line break is recommended
    """
    # Long pause = definite break
    if pause_duration >= LONG_PAUSE:
        return True
    
    # Medium pause = likely break
    if pause_duration >= MEDIUM_PAUSE:
        return True
    
    # Check for natural ending punctuation
    if text and text[-1] in LINE_ENDINGS:
        return True
    
    # Check for short pause with punctuation
    if pause_duration >= SHORT_PAUSE and len(text) > 30:
        return True
    
    return False


def should_add_blank_line(pause_duration: float, segment_index: int) -> bool:
    """
    Determine if a blank line (verse break) should be added.
    
    Args:
        pause_duration: Pause between segments in seconds
        segment_index: Current segment index
    
    Returns:
        True if a blank line should be inserted
    """
    # Very long pause indicates section break
    if pause_duration >= LONG_PAUSE * 1.5:
        return True
    
    return False


# =============================================================================
# Lyrics Formatting
# =============================================================================

def format_as_lyrics(
    segments: List[Dict[str, Any]],
    include_timestamps: bool = False,
    max_line_length: int = 60
) -> str:
    """
    Format transcription segments as song lyrics.
    
    This function analyzes timing gaps between segments to
    intelligently break the text into lyric lines and verses.
    
    Args:
        segments: List of Whisper segments with 'start', 'end', 'text'
        include_timestamps: If True, include timestamps at line starts
        max_line_length: Maximum characters per line before forcing break
    
    Returns:
        Formatted lyrics string
    """
    if not segments:
        return ""
    
    lyrics_lines = []
    current_line = ""
    
    for i, segment in enumerate(segments):
        text = clean_text(segment.get('text', ''))
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        
        if not text:
            continue
        
        # Calculate pause from previous segment
        if i > 0:
            prev_end = segments[i - 1].get('end', 0)
            pause_duration = start - prev_end
        else:
            pause_duration = 0
        
        # Check if we need a blank line (verse break)
        if i > 0 and should_add_blank_line(pause_duration, i):
            if current_line:
                if include_timestamps:
                    line_start = segments[max(0, i-1)].get('start', 0)
                    lyrics_lines.append(
                        f"{format_timestamp_simple(line_start)} {current_line}"
                    )
                else:
                    lyrics_lines.append(current_line)
                current_line = ""
            lyrics_lines.append("")  # Blank line for verse break
        
        # Get next segment text for lookahead
        next_text = ""
        if i < len(segments) - 1:
            next_text = segments[i + 1].get('text', '')
        
        # Determine if we should break the line
        if i > 0 and should_break_line(current_line, text, pause_duration):
            if current_line:
                if include_timestamps:
                    line_start = segments[max(0, i-1)].get('start', 0)
                    lyrics_lines.append(
                        f"{format_timestamp_simple(line_start)} {current_line}"
                    )
                else:
                    lyrics_lines.append(current_line)
                current_line = ""
        
        # Add text to current line
        if current_line:
            # Check if adding would exceed max length
            if len(current_line) + len(text) + 1 > max_line_length:
                if include_timestamps:
                    lyrics_lines.append(
                        f"{format_timestamp_simple(start)} {current_line}"
                    )
                else:
                    lyrics_lines.append(current_line)
                current_line = text
            else:
                current_line = f"{current_line} {text}"
        else:
            current_line = text
    
    # Don't forget the last line
    if current_line:
        if include_timestamps and segments:
            last_start = segments[-1].get('start', 0)
            lyrics_lines.append(
                f"{format_timestamp_simple(last_start)} {current_line}"
            )
        else:
            lyrics_lines.append(current_line)
    
    return '\n'.join(lyrics_lines)


def format_with_timestamps(segments: List[Dict[str, Any]]) -> str:
    """
    Format each segment with its precise timestamp.
    
    This is a simpler format where each segment is on its own line.
    
    Args:
        segments: List of Whisper segments
    
    Returns:
        Formatted text with timestamps
    """
    lines = []
    for segment in segments:
        start = segment.get('start', 0)
        text = clean_text(segment.get('text', ''))
        if text:
            timestamp = format_timestamp(start)
            lines.append(f"{timestamp} {text}")
    
    return '\n'.join(lines)


def format_plain_text(segments: List[Dict[str, Any]]) -> str:
    """
    Format segments as plain continuous text.
    
    Args:
        segments: List of Whisper segments
    
    Returns:
        Plain text transcription
    """
    texts = []
    for segment in segments:
        text = clean_text(segment.get('text', ''))
        if text:
            texts.append(text)
    
    return ' '.join(texts)


# =============================================================================
# Export Functions
# =============================================================================

def generate_lyrics_file_content(
    transcription_result: Dict[str, Any],
    include_timestamps: bool = False,
    include_metadata: bool = True
) -> str:
    """
    Generate the full content for a lyrics file.
    
    Args:
        transcription_result: Full Whisper transcription result
        include_timestamps: If True, include timestamps
        include_metadata: If True, add a header with metadata
    
    Returns:
        Complete file content as string
    """
    segments = transcription_result.get('segments', [])
    language = transcription_result.get('language', 'unknown')
    
    content_lines = []
    
    # Add metadata header if requested
    if include_metadata:
        content_lines.append("=" * 50)
        content_lines.append("LYRICS TRANSCRIPT")
        content_lines.append(f"Language: {language.upper()}")
        content_lines.append("=" * 50)
        content_lines.append("")
    
    # Add formatted lyrics
    lyrics = format_as_lyrics(segments, include_timestamps)
    content_lines.append(lyrics)
    
    # Add footer
    if include_metadata:
        content_lines.append("")
        content_lines.append("-" * 50)
        content_lines.append("Generated with Lyrics Generator")
        content_lines.append("-" * 50)
    
    return '\n'.join(content_lines)


def estimate_song_structure(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Attempt to estimate the song structure based on timing patterns.
    
    This is experimental and provides rough estimates.
    
    Args:
        segments: List of Whisper segments
    
    Returns:
        Dictionary with structure estimates
    """
    if not segments:
        return {'verses': 0, 'total_duration': 0}
    
    # Calculate pauses between segments
    pauses = []
    for i in range(1, len(segments)):
        prev_end = segments[i-1].get('end', 0)
        curr_start = segments[i].get('start', 0)
        pauses.append(curr_start - prev_end)
    
    # Count significant breaks (potential verse boundaries)
    verse_breaks = sum(1 for p in pauses if p >= LONG_PAUSE)
    
    # Get total duration
    if segments:
        total_duration = segments[-1].get('end', 0)
    else:
        total_duration = 0
    
    return {
        'estimated_verses': verse_breaks + 1,
        'total_duration': total_duration,
        'total_segments': len(segments),
        'average_segment_length': total_duration / len(segments) if segments else 0
    }
