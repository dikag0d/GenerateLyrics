# 🎤 Lyrics Generator

A Python-based web application that automatically transcribes audio files (songs) into clean lyric transcripts using OpenAI's Whisper model.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-green.svg)

## ✨ Features

- **🎵 Multi-format Support**: Upload MP3, WAV, and M4A audio files
- **🤖 Model Selection**: Choose between Whisper base, small, or medium models
- **⏱️ Optional Timestamps**: Include timestamps in the generated lyrics
- **🎚️ Vocal Isolation**: Optionally separate vocals from background music (requires Spleeter)
- **📥 Easy Export**: Download lyrics as a clean `.txt` file
- **🎨 Beautiful UI**: Modern, responsive Streamlit interface with visual feedback

## 📋 Prerequisites

- Python 3.8 or higher
- FFmpeg (required for audio processing)

### Installing FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS (with Homebrew):**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

## 🚀 Quick Start

### 1. Clone or Navigate to the Project

```bash
cd /media/dika/TEAM/GenerateLyrics
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: The first run will download the Whisper model (~140MB - 1.5GB depending on model size).

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

## 📖 Usage

1. **Upload Audio**: Drag and drop or click to upload your audio file (MP3, WAV, or M4A)
2. **Configure Settings**: 
   - Select Whisper model size (larger = more accurate but slower)
   - Toggle timestamp inclusion
   - Enable vocal isolation if needed
3. **Generate Lyrics**: Click the "Generate Lyrics" button
4. **Edit & Export**: Review/edit the generated lyrics and download as `.txt`

## ⚙️ Configuration Options

### Whisper Models

| Model  | Size   | Speed    | Accuracy | VRAM Required |
|--------|--------|----------|----------|---------------|
| base   | ~140MB | Fast     | Good     | ~1GB          |
| small  | ~460MB | Moderate | Better   | ~2GB          |
| medium | ~1.5GB | Slower   | High     | ~5GB          |

### Output Options

- **Include Timestamps**: Adds `[MM:SS]` at the start of each line
- **Include Metadata**: Adds a header with language and generation info

## 🎚️ Vocal Isolation (Optional)

For songs with heavy instrumentation, you can enable vocal isolation to improve transcription accuracy.

### Installing Spleeter

```bash
pip install spleeter
```

> **Warning**: Spleeter requires additional downloads (~100MB) on first use and uses significant memory.

## 📁 Project Structure

```
GenerateLyrics/
├── app.py                  # Main Streamlit application
├── audio_processing.py     # Audio loading and preprocessing
├── transcription.py        # Whisper transcription engine
├── lyrics_formatter.py     # Lyrics formatting logic
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🔧 Troubleshooting

### Common Issues

**1. "ffmpeg not found"**
- Ensure FFmpeg is installed and in your system PATH
- Restart your terminal after installation

**2. "CUDA out of memory"**
- Use a smaller Whisper model
- Close other GPU-intensive applications
- The app will automatically fall back to CPU if needed

**3. "Transcription is slow"**
- GPU acceleration is recommended for faster processing
- Smaller models (base, small) are faster but less accurate

**4. "Module not found" errors**
- Make sure your virtual environment is activated
- Re-run `pip install -r requirements.txt`

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Report bugs or issues
2. Suggest new features
3. Submit pull requests

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [Streamlit](https://streamlit.io/) - Web application framework
- [Spleeter](https://github.com/deezer/spleeter) - Audio source separation (optional)
