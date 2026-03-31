# 📘 Project Guide — Human Emotion Analyser

A step-by-step guide for setting up, understanding, and using the **Multimodal Human Emotion Recognition System**.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Installation](#4-installation)
5. [Running the Notebook](#5-running-the-notebook)
6. [How to Use — Step-by-Step](#6-how-to-use--step-by-step)
   - [Analyze an Image](#61-analyze-an-image)
   - [Analyze a Video](#62-analyze-a-video)
   - [Analyze Audio](#63-analyze-audio)
   - [Analyze Text](#64-analyze-text)
   - [Preprocessing Only](#65-preprocessing-only)
7. [Supported Formats](#7-supported-formats)
8. [Emotion Classes](#8-emotion-classes)
9. [Pre-trained Models Used](#9-pre-trained-models-used)
10. [Project Structure](#10-project-structure)
11. [Troubleshooting](#11-troubleshooting)
12. [Future Improvements](#12-future-improvements)
13. [License](#13-license)

---

## 1. Project Overview

The **Human Emotion Analyser** is a multimodal AI system that recognises human emotions from four types of input:

| Input Type | Method |
|------------|--------|
| 🖼️ Image   | Facial expression analysis using a Vision Transformer (ViT) |
| 🎬 Video   | Frame-by-frame face detection + ViT classification |
| 🎵 Audio   | Speech-to-text transcription + NLP emotion classification |
| 📝 Text    | Direct NLP classification using DistilRoBERTa |

All input types are routed through a single unified interface (`main_predictions()`), making the system easy to use regardless of the input format.

---

## 2. System Architecture

```
Input (Image / Video / Audio / Text)
           │
           ▼
   ┌───────────────┐
   │  main_predictions()  │   ← Unified Entry Point
   └───────────────┘
           │
    ┌──────┴──────────────────┐
    │                         │                         │                        │
    ▼                         ▼                         ▼                        ▼
Image Pipeline          Video Pipeline           Audio Pipeline           Text Pipeline
──────────────          ──────────────           ──────────────           ─────────────
PIL load image       OpenCV frame              pydub / librosa           NLTK clean text
    │                  extraction                   load audio                 │
    ▼                      │                         │                         ▼
preprocess_image()     MTCNN face              preprocess_audio()        preprocess_text()
(resize 224×224,        detection                 (MFCC, mel                 (tokenise,
 normalise)               │                        spectrogram)            remove stopwords)
    │                      ▼                         │                         │
    ▼              preprocess_image()         Google Speech API               ▼
ViT Model                  │                  (speech-to-text)          DistilRoBERTa
(trpakov/               ViT Model                   │                    NLP Model
vit-face-                  │                        ▼               (j-hartmann/emotion-
expression)                │              preprocess_text()          english-distilroberta
    │                      │                         │                   -base)
    ▼                      ▼                         ▼                        │
Emotion Label +      Aggregated              Emotion Label +                  ▼
Probabilities        Probabilities           Confidence Score          Emotion Label +
                     + Bar Chart                                       Confidence Score
```

---

## 3. Prerequisites

- **Python** 3.8 or later
- **pip** (Python package manager)
- **Jupyter Notebook** or **JupyterLab**
- An internet connection (required to download models from HuggingFace and to use Google Speech Recognition)
- *(Optional)* A CUDA-compatible GPU for faster inference — the system will automatically fall back to CPU if no GPU is available

---

## 4. Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/sunset0331/Human-Emotion-Analyser.git
cd Human-Emotion-Analyser
```

### Step 2 — (Recommended) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### Step 3 — Install all dependencies

```bash
pip install transformers torch torchvision torchaudio \
            opencv-python matplotlib pillow \
            pydub openai-whisper mtcnn \
            huggingface-hub lz4 librosa \
            speechrecognition moviepy numpy nltk
```

> **Note for Google Colab users:** The notebook already includes `!pip install` commands in the early cells. Simply run those cells first.

### Step 4 — Download NLTK data

Run the following once inside Python (or in the notebook):

```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt_tab")
```

### Step 5 — (Optional) Verify GPU availability

```python
import torch
print(torch.cuda.is_available())   # True = GPU detected; False = CPU only
```

---

## 5. Running the Notebook

Launch Jupyter and open the main notebook:

```bash
jupyter notebook Multimodal_Sentiment_Analysis.ipynb
```

**Run the cells in order from top to bottom.** The notebook is divided into logical sections:

| Section | Description |
|---------|-------------|
| Cells 1–7 | Package installation and imports |
| Cell 8 | Image emotion prediction function |
| Cell 10 | Video emotion prediction function |
| Cells 13–14 | Audio emotion prediction + audio extraction from video |
| Cells 17–18 | Speech-to-text + text emotion prediction |
| Cell 22 | Preprocessing utilities (image, audio, video, text) |
| Cell 24 | `main_preprocess()` unified preprocessing interface |
| Cell 25 | `main_predictions()` unified prediction interface |
| Cells 26–32 | Test examples with sample files |

---

## 6. How to Use — Step-by-Step

All predictions use the single `main_predictions()` function. The system automatically detects the input type from the file extension (or absence of a file path for text input).

### 6.1 Analyze an Image

```python
main_predictions("path/to/image.jpg")
```

**What happens:**
1. The image is loaded with PIL and resized to 224×224 pixels.
2. The ViT face-expression model classifies the facial emotion.
3. The predicted emotion label and confidence probabilities for all 7 emotion classes are printed.

**Example output:**
```
🎭 Predicted Emotion: Happy
📊 Emotion Probabilities:
   Angry    :  2.1%
   Disgust  :  0.8%
   Fear     :  1.3%
   Happy    : 91.5%
   Neutral  :  2.4%
   Sad      :  1.2%
   Surprise :  0.7%
```

---

### 6.2 Analyze a Video

```python
main_predictions("path/to/video.mp4")
```

**What happens:**
1. OpenCV extracts frames from the video at every 30th frame (configurable via `frame_interval`).
2. MTCNN detects faces in each extracted frame.
3. Each detected face is classified by the ViT model.
4. Emotion predictions are aggregated across all frames.
5. If the video has audio, the audio is extracted, transcribed, and the text emotion is also predicted.
6. A bar chart of aggregated emotion probabilities is displayed.

**To change the frame sampling rate:**
```python
predict_emotion_from_video("video.mp4", frame_interval=15)  # sample every 15th frame
```

---

### 6.3 Analyze Audio

```python
main_predictions("path/to/audio.wav")
```

**What happens:**
1. The audio file is loaded and preprocessed (MFCC + mel spectrogram extraction).
2. Google Speech Recognition transcribes the speech to text.
3. The transcribed text is classified by the DistilRoBERTa NLP model.
4. The predicted emotion label and confidence score are printed.

> **Tip:** For best accuracy, use clear speech audio with minimal background noise.

---

### 6.4 Analyze Text

```python
main_predictions(text_input="I am feeling really excited about this!")
```

**What happens:**
1. The text is tokenized and cleaned using NLTK (stopword removal, lemmatization).
2. The DistilRoBERTa model classifies the emotion directly from the text.
3. The predicted emotion label and confidence score are printed.

**Example output:**
```
📝 Text: "I am feeling really excited about this!"
🎭 Predicted Emotion: Joy
✅ Confidence: 94.3%
```

---

### 6.5 Preprocessing Only

Use `main_preprocess()` to run preprocessing without making a prediction:

```python
# Preprocess a video
main_preprocess("/path/to/video.mp4")

# Preprocess an audio file
main_preprocess("/path/to/audio.wav")

# Preprocess an image
main_preprocess("/path/to/image.jpg")

# Preprocess text
main_preprocess(text_input="sample text to preprocess")
```

This is useful for inspecting extracted features (MFCCs, mel spectrograms, normalised frames) before running prediction.

---

## 7. Supported Formats

| Media Type | Supported Extensions |
|------------|----------------------|
| Image      | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff` |
| Video      | `.mp4`, `.avi`, `.mov`, `.mkv` |
| Audio      | `.mp3`, `.wav`, `.flac` |
| Text       | Pass a string directly to `text_input=` parameter |

---

## 8. Emotion Classes

The system classifies input into one of **7 emotion categories**:

| # | Emotion  | Description |
|---|----------|-------------|
| 1 | Angry    | Expressions of anger or frustration |
| 2 | Disgust  | Expressions of disgust or revulsion |
| 3 | Fear     | Expressions of fear or anxiety |
| 4 | Happy    | Expressions of happiness or joy |
| 5 | Neutral  | No strong emotion expressed |
| 6 | Sad      | Expressions of sadness or grief |
| 7 | Surprise | Expressions of surprise or shock |

---

## 9. Pre-trained Models Used

| Model | Source | Used For |
|-------|--------|----------|
| `trpakov/vit-face-expression` | HuggingFace | Image and video facial emotion recognition |
| `j-hartmann/emotion-english-distilroberta-base` | HuggingFace | Text and audio (via transcript) emotion classification |
| MTCNN | `mtcnn` library | Real-time face detection in images and video frames |

Models are downloaded automatically from HuggingFace on first run and cached locally.

---

## 10. Project Structure

```
Human-Emotion-Analyser/
│
├── Multimodal_Sentiment_Analysis.ipynb   # Main project notebook (all code)
├── PROJECT_GUIDE.md                       # This guide
├── README.md                              # Short project description
└── README_Multimodal_Emotion_Recognition.md  # Technical reference documentation
```

---

## 11. Troubleshooting

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| `ModuleNotFoundError` for any package | Dependency not installed | Run the `pip install` commands in [Installation](#4-installation) |
| `Could not understand audio` | Low audio quality or background noise | Use cleaner audio; try a different file |
| `No audio stream found in video` | Video has no audio track | The video emotion prediction still works; audio step is skipped |
| `No face detected` in a video frame | Face not visible or too small | Reduce `frame_interval` to sample more frames; ensure face is visible |
| `CUDA out of memory` | GPU VRAM insufficient | Reduce batch size or switch to CPU: `device = torch.device("cpu")` |
| Slow inference | Running on CPU | Use a GPU-enabled environment (e.g., Google Colab with T4 GPU) |
| `OSError` when loading model | First-run model download failed | Check internet connection and retry |
| `ValueError: Unsupported file type` | File extension not recognised | Use a supported format listed in [Supported Formats](#7-supported-formats) |
| NLTK `LookupError` | NLTK data not downloaded | Run the NLTK download commands in [Step 4](#step-4--download-nltk-data) |

---

## 12. Future Improvements

- **Real-time inference** — Support live webcam and microphone streams
- **Multimodal fusion** — Combine predictions from image, audio, and text into a single weighted score
- **Multiple face tracking** — Detect and classify emotions for multiple faces in a single frame
- **Offline speech recognition** — Replace Google Speech API with fully local inference (e.g., Whisper)
- **Model quantisation** — Reduce model size for faster inference on low-resource devices
- **Web interface** — A simple browser-based UI for uploading files and viewing results

---

## 13. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

*For technical API details and known issues, see [README_Multimodal_Emotion_Recognition.md](README_Multimodal_Emotion_Recognition.md).*
