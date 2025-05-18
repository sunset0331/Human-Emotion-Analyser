
# Multimodal Emotion Recognition System

A comprehensive multimodal emotion recognition system that predicts human emotions from various input types — **images, audio, video, and text** — using state-of-the-art deep learning models and preprocessing pipelines.

##  Overview

This system leverages pre-trained models and modular components to:
- Detect facial expressions from images and videos.
- Transcribe and classify speech-based emotions from audio.
- Extract and analyze text-based emotions.
- Handle preprocessing and prediction for all supported media types.

## Features

-  **Image Emotion Recognition** using ViT (Vision Transformer)
-  **Video Emotion Recognition** with frame extraction + MTCNN face detection
-  **Audio Emotion Recognition** via speech-to-text + DistilRoBERTa model
-  **Audio Extraction from Video** using `moviepy`
- **Text Emotion Recognition** with custom NLP pipeline
- **Modular Preprocessing Utilities** for all media types
-  **Visualizations** of emotion probabilities (for video input)

##  Modules

| Module | Description |
|--------|-------------|
| `predict_emotion_from_image()` | Emotion prediction from image using ViT |
| `predict_emotion_from_video()` | Frame-wise video analysis with face detection |
| `predict_emotion_from_audio()` | Speech transcription + text-based classification |
| `extract_audio_from_video()` | Extracts audio from video files |
| `preprocess_image/audio/video/text()` | Cleans and formats input for model compatibility |
| `main_predictions()` | Unified interface to route and predict based on input type |

##  Dependencies

- `transformers`, `torch`, `opencv-python`, `moviepy`, `librosa`, `matplotlib`, `speechrecognition`, `nltk`

Install all dependencies:
```bash
pip install -r requirements.txt
```

##  Example Usage

```python
# Predict emotion from video
main_predictions("sample_video.mp4")

# Predict emotion from image
main_predictions("happy_face.jpg")

# Predict emotion from audio
main_predictions("speech.wav")

# Predict emotion from raw text
main_predictions("I'm feeling really excited about the results!")
```



##  Known Issues & Fixes

| Area | Common Errors | Fixes |
|------|---------------|-------|
| Facial recognition | Size mismatch, unreadable images | Resize to model input size, add error handling |
| Audio processing | MFCC issues, unsupported formats | Convert to WAV, standardize sampling |
| Speech-to-text | Transcription errors | Handle exceptions, validate text |
| Video | Missing audio, broken files | Pre-validate video, use try-except |
| Integration | File type mismatches, tensor device conflicts | Centralized validation, ensure device consistency |

##  Future Improvements

- Real-time inference with lower latency
- Fusion model to combine multimodal predictions
- Enhanced error handling for edge cases

##  Conclusion

This project provides a unified solution to analyze emotions across multiple input formats using pre-trained AI models. Suitable for applications in sentiment analysis, interactive systems, and HCI.

---

**License:** MIT  

