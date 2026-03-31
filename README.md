# Human Emotion Analyser

A comprehensive multimodal emotion recognition system that predicts human emotions from various input types — **images, audio, video, and text** — using state-of-the-art deep learning models and preprocessing pipelines.

## Quick Start

```bash
# Install dependencies
pip install transformers torch torchvision opencv-python librosa speechrecognition moviepy mtcnn

# Open the notebook
jupyter notebook Multimodal_Sentiment_Analysis.ipynb
```

## Usage

```python
# Predict emotion from any input type
main_predictions("photo.jpg")         # image
main_predictions("clip.mp4")          # video
main_predictions("speech.wav")        # audio
main_predictions(text_input="I'm so happy today!")  # text
```

## Documentation

- 📘 **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** — Full step-by-step setup and usage guide
- 📄 **[README_Multimodal_Emotion_Recognition.md](README_Multimodal_Emotion_Recognition.md)** — Technical reference

## License

MIT
