
**Your repo currently contains:** 

* `LAST WORK.py`
* `LAST33.py`
* `Last-Work-Streamlit.py` (Streamlit app)
* `MY WORK MAN.pdf` (report)
* `Chapter4_Implementation_Results.docx`
* `speaker-diarize.zip`


---

# Speaker Diarization System (CNN-LSTM)

A deep learning–based **speaker diarization system** built as a **final year project**.

The system processes audio recordings and automatically **segments speech by speaker**, identifying when each speaker talks in the audio.

The project uses a **hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM)** architecture trained on **MFCC (Mel Frequency Cepstral Coefficients)** features.

A **Streamlit web interface** is provided to visualize diarization results and interact with the model.

---

# Project Overview

Speaker diarization answers the question:

> **"Who spoke when?"**

This project analyzes an audio file (`street-interview.wav`) and performs:

* Speaker segmentation
* Speaker clustering
* Speech detection
* Visualization of speaker timelines

The model was evaluated on a **66.9 second audio sample** recorded at **16kHz mono**.

---

# Model Architecture

The diarization model combines convolutional and sequential neural networks.

### Architecture

```
MFCC Features
      ↓
Conv1D Layer (64 filters)
      ↓
Batch Normalization
      ↓
MaxPooling
      ↓
LSTM Layer (128 units)
      ↓
Dropout (0.3)
      ↓
Dense Layers
      ↓
Softmax Output
```

The CNN extracts **local speech features**, while the LSTM captures **temporal speech patterns**.

---

# Feature Extraction

Audio is converted into **MFCC features**, which are commonly used in speech recognition.

Configuration:

| Parameter         | Value  |
| ----------------- | ------ |
| Sampling rate     | 16 kHz |
| Frame length      | 512    |
| Hop length        | 256    |
| MFCC coefficients | 13     |
| Total frames      | 4182   |

---

# Training Performance

| Epoch | Training Accuracy | Validation Accuracy |
| ----- | ----------------- | ------------------- |
| 1     | 52.84%            | 59.06%              |
| 2     | 66.58%            | 63.99%              |
| 3     | 74.20%            | 70.47%              |
| 4     | 83.94%            | 80.07%              |
| 5     | 86.52%            | 84.39%              |
| 6     | 90.67%            | 85.59%              |
| 7     | 91.95%            | 83.43%              |
| 8     | 94.44%            | 88.24%              |
| 9     | 95.02%            | 90.04%              |
| 10    | **96.15%**        | **95.80%**          |

The model demonstrates strong generalization with minimal overfitting.

---

# Evaluation Metrics

Two main evaluation metrics were used.

### Silhouette Score

Measures clustering quality.

| Model            | Score |
| ---------------- | ----- |
| CNN-LSTM         | ~0.62 |
| K-Means Baseline | ~0.51 |

Higher scores indicate better speaker separation.

---

### Diarization Error Rate (DER)

Measures the percentage of incorrectly classified frames.

Example result:

```
DER ≈ 4.32%
```

Most errors occur in **overlapping speech regions**.

---

# Visualizations

The Streamlit application generates several plots:

### Waveform with Speaker Labels

Shows where each speaker appears in the audio.

### Speech Detection

Voice Activity Detection highlights speech segments.

### Diarization Timeline

Displays speaker changes and overlap regions.

---

# Project Files

```
speaker-diarize-latest/
│
├── Last-Work-Streamlit.py   # Main Streamlit application
├── LAST WORK.py             # Model training / diarization logic
├── LAST33.py                # Alternative implementation
│
├── MY WORK MAN.pdf          # Full project report
├── Chapter4_Implementation_Results.docx
│
└── speaker-diarize.zip      # Archived project files
```

---

# Running the Project

### 1 Install Dependencies

```bash
pip install streamlit librosa numpy matplotlib scikit-learn tensorflow
```

---

### 2 Run the Streamlit App

```bash
streamlit run Last-Work-Streamlit.py
```

---

### 3 Open the Web Interface

Streamlit will generate a local link:

```
http://localhost:8501
```

Open it in your browser to interact with the diarization system.

---

# Key Features

* CNN-LSTM deep learning architecture
* MFCC speech feature extraction
* Automatic speaker estimation
* Manual labeling interface
* Diarization visualizations
* Speech detection (VAD)
* Overlap **detection**

---

# Limitations

Current limitations include:

* Difficulty handling overlapping speakers
* Limited labeled training data
* Single-channel audio processing

Future work could include:

* Transformer-based diarization models
* Larger speech datasets
* Real-time diarization
* Multi-channel audio processing

---

# Author

**Olateju Olamide Emmanuel**

Final Year Computer Science Project

---