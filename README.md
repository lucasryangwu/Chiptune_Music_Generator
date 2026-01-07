# Chiptune Music Generator (v1.0)

---

## Overview

Pretrained LSTM model for generating polyphonic, chiptune-style music.

This model was trained on the [NES-MDB](https://github.com/chrisdonahue/nesmdb/tree/master) dataset,
and makes use of the Tonnetz structure to represent music data geometrically as originally explored
by Ching-Hua Chuan and Dorien Herremans in their [2018 paper](https://ojs.aaai.org/index.php/AAAI/article/view/11880).

The system utilizes a Convolutional Neural Network to encode the Tonnetz-translated MIDI data into a latent space.
These embeddings are then fed into a LSTM network to generate polyphonic music using a provided seed.

Future revisions of this model will include a user-friendly interface and a more robust training pipeline.

### Basic Information
- **Developed by:** Lucas Ryan ([LinkedIn](https://www.linkedin.com/in/lucasryan360/))
- **Model Version:** 1.0
- **Version Date:** December 2025
- **Model Type:** Hybrid CNN-Autoencoder + LSTM Sequence Predictor
- **Language(s):** Python 3.13
- **License:** MIT

### Intended Use
- **Primary intended uses:** To investigate the potential of using deep learning to generate polyphonic music.
- **Out-of-scope use cases:** This model was developed for educational purposes only. Any use beyond that is out-of-scope.

---

## Training Data
- **Dataset:** NESMDB (NES Music Database).
  - NOTE: this dataset is not provided with this repository. Please refer to the source for more information.
- **Source:** [NES-MDB](https://github.com/chrisdonahue/nesmdb/tree/master)
- **Dataset size:** 5,278 songs
  - 296 unique composers
  - Over two million notes
- **Dataset format:** individual `.mid` files
- **Partition information**
  - Training: 4,160 songs 
  - Validation: 373 songs 
  - Test: 331 songs

### Preprocessing
- Minimum track duration: 2.5 seconds.
- Minimum note duration: 0.0625 seconds (32nd note at 120 BPM).
- Removal of auxiliary tracks (Percussion, Breath Noise).
- Normalization of MIDI velocities and control changes.

### Factors
- **Quantization:** Operates on a beat-quantized grid (default 1.0 beat per slice).
- **Geometric Encoding:** Uses the Tonnetz lattice (24x12 grid) to represent tonal relationships between notes.

### Metrics
- **Training Loss:** Binary Cross-Entropy with Logits (BCEWithLogitsLoss) for sequence prediction.
- **Pre-training Loss:** L1 Loss used for the Convolutional Autoencoder.
- **Validation:** Early stopping based on validation loss improvement (patience of 5 epochs).

---

## Model Details
- **Autoencoder Architecture:** 
    - 2-layer Convolutional Encoder (feat_maps: 20, 10).
    - Latent dimension: 128.
- **Sequence Predictor Architecture:**
    - 2-layer LSTM.
    - Hidden dimension: 256.
    - Input: Sequence of 16 latent vectors.

---

## Ethical Considerations
- **Data Source:** The model is trained on copyrighted music from classic video games. Users should be aware of potential copyright implications if using generated music for commercial purposes.
- **Bias:** The model is biased towards the specific harmonic and melodic structures found in 1980s-era Japanese and Western video game music.

## Caveats and Recommendations
- **Seed Sensitivity:** The quality of the generated music is highly dependent on the quality and length of the "seed" MIDI sequence provided.
- **Quantization Limits:** Because the model quantizes to the beat, rapid trills or complex rhythmic syncopations may be lost in the encoding process.
- **Future Work:** Integration of a GUI for real-time generation and playback will be implemented for better user experience.

---