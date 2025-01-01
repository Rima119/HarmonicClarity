# Harmonic Clarity: Audio Source Separation Techniques on Music

## Project Overview
This repository contains the final project for the Machine Learning course, titled **“Harmonic Clarity: Audio Source Separation Techniques on Music.”**  
Our goal is to address the challenges faced by individuals with hearing loss when perceiving music. By separating different audio sources within a music piece, we aim to create personalized remixes tailored to users' specific gain preferences, enhancing their listening experience.

## Problem Statement
Hearing loss significantly alters how music is perceived, often resulting in:
- Quiet passages becoming inaudible.
- Instruments becoming indistinguishable.
- Lyrics becoming difficult to comprehend.
- Pitches becoming distorted.  

Traditional hearing aids struggle to address these complexities in musical compositions, making personalized solutions necessary. Our project aims to bridge this gap by using audio source separation and remixing techniques.

---

## Our Approach
We propose a pipeline consisting of:
1. **Audio Source Separation**: Isolating different elements of a music piece (e.g., vocals, bass, drums).
2. **Rebalancing and Remixing**: Adjusting isolated components to match personalized hearing preferences.

This task is particularly challenging for music due to:
- Higher sample rates.
- Complex layering.
- Diverse sound dynamics.

---

## Architecture
The project focuses on the **Music Enhancement** portion of the pipeline. We implement and evaluate three distinct model architectures for separating instruments and vocals:

### 1. Hybrid Demucs  
**Location**: `clarity` folder  
- A hybrid time-frequency encoder-decoder network designed for high-quality audio separation.  

### 2. ConvTasNet  
**Location**: `ConvTasNet` folder  
- A time-domain encoder-decoder model utilizing temporal convolutional masks for separation.

### 3. BandSplitRNN  
**Location**: `bsrnn` folder  
- Recurrent neural network-based architecture for processing frequency bands independently.

---

## Data
Our models are trained and evaluated using the following datasets:
- **MUSDB18-HQ**: 150 songs with various genres and complexities.
- **EnsembleSet**: A collection of 80 ensemble pieces.
- **CadenzaWoodwind**: A curated dataset of 38 woodwind instrument pieces.

---

## Evaluation Metrics
We assess the performance of our models using the following metrics:
1. **uSDR (Unified Scale-Invariant Signal-to-Distortion Ratio)**:  
   Evaluates the separation quality of audio sources. Our BandSplitRNN results are benchmarked against the MDX 2021 baseline.
2. **HAAQI (Hearing-Aid Audio Quality Inventory)**:  
   Measures the perceived audio quality post-enhancement.

---

## Repository Structure
- `clarity/`: Code and configurations for the Hybrid Demucs model.
- `ConvTasNet/`: Code for the ConvTasNet model.
- `bsrnn/`: Implementation of the BandSplitRNN model.

---

## How to Use
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/harmonic-clarity.git


## Acknowledgments
We extend our heartfelt gratitude to the instructors and teaching assistants of the Machine Learning course for their guidance and support throughout this project. The concepts and techniques learned during the course provided the foundation for this work, and their constructive feedback played a pivotal role in shaping the final outcome.  
