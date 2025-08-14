# Image Captioning

A simple and task-oriented image captioning project built with Python and deep learning.

## Overview

This project implements an image captioning system using an encoder-decoder architecture. 
This task is applicable to Intelligent On-Site Proctoring.
The encoder extracts features from images, and the decoder generates descriptive captions based on those features.

## Features

- **Encoder**: CNN-based feature extractor (e.g., ResNet, Inception)
- **Decoder**: LSTM-based caption generator with optional attention
- **Inference**: Generate captions using `caption_generate.py`
- **GUI**: Minimal interface via `main_ui3.py` for loading images and viewing captions

## Getting Started

### Prerequisites

- Python 3.8.0
- Required packages (see `requirements.txt`)

### Installation

```bash
git clone https://github.com/kennyliuu/image_captioning.git
cd image_captioning
pip install -r requirements.txt
