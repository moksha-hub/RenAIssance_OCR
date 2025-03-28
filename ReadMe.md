# Hybrid Transformer-Based OCR for 17th-Century Spanish Texts

This project implements an OCR pipeline tailored for centuries-old printed and handwritten Spanish texts. It leverages hybrid end-to-end transformer models (e.g., VIT-RNN, CNN-TF, or VIT-TF) and incorporates advanced augmentation, preprocessing, and fine-tuning techniques (using AdaLoRA and EMA) to achieve robust text recognition on degraded historical documents.

---

## Overview

Historical documents pose unique challenges due to non-standard typography, degraded prints, and handwritten scripts. Traditional OCR tools (such as Adobe Acrobat's OCR) often fail with these documents. Our solution focuses on:
- **Text Recognition:** Training transformer-based OCR models (using architectures such as TrOCR) on 17th-century Spanish texts.
- **Augmentation & Preprocessing:** Applying methods like deskewing, denoising, and diverse image augmentations.
- **Fine-Tuning Techniques:** Employing lightweight fine-tuning on the decoder with AdaLoRA and improving generalization through an Exponential Moving Average (EMA) callback.
- **Annotation & Evaluation:** Using metrics like Character Error Rate (CER) and Word Error Rate (WER), with a custom text normalization function to handle historical character irregularities.
- Interchangeable Characters: Characters like 'u' & 'v', and 'f' & 's' were used interchangeably. Assume 'u' at the beginning of word and 'v' inside word. Assume 's' at the beginning/end of a word, 'f' within a word.


---

## Project Structure

1. **Mount Google Drive (Optional):**  
   For Colab users who wish to access datasets or save checkpoints from Google Drive.
2. **Install Dependencies:**  
   System utilities (e.g., `poppler-utils`) and Python packages (e.g., `paddleocr`, `transformers`, `peft`, `albumentations`, `jiwer`) are installed.
3. **Imports and Utility Functions:**  
   Contains preprocessing functions, custom CER computation using Levenshtein distance, and text normalization routines.
4. **Normalization Function:**  
   The `normalize_text` function is explicitly defined to replace historical characters (e.g., 'ç' and 'ſ') and remove diacritical marks, ensuring text consistency.
5. **Line-Level Dataset with Augmentation:**  
   A dataset class that loads image-text pairs and applies augmentations using Albumentations.  
   **Note:** All class methods use standard dunder names (`__init__`, `__len__`, and `__getitem__`).
6. **Dataset Gathering:**  
   A helper function to traverse the dataset directory and collect line-level image folders.
7. **EMA Callback:**  
   A custom Trainer callback to apply an Exponential Moving Average (EMA) to model parameters for better generalization.
8. **Training Function:**  
   Combines data loading, model initialization (using `qantev/trocr-base-spanish`), AdaLoRA fine-tuning, custom training loss, and metric computation (CER and WER).
9. **Main Execution:**  
   A main function to trigger the training process by setting the dataset root.

---

## Installation and Setup

### 1. Mount Google Drive (Optional)

For Google Colab users, mount your Google Drive to access datasets or save checkpoints:
```python
from google.colab import drive
drive.mount('/content/drive')

