# DecodeLabs Project 4: Optical Character Recognition (OCR) System

## Project Overview

This project focuses on developing an Optical Character Recognition (OCR) system using Python and Tesseract OCR. The system processes images containing text, extracts readable information, evaluates recognition confidence, and visually identifies detected text using bounding boxes.

The project demonstrates how image processing techniques and OCR technology can be combined to automate text extraction from digital images.

## Project Objectives

- Understand image processing using Python.
- Convert images into grayscale.
- Resize images to improve text recognition.
- Apply adaptive thresholding to enhance text visibility.
- Extract text using Tesseract OCR.
- Evaluate OCR confidence scores against an 80% threshold.
- Detect text regions using bounding boxes.
- Measure recognition performance using Word Error Rate (WER) and Character Error Rate (CER).
- Compare different Tesseract Page Segmentation Modes.

## Technologies and Libraries

- Python
- Google Colab
- OpenCV
- Pillow (PIL)
- NumPy
- Matplotlib
- Pytesseract
- Tesseract OCR
- JiWER
- Pandas

## Methodology

The project follows these processing stages:

1. Image acquisition and inspection.
2. Grayscale conversion.
3. Image resizing using interpolation.
4. Adaptive thresholding.
5. Text extraction using Tesseract OCR.
6. Confidence score evaluation.
7. Bounding box detection and visualization.
8. OCR accuracy evaluation using WER and CER.
9. Comparison of Tesseract PSM 3, PSM 6 and PSM 11.

## Performance Evaluation

The system evaluates its performance using:

- Average OCR confidence.
- Word Error Rate (WER).
- Character Error Rate (CER).
- Word accuracy estimate.
- Character accuracy estimate.

An 80% confidence threshold was used to identify words accepted by the system. This threshold is an acceptance criterion and does not independently establish recognition accuracy.

## Project Files

- `ocr_project.ipynb`: Complete implementation and experiments.
- `OCR_Evaluation_Report.txt`: Summary of the evaluation.
- `OCR_Performance_Comparison.csv`: Comparison of OCR configurations.
- `OCR_Detected_Text.png`: Visual output showing detected text regions.

## Conclusion

This project demonstrates the practical application of image processing and OCR techniques in automated text recognition. It also highlights the importance of preprocessing, confidence evaluation, and performance testing when developing computer vision applications.

## Author

**Nwonu Ikenna Basil**

DecodeLabs AI/ML Internship Project 4
