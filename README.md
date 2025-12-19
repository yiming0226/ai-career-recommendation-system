# AI Career Recommendation System

This project is an AI-powered system that parses resumes, extracts skills, and recommends suitable career paths. It includes:

- Resume parsing (PDF input)
- Skill extraction
- Career path recommendation using ML models (SVM, LR, KNN, DT.)
- Top-K recommendation with explanations
- Interactive interface using Gradio

## Folder Structure

- `data/` : datasets, cleaned datasets, test resumes, and output files
- `training/` : training scripts and saved ML models
- `parsers/` : resume parsing modules
- `recommender/` : career recommendation modules

## Usage

1. Run the Gradio app: `app.py`.
2. Upload resume and click "Process resume" button.
3. Check parsed data and recommendations.

## Notes

- `.gitignore` is configured to ignore datasets and model files (`.pkl`, `.joblib`).
- `.DS_Store` files are removed and ignored.

## Supplementary Materials

Due to data privacy, file size constraints, and ethical considerations, the full datasets, trained models, and demo resumes are **not included in this GitHub repository**.

These resources are available in a separate OneDrive folder:

**[FYP_Project_Supplementary_Materials – Access Here](https://drive.google.com/drive/folders/1nxfx_SO8c4naqWUEw-OhTz7gBqHxOeuS?usp=sharing)**

> The folder is shared with **view-only access** to ensure data integrity while allowing examiners to download and evaluate the system.
