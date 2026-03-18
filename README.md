# Explainable_AI 

## Overview

This project investigates how different explainable AI (XAI) methods explain predictions of a BERT-based toxicity classifier. The goal is to compare explanation methods and evaluate whether they highlight similar important words in text.

The project focuses on three methods:

Attention (from BERT)

Integrated Gradients (IG)

LIME (Local Interpretable Model-agnostic Explanations)

Explanations are compared using Jaccard similarity on the top-5 most important tokens.

## Research Goal

The main research question is:

In a BERT-based toxicity classifier, how do attention, Integrated Gradients, and LIME differ in token importance, and how much do their explanations agree with each other?

Additionally, the project evaluates:
  - Agreement between explanation methods
  - Alignment with human intuition
  - Behavior on identity-related text

## Repository Structure:
<pre> 
Explainable_AI/
│ 
├── toxicity_classifier.py # Main script (runs full demo) 
├── README.md # Project documentation 
├── requirements.txt # Dependencies
└── data/ # Dataset folder 
  └── civil_comments_toxic_sample.csv # 250 samples labelled as toxic from the original civil comments dataset
</pre>

## Setup Instructions
### 1. Clone the repository
git clone https://github.com/SemDdVries/Explainable_AI.git
cd Explainable_AI

### 2. Create a Virtual Environment using Python version 3.12.x
Version 3.12 is required as most installments will not work on python versions 3.13 and up. 
To create the environment, use the steps below:
1. Check python version by running "python --version"
2. If version 3.12.x is listed, skip the following steps and go to installments. Else, follow the next steps.
3. Run "py -0" to check your installed versions. If 3.12 is not listed, download it here: https://www.python.org/downloads/
4. Create the virtual environment:
   py -3.12 -m venv .venv #Windows
   python3.12 -m venv .venv #Mac/Linux
5. Activate the environment:
   source .venv/Scripts/activate #Bash
   .venv\Scripts\activate.bat #Command prompt (VScode terminal)
6. Verify version: python --version
7. It should now list version 3.12.x

### 3. Download requirements
run "pip install pandas numpy matplotlib torch transformers captum lime" to install all neccessary libraries

## Dataset
This project uses a sample of the Jigsaw Civil Comments dataset:

For more information about this dataset and to find a download link:
https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data

A sample containing 250 rows with a toxicity score of 0.5 or higher is given in the data folder, so no download is neccessary. 

## What The Code Does
### 1. Model
Uses a pretrained BERT model:

  *unitary/toxic-bert*

This model predicts toxicity probabilities for text input.
  
### 2. Data Preparation
  - Loads the dataset (commented out as it's not needed for a demo) 
  - Converts toxicity scores into binary labels (toxic / non-toxic)
  - Samples 250 comments for faster experimentation
  - Saves a smaller dataset for reuse

### 3. Explanation Methods
**Attention**
  - Extracts attention weights from the last transformer layer
  - Uses CLS-token attention as importance scores

**Integrated Gradients**
  - Computes gradients of input embeddings
  - Measures contribution of each token to the prediction

**LIME**
  - Generates modified samples of the text
  - Trains a local interpretable model
  - Returns most influential words

### 4. Evaluation
The top 5 most important tokens are selected

Agreement between methods is measured using Jaccard similarity over 5 identity-related comments:

J(A, B) = $\frac{1}{5}\sum_{i=1}^{5}\frac{|A_i ∩ B_i|}{|A_i ∪ B_i|}$

The result is shown in a plot

### 5. Output
The script produces a bar plot showing similarity between:
  - Attention vs IG
  - IG vs LIME
  - Attention vs LIME

And a terminal output with:
  - Top 5 tokens per method on a demo sentence
  - Jaccard scores (in the same way as listed above) for a demo sentence

## Limitations
  - Small sample size is used (250 comments)
  - LIME uses reduced sampling (n=100 instead of n=5000)
  - Explanations may differ across methods
  - Only one demo sentence is shown, of which the results may differ per run

## Author

Sem de Vries
s1093742
2nd year Bsc AI student



