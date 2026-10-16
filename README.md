# Multi-label Toxic Comment Classification

NLP project to detect multiple types of toxicity in a single comment (a comment can have more than one label).

## Dataset
Jigsaw Toxic Comment Classification Challenge (Kaggle), about 160k Wikipedia comments.
Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate.
Download train.csv from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data and place it in the data/ folder.
Note: the data contains offensive language.

## Approach
1. EDA: label counts, labels per comment, label correlation
2. Text cleaning: lowercase, remove new lines
3. TF-IDF features (unigrams and bigrams, 30000 features)
4. Multi-label models using One-vs-Rest: Logistic Regression and Linear SVM (class_weight='balanced')
5. Evaluation: per-label precision, recall, F1, macro F1, macro ROC-AUC, hamming loss

## Results
- Best model: <naam daalo>
- Macro F1: <value daalo>
- Macro ROC-AUC: <value daalo>

## Key Insights
- <Cell 4 ke percentage se ek line>
- <Cell 5 se ek line>
- <Cell 7 ya Cell 11 se ek line>

## How to run
pip install -r requirements.txt
cd src
python train_model.py
python predict.py