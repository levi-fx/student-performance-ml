# student-performance-ml
# Student Performance Prediction & Risk-Group Clustering

Predicting students' final marks (G3) and grouping students into risk
categories using machine learning.

## Dataset
UCI Student Performance dataset (Portuguese secondary schools),
maths subject: 395 students, 33 features.
Source: https://archive.ics.uci.edu/dataset/320/student+performance

## Methods
- Regression: Ridge, SVR, Gradient Boosting, MLP (neural network)
- Two versions: with previous grades (G1/G2) and without (early prediction)
- Clustering: K-Means (k=3), visualised with PCA

## Key Findings
1. Gradient Boosting was best with G1/G2 (MAE 1.16, R2 0.80).
2. Without G1/G2, all models struggled (best R2 0.21), so previous
   grades carry most of the signal.
3. The MLP did worst, likely because the dataset is small (316 training rows).
4. Clustering found three groups: On track, Lifestyle-risk,
   and Academically struggling (grades falling from G1 to G3).

## Limitations
- Data is from Portugal, so results may not apply to Kerala.
- Small dataset; the test set is only 79 students.
- G3 = 0 likely means dropout or skipped exam.

## How to run
Open the notebook in Google Colab, upload student-mat.csv, run all cells.
