# Student Performance Prediction & Risk-Group Clustering

Predicting students' final marks (G3) and grouping students into risk
categories using machine learning.

## Dataset

UCI Student Performance dataset (Portuguese secondary schools),
maths subject: 395 students, 33 features.
Source: https://archive.ics.uci.edu/dataset/320/student+performance

## Project Structure

```
student-performance-ml/
├── student_performance_analysis.ipynb  # Main notebook (EDA + models + clustering)
├── student-mat.csv                     # Dataset (Maths subject, 395 students)
├── requirements.txt                    # Python dependencies
└── README.md
```

## Methods

- Regression: Ridge, SVR, Gradient Boosting, MLP (neural network)
- Two versions: with previous grades (G1/G2) and without (early prediction)
- Clustering: K-Means (k=3), visualised with PCA

## Model Results

### With Previous Grades (G1/G2)

| Model             | MAE  | RMSE | R²   |
|-------------------|------|------|------|
| Gradient Boosting | 1.16 | 2.00 | 0.80 |
| Ridge             | 1.64 | 2.37 | 0.73 |
| SVR               | 1.84 | 2.73 | 0.64 |
| MLP (ANN)         | 2.02 | 2.78 | 0.62 |

### Without Previous Grades (Early Prediction)

| Model             | MAE  | RMSE | R²    |
|-------------------|------|------|-------|
| Gradient Boosting | 3.23 | 4.03 | 0.21  |
| Ridge             | 3.39 | 4.19 | 0.14  |
| SVR               | 3.40 | 4.23 | 0.13  |
| MLP (ANN)         | 4.35 | 5.27 | -0.35 |

## Key Findings

1. Gradient Boosting was best with G1/G2 (MAE 1.16, R² 0.80).
2. Without G1/G2, all models struggled (best R² 0.21), so previous
   grades carry most of the signal.
3. The MLP did worst, likely because the dataset is small (316 training rows).
4. Clustering found three groups: On track, Lifestyle-risk,
   and Academically struggling (grades falling from G1 to G3).

## Limitations

- Data is from Portugal, so results may not apply to Kerala.
- Small dataset; the test set is only 79 students.
- G3 = 0 likely means dropout or skipped exam.

## How to Run

1. Clone the repo and upload **`student-mat.csv`** to your environment
2. Install dependencies: `pip install -r requirements.txt`
3. Open `student_performance_analysis.ipynb` in Jupyter or Google Colab and run all cells

> **Note:** The notebook references `student-por.csv` in an early cell — you can safely comment that line out, as the main analysis only uses `student-mat.csv`.

## Next Steps

- Add hyperparameter tuning (GridSearchCV)
- Try the same pipeline on the Portuguese-subject dataset

## Demo App
A Streamlit app predicts a student's final mark and risk group.

    pip install -r app/requirements.txt
    streamlit run app.py
