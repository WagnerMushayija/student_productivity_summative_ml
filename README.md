# Student Productivity Prediction

Predicting whether a student has high or low productivity using machine learning and deep learning models. Built for the Summative Assignment in Introduction to Machine Learning.

## Project Overview

This project uses a dataset of 20,000 students to predict if a student's productivity score is above or below the median (binary classification: high = 1, low = 0).  
The goal is to help teachers and schools spot students who might need extra support early — like better sleep habits, less phone time, or more study structure.

I compared regular machine learning models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) with deep learning neural networks (TensorFlow Sequential models with different experiments).  
**Main finding:** Tree-based models (especially XGBoost) performed best on this tabular data.

This work aligns with my mission to make education fairer, especially in places like Rwanda where many students face big classes and limited resources.

## Dataset

- File: `student_productivity.csv`
- Rows: 20,000 students
- Features: 18 columns (age, gender, study_hours_per_day, sleep_hours, phone_usage_hours, social_media_hours, youtube_hours, gaming_hours, breaks_per_day, coffee_intake_mg, exercise_minutes, assignments_completed, attendance_percentage, stress_level, focus_score, final_grade, productivity_score)
- Target: Binary `target` created by splitting `productivity_score` at the median (50.23)
- Source: Open online dataset (not from sklearn/keras built-ins)
- No missing values, perfectly balanced classes after split (10,000 high / 10,000 low)

## Key Experiments & Results

### Traditional ML
- Logistic Regression → AUC ~0.9285
- Random Forest → AUC ~0.9280
- Gradient Boosting → AUC ~0.9463
- **XGBoost** → AUC ~0.9406 (strong performer)

### Deep Learning (TensorFlow Sequential)
- Baseline (256→64, Adam, dropout 0.3, L2 0.001, batch 256) → AUC ~0.9498
- Deeper architecture → AUC 0.9471
- Lower learning rate (0.0003) → AUC 0.9487
- RMSprop optimizer → AUC 0.9487
- High dropout (0.4) → AUC 0.8423 (underfitting)
- Smaller batch size (64) → AUC 0.9479
- Stronger L2 (0.005) → AUC 0.9487

**Winner:** XGBoost — best balance of AUC, accuracy, and recall for high productivity.

[See full results table in the notebook → Overall Summary section]

## Files in this Repository

- `summative_intro_to_ml_student_productivity.ipynb` → Main notebook with all code, experiments, visualizations, and explanations
- `student_productivity.csv` → The dataset (if allowed to upload; otherwise download from source)
- `README.md` → This file
- [Youtube Link][((https://youtu.be/WvypheV1W00))]

## How to Run

1. Open the notebook in Google Colab or Jupyter
2. Upload `student_productivity.csv` (or change path if it's already in your drive)
3. Run cells top to bottom
4. All libraries are imported in the first cell — no extra installs needed (uses standard Colab environment)

Requirements:
- Python 3
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- tensorflow
- xgboost

## Key Insights

- Tree models (XGBoost, Gradient Boosting) beat neural networks on this tabular dataset — common pattern when features are few and data isn't huge.
- Daily habits matter most: good sleep, limited phone/social media, regular study hours, and exercise strongly predict high productivity.
- Removing highly correlated features (e.g., attendance and assignments) helped models learn cleaner patterns.
- Neural nets hit a ceiling around AUC 0.949 — tweaking architecture/LR/optimizer didn't break through.

## Future Work Ideas

- Tune XGBoost hyperparameters with Optuna or GridSearch
- Add SHAP explanations to show which habits hurt productivity most
- Build a simple web app for teachers to input student data and get predictions
- Test on real school data (with privacy protection)

## License

MIT License — feel free to use and modify.

Made by Wagner in Kigali, Rwanda  
For Introduction to Machine Learning Summative Assignment  
February 2026
