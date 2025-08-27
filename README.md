#📊 Salary Prediction using Regression

📌 Project Overview

This project aims to predict salary based on academic and related features using Regression Analysis.
Currently, the dataset contains percentage in grade 10 as the primary predictor, and we are experimenting with extending the model by adding extra attributes to improve accuracy.

The model has been evaluated with baseline results:

Root Mean Squared Error (RMSE): 29,281.42

R-squared (R²): 0.7198

This means the model explains ~72% of the variance in salaries, which is a strong baseline, but there’s scope for improvement.

🗂️ Dataset

Size (current): 500 entries (synthetic dataset).

Columns:

S No → Serial number

Percentage in Grade 10 → Student’s academic performance

Salary → Annual salary (target variable)

🔹 In the next stage, more features (attributes) like experience, degree type, and job location will be added for better predictive power.

⚙️ Workflow

Data Collection & Preparation

Gathered synthetic dataset of percentages vs salaries.

Cleaned and structured using pandas.

Exploratory Data Analysis (EDA)

Scatter plots and trend analysis to visualize correlation.

Summary statistics of features.

Model Building

Linear Regression using scikit-learn.

Training/Test split performed.

Model Assumptions Check

Checked residual distribution (Normality).

Performed Durbin-Watson test for independence (≈1.98, which is good).

Model Evaluation

Metrics used: RMSE, R².

Results:

RMSE ≈ 29,281.42

R² ≈ 0.7198

Next Steps

Add more features/attributes (experience, job role, degree, etc.).

Try advanced models: Polynomial Regression, Ridge, Lasso.

Hyperparameter tuning.

Compare performance improvements.

📦 Tech Stack

Language: Python

Libraries:

pandas (data handling)

numpy (numerical operations)

matplotlib / seaborn (visualization)

scikit-learn (modeling & evaluation)

statsmodels (assumption testing)

📈 Results & Insights

Percentage in grade 10 shows a strong correlation with salary but is not the only determinant.

Adding more attributes is expected to reduce error and improve R² significantly.

Current baseline model is a solid foundation to build upon.

🚀 Future Work

Add extra features: experience, degree, industry, location.

Experiment with ensemble methods (Random Forest, XGBoost).

Deploy as a simple web app (Streamlit/Flask) for interactive salary prediction.

👨‍💻 Author

Shivanshu Maurya

B.Tech CSE (2nd Year)

Enthusiast in Machine Learning & Competitive Programming
