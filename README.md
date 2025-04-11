# Building energy efficiency classification
**Tools:** Python, scikit-learn, Pandas, NumPy, Seaborn, Matplotlib

**Summary:**

This project classifies residential buildings into energy efficiency categories based on architectural features using machine learning. It includes data preprocessing, feature engineering, model training with hyperparameter tuning, and ensemble learning through a VotingClassifier.

**Dataset:** Energy efficiency dataset (UCI) (https://archive.ics.uci.edu/dataset/242/energy+efficiency)

**Analysis:**

- Performed data cleaning, correlation analysis, and quantile-based classification of energy loads
- Engineered a new target feature `charges_classes` from heating and cooling load sums
- Standardized numerical features and split dataset into train/test sets
- Trained and tuned KNN, SVM, and Random Forest classifiers using GridSearchCV
- Evaluated performance using confusion matrices and accuracy scores
- Combined top models into a hard voting ensemble to improve classification accuracy

**Key Skills:** Data preprocessing, classification modeling, hyperparameter tuning, ensemble methods, model evaluation, feature engineering