# Diabetes Prediction
This data science project is about prediction of diabetes, based on personal medical characteristics, using supervised machine learning models. The project was implemented with python's data analysis and visualizations libraries (sklearn, pandas, numpy, matplotlib, seaborn).

Background – diabetes is a chronic metabolic disease, characterized by a high blood sugar level over time and disorders in the insulin hormone's function.

Project's data – medical data about 800 adult women, with and without diabetes. Can be found at https://www.kaggle.com/uciml/pima-indians-diabetes-database

Data exploration (EDA.py) – first glance on the data, visualizing features' general distributions, comparing features' distributions of women with and without diabetes and checking correlations between features. It already can be noticed that glucose, bmi, insulin might impact on diabetes onset.

Data Completion (KNNFillData.py) - some features have large amount of missing data, knn was used to complete it, optimal k was chosen by using the elbow curve method. Data completion was made based on correlation with some other features in the data. Features with small amount of missing data were completed by using measures of central tendency.

Feature Engineering (FeatureEngineering.py) – some features were divided to categories (for example bmi) to check relations of features' categories with the outcome (by looking at general correlation and pie charts among diabetic and non-diabetic women).

Models (predict.py) – logistic regression, random forest, naïve bayes, svm, gradient boosting trees and all models' ensemble were used to predict if a woman from the dataset's population might be diagnosed with diabetes, based on her medical characteristics.

Conclusions – random forest had the best accuracy and recall, svm has the best precision. The ensemble does not outperform some of the models. Random Forest's feature importance indicates that glucose, bmi, insulin indicators were more prominent in the diabetes prediction.

