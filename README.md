# Analyzing-COVID-19-Sentiments-on-Twitter-An-Effective-Machine-Learning-Approach

# Project Overview
This repository is based on the research article titled "Analyzing COVID-19 Sentiments on Twitter: An Effective Machine Learning Approach", which explores the application of machine learning techniques for sentiment classification on COVID-19-related tweets.

# Purpose
The aim of the study is to assess public sentiment—categorized as positive, negative, or neutral—by analyzing Twitter posts shared during the COVID-19 pandemic. Understanding these sentiments provides valuable insights for policymakers, health organizations, and researchers studying public behavior and communication during global crises.

# Methodology
##à Text Preprocessing

### Auto-correction of misspellings

### Tokenization

### Stop-word removal using NLTK

### Stemming and lemmatization via WordNet

# Feature Extraction

### Employed CountVectorizer and TF-IDF to convert textual data into numerical vectors

# Machine Learning Model

### Utilized Multinomial Logistic Regression (MLR) for multi-class classification

### Compared with existing approaches like SVM, Random Forest, and BERT-based classifiers

# Results
### Achieved an accuracy of 95.14% with the MLR model

### High precision and F1-scores across all sentiment classes

### Demonstrated improved performance with increased training data

### Applied K-Fold Cross-Validation for robust evaluation


The study confirms that Multinomial Logistic Regression is a simple yet highly effective model for real-time sentiment analysis on social media platforms.

The approach proves scalable and adaptable to other domains involving short text classification.

Insights from this analysis can help track public emotion trends and support timely decision-making during health emergencies.

