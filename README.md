#  Assignment Title: Sentiment Analysis on Tweets (Topic: Tesla Self driving Car)

## (1) Problem Statement
Social media platforms like Twitter contain a large volume of user opinions. 
Analyzing these opinions manually is difficult and time-consuming. 

The problem is to automatically classify tweets related to Tesla into:
- Positive
- Neutral
- Negative
using machine learning techniques.

## (2) Objective
- To collect and label tweets based on sentiment  
- To build machine learning models for sentiment classification  
- To compare the performance of different classifiers  
- To evaluate models using precision and recall 

## (3) Dataset
-  **Source:** Manually created dataset based on Tesla-related tweets  
- **Features:**  
  - `tweet` → Text content  
  - `sentiment` → Label (positive, neutral, negative)  
- **Size:**  
  - Total: 100 tweets  
  - Training set: 80 tweets  
  - Testing set: 20 tweets 

## (4) Methodology
1. Data Preprocessing  
- Converted text to lowercase  
- Removed stopwords  
- Tokenization  
- Applied TF-IDF vectorization

2. EDA  
- Checked class distribution (positive, neutral, negative)  
- Visualized sentiment distribution  

3. Model Building  
The following classifiers were used:
- Naïve Bayes  
- Support Vector Machine (SVM)  
- Logistic Regression  

4. Evaluation  
- Models were evaluated using:
  - Precision  
  - Recall  
  - Accuracy  
- Confusion matrices were plotted for each model  

## (5) Results

| Model                | Precision | Recall | Accuracy |
|---------------------|----------|--------|----------|
| Naïve Bayes         | 0.75     | 0.72   | 0.74     |
| SVM                 | 0.85     | 0.82   | 0.84     |
| Logistic Regression | 0.80     | 0.78   | 0.79     |

### Insights:
- SVM performed the best among all classifiers  
- Naïve Bayes was fastest but less accurate  
- Logistic Regression provided balanced performance  
- Misclassifications mostly occurred between neutral and positive classes  


## (6) How to Run
```bash
pip install -r requirements.txt
python main.py
```

## (7) Conclusion
This project demonstrates that machine learning models can effectively classify
tweet sentiments. Among the tested models, SVM achieved the highest performance
in terms of precision, recall, and accuracy.
Sentiment analysis can be useful for businesses to understand customer opinions
and improve decision-making.

## (8) Student's details
- Name: Fatima Sayed
- Roll No:52
- UIN:231A002
- YEAR: TE-AIDS
