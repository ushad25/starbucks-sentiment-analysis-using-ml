# starbucks-sentiment-analysis-using-ml
# Starbucks Sentiment Analysis: Uncovering Customer Feedback Using NLP and Logistic Regression(TASK 2)

# **Sentiment Analysis on Starbucks Customer Reviews**  
*A Machine Learning Project by CodeTech IT Solutions*

---

## **Author Information**  
| **Author**        | **Usha Rahul**              |  
|--------------------|-----------------------------|  
| **Date**          | January 7, 2025            |  
| **Company**       | CodeTech IT Solutions      |  
| **Intern ID**     | CT0806HT                   |  
| **Domain**        | Machine Learning           |  
| **Mentor**        | Neela Santhosh Kumar       |  
| **Batch Duration**| December 30, 2024 – February 14, 2025 |  

---

## **Overview**  
This project performs sentiment analysis on Starbucks customer reviews using **TF-IDF Vectorization** and **Logistic Regression**. The aim is to classify reviews into three sentiment classes: **Negative**, **Neutral**, and **Positive**, providing valuable insights into customer experiences.  

### **Highlights**  
- Advanced text preprocessing with NLP techniques.  
- TF-IDF for feature extraction.  
- Logistic Regression for classification.  
- Evaluation metrics such as precision, recall, and F1-score.  

---

## **Dataset Description**  
The dataset consists of Starbucks customer reviews scraped from the ConsumerAffairs website.  

### **Features:**  
- **Name**: Reviewer's name.  
- **Location**: Reviewer’s city/location.  
- **Date**: Date of the review.  
- **Rating**: Star rating (1 to 5).  
- **Review**: Textual review content.  
- **Image Links**: Links to associated images.  

---

## **Project Workflow**  
1. **Data Preprocessing**: Cleaning, removing stop words, and lowercasing.  
2. **Feature Extraction**: Using **TF-IDF Vectorization** to convert text into numerical data.  
3. **Modeling**: Logistic Regression to classify sentiments.  
4. **Evaluation**: Metrics include accuracy, precision, recall, and F1-score.  
5. **Visualization**: Sentiment distribution, confusion matrix, and insights.  

---

## **Technologies Used**  
- **Programming Language**: Python  
- **Libraries**:  
  - Data Processing: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`  
  - NLP: `nltk`, `scikit-learn`  

---

## **Results**  
- **Accuracy**: **71%**  
- **Class Performance**:  
  - **Negative**: High recall (1.0), precision (0.68).  
  - **Neutral**: Poor performance (0.0 metrics).  
  - **Positive**: High precision (0.93), low recall (0.28).  

---

## **Challenges and Recommendations**  
- **Class Imbalance**: Neutral class underrepresented; requires techniques like SMOTE or oversampling.  
- **Model Tuning**: Experiment with XGBoost or Random Forest.  
- **Feature Engineering**: Include n-grams, lemmatization, and embeddings like Word2Vec.  

---



### **Contact Information**  
For any queries or contributions, please contact:  
**Email**: ushamuth18@gmail.com  

---

