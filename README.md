# 🧠 Mental-Health-Risk-Web-Application 

                                            

<br><br>

## 📚 Problem Statement

Mental health issues are often underdiagnosed due to a lack of awareness and timely intervention. 
Our _Mental Health Risk Prediction Web Application_ aims to bridge this gap by using machine learning to identify individuals at risk and encourage early intervention. This project empowers users to gain insights into their mental health and provides valuable resources to promote positive mental health outcomes.

<br>



## 🎯 Why This Matters

With rising mental health concerns worldwide, leveraging technology to provide timely predictions and personalized recommendations can make a real difference. By analyzing user inputs, our application predicts potential mental health risks and offers tailored support suggestions.

<br>

## 📊 Dataset Overview:


### - Source of the Data?
[Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)


### - What's in the Dataset?
This dataset contains 1,239 responses from a 2014 survey that measures attitudes towards mental health and frequency of mental health disorders in the tech workplace. 


#### Key Features:
Age, gender, work environment, family history, previous diagnosis, willingness to seek help, etc. 

#### Target: 
treatment requirement




### - Data Preparation?

Handling missing data using imputation.

Encoding categorical variables to numeric formats.

Standardizing numerical features to ensure consistent scaling.

Feature engineering to improve model performance.


<br>

## ✅ Our Solution


### Step 1: Understanding the Data

We started with exploratory data analysis (EDA) to uncover patterns and correlations in the dataset. This step helped us identify the most critical features for predicting mental health risks.


### Step 2: Model Development

We used Python for data processing, model building, and evaluation. 
Several machine learning models were trained and compared, including:

1. Logistic Regression

2. Random Forest Classifier
   
3. XGBoost



### Step 3: Model Evaluation

We evaluated the models using multiple metrics to ensure the best results:

- Accuracy,
- Precision,
- Recall and
- F1-Score

Recall helped us determine the best model, i.e. Random Forest Classifier.
  

### Step 4: Building the Web Application

Backend & Frontend: We developed an interactive web interface using streamlit to allow users to easily input their data and receive predictions in real-time.


<br>

## 📌 Results and Key Findings


Top Model: Random Forest Classifier achieved a recall of 0.911 and an F1-score of 0.90.

Feature Importance: Insights into the most influential factors affecting mental health.

Observed high recall, reflecting strong model performance.


<br>

## 📈 Output

Clear insights and actionable recommendations with regard to mental health condition. 


<br>

### 🚀 Give It a Try!
