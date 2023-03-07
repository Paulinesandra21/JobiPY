
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from itertools import cycle
from joblib import load
from utils import role_distribution,age_distribution,Ide_distribution, ML_distribution,education_vs_role,age_vs_role,age_vs_codingexperience,language_vs_role
#df = pd.read_csv('C:/Users/Julia/Documents/Python Scripts/kaggle_survey_2020_responses.csv')
st.set_page_config(layout="centered")
df = pd.read_csv('kaggle_survey_2020_responses.csv')
st.markdown("""
            <style>
            .css-9s5bis.edgvbvh3
            {visibility: hidden;}
            .css-h5rgaw.egzxvld1
            {visibility: hidden;}
            .css-18ni7ap.e8zbici2
            {visibility: hidden;}
            </style>
            """, unsafe_allow_html = True)
st.sidebar.title('**Data job**')


pages = ['About', 'Data Visualisation', 'Modeling - Strategy', 'Modeling - Survey','Conclusion']
page = st.sidebar.radio('**Choose a page**:', pages)

st.sidebar.markdown('**App created by:**')
st.sidebar.markdown('Julia Burkhardt')
st.sidebar.markdown('Pauline Sandra Paul Stephen Raj')
st.sidebar.markdown('**Project Mentor:**')
st.sidebar.markdown('Maëlys BASSILEKIN BLANC')

                                                                                                     

if page == pages[0]:
    st.title('Predicting the ideal profession for you!')
    st.markdown('''We are right now living in a data-driven world, where data plays a pivotal role in our lives. Data analysts/data scientists are currently a much-sought professionals
    around the globe. Data Science unsurprisingly has become a fascinating field to many non-IT professionals as well.  With so much growth 
    in data science and the opportunities it brings, it is imperative to acquire the best of skills that would make the candidacy suitable for a 
    given post, especially for students and young professionals.
    ''')
    st.image('Yougotthis.jpg', caption= 'Photo by Prateek Katyal on Unsplash')
    st.subheader('Aim')
    st.markdown('The objective of the DATA JOB app is to help users find their ideal job using Machine Learning (ML) algorithms based on their acquired skills, education, and experience.')
    st.subheader('Datasets')
    st.markdown('''The dataset [2020 Kaggle Machine Learning & Data Science Survey](https://www.kaggle.com/c/kaggle-survey-2020/overview) 
    was collected from the Kaggle Database www.kaggle.com. This dataset contains the responses from 20 000 participants to an industry-wide online 
    survey conducted by Kaggle in 2020.The total size of the dataset is 25.59 MB and it contains the following two types of datasets-''')
    st.markdown('**1. Main data**')
    st.markdown(''' There are around 40 Questions & Responses from 20 000 participants recorded in kaggle_survey_2020_responses.csv file. Multiple choice questions are of two
                types - i) A single choice responses, that are recorded in individual columns,
                and ii) Multiple choice responses, that are split into multiple columns (with
                one column per answer choice).''')
    st.dataframe(df.head())
    st.markdown('**2. Supplementary data**')
    st.markdown(''' The list of answer choices for every question, and footnotes describing which questions are
                asked to which respondents are recorded in kaggle_survey_2020_answer_choices.pdf file''')
    st.markdown(''' The description of how the survey
                was conducted is available in kaggle_survey_2020_methodology.pdf.''')   

if page == pages[1]:
    st.title('Exploratory data analysis')
    EDA = st.selectbox('Please select',options = ('Global overview', 'Relationship with target variable(Current Profession)', 'Relationship between Age and Coding experience') )
    if EDA == 'Global overview':
        role_distribution()
        st.markdown('There are a total of 11 data professions and we found that most of the participants are Data scientists, followed by Software Engineers, and the least participants are of  DBA/Database Engineers. Please note that we have dropped two categories from the list of current profession because we were interested only the responses from the participants who are employed. ' )
        age_distribution()
        st.markdown('Most of the participants are of the age group 25-29 years, followed by 30-24 & 22-24 years. We found that there is a gradual decrease after 25-29 age groups.')
        Ide_distribution()
        st.markdown('The Jupyter IDE is the widely used IDE among the data professionals, followed by Visual Studio Code and PyCharm.')
        st.markdown('**Distribution of ML algorithms**')
        st.image('ML_algorithms_Distribution.png')
        st.markdown('Linear or Logistic regression is the prominently used ML algorithm in Data jobs, followed by Decision Trees or Random Forests and Convolutional Neural Networks. The Gradient Boosting Machines are also trending among data professionals. ')
    if EDA == 'Relationship with target variable(Current Profession)':
        education_vs_role()
        st.markdown('Most of participants are Master degree holders, followed by Bachelor degree and Doctoral degree. Software Engineers, Data Scientist, and Research Scientist are the predominant professions of Bachelor, Master, and Doctoral degree holders respectively.')
        age_vs_role()
        st.markdown('We can see that for different age groups, there is a tendency for a specific type of jobs. Data Scientist has a tremendous rise until age group 40, then the Project Managers takes the lead, showing that the professionals after plenty of experiences takes up the lead roles in their team. ')
        language_vs_role()
        st.markdown('Python is the most popular language used by different professions. Data Scientist primarly uses Python followed by SQL, R, Julia and Bash. Among Statistician, R is the most popular language. So this clearly suggest to us that the Programming language is profession dependent. ')
    if EDA == 'Relationship between Age and Coding experience':
         age_vs_codingexperience()
         st.markdown('As clearly depicted in the figure, professionals gain coding experiences as they grow older.')
if page == pages[2]:
    st.title('Modeling')
    st.markdown('Our goal is to select the best model based on the performance. ') 
    st.markdown('Three different strategies are carried out in our project.')
    strategies = st.selectbox('Please select the Strategy',options = ('Strategy 1', 'Strategy 1.1', 'Strategy 2') )
    if strategies == 'Strategy 1':
        st.markdown('In this strategy, we have kept all the 11 professions (target variable) to train the models on.')
        col1, col2, col3 = st.columns([1,6,1])
        with col2:
            st.subheader("PERFORMANCE ON DIFFERENT MODELS")
            st.image("Figure_1a.png")
            st.subheader("CLASSIFICATION REPORT")
        col1, col2, col3, col4 = st.columns([1,1,1,1])
        with col1:
            st.image("Figure_3.png")
        with col2:
            st.image("Figure_4.png")
        with col3:
            st.image("Figure_5.png")
        with col4:
            st.image("Figure_6.png")
        col5, col6, col7, col8 = st.columns([1,1,1,1])
        with col5:
            st.image("Figure_7.png")
        with col6:
            st.image("Figure_8.png")
        with col7:
            st.image("Figure_9.png")
        with col8:
            st.image("Figure_10.png")
        col9, col10, col11 = st.columns([1,1,1])
        with col9:
            st.image("Figure_11.png")
        with col10:
            st.image("Figure_12.png")
        with col11:
            st.image("Figure_13.png")
        st.subheader('Methods used to address OverFitting')
        overfitting = st.selectbox('Please select',options = ('Stratified K fold cross validation', 'PCA') )
        if overfitting == 'Stratified K fold cross validation':
            st.image("Skcv_strategy_1.png")
        if overfitting == 'PCA':
            st.image('Strategy_1_pca.png')
            st.markdown('In Strategy 1, Random Forest performed better in both the methods. By PCA, we could acheive an accuracy of ~40% only on test data after tuning parameters with GridSearch 10-CV. ')
    if strategies == 'Strategy 1.1':
        st.markdown('In this strategy , we have excluded two classes “Data Base Analyst/Engineer” and “Product/Project Manager” as we observed low precision & recall scores in strategy 1.')
        col1, col2, col3 = st.columns([1,6,1])
        with col2:
            st.subheader("PERFORMANCE ON DIFFERENT MODELS")
            st.image("Fig_1a.png")
            st.subheader("CLASSIFICATION REPORT")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            st.image("Fig_2.png")
        with col2:
            st.image("Fig_3.png")
        with col3:
            st.image("Fig_4.png")
        
        col4, col5, col6= st.columns([1,1,1])
        with col4:
            st.image("Fig_5.png")
        with col5:
            st.image("Fig_6.png")
        with col6:
            st.image("Fig_7.png")
        col7, col8,col9 = st.columns([1,1,1])   
        with col7:
            st.image("Fig_8.png")
        with col8:
            st.image("Fig_9.png")
        with col9:
            st.image("Fig_10.png")
        overfitting = st.selectbox('Please select',options = ('Stratified K fold cross validation', 'PCA') )
        if overfitting == 'Stratified K fold cross validation':
            st.image("Skcv_strategy_1.1.png")
        if overfitting == 'PCA':
            st.image('Strategy_1.1_pca.png')
            st.markdown('In this Strategy, Random Forest again performed better in both the methods. By Stratified K fold Cross Validation, we could acheive an accuracy of 45% only on test data after tuning parameters with GridSearch 10-CV. In the scatter plot, we see that the individual classes are all clustered together and not discriminated.')
    if strategies == 'Strategy 2':
        st.markdown('In this Strategy, we have only kept the data of the five most represented jobs as target variable.')
        st.subheader("PERFORMANCE ON DIFFERENT MODELS")
        col1, col2, col3 = st.columns([1,6,1])
        with col2:
            st.image("Figu_1.1.png")
            st.subheader("CLASSIFICATION REPORT")
            st.image("Classreports.png")
        overfitting = st.selectbox('Please select',options = ('Stratified K fold cross validation', 'PCA') )
        if overfitting == 'Stratified K fold cross validation':
            st.image("Skcv_strategy_2.png")
        if overfitting == 'PCA':
            st.image('Strategy_2_pca.png')
            st.markdown('In this Strategy, Random Forest again performed better in both the methods. By Stratified K fold Cross Validation & PCA, we could acheive an accuracy of 54% only on test data after tuning parameters with GridSearch 10-CV. In the scatter plot, we see that the individual classes are all clustered together and not discriminated.')
        Boosting_tech = st.selectbox('Select Boosting technique',options = ('XGBoost', 'AdaBoost') )
        if Boosting_tech == 'XGBoost':
            code = '''Test score: 0.607'''
            st.code(code, language = 'python')
        if Boosting_tech == 'AdaBoost':
            code = '''Test score: 0.583'''
            st.code(code, language = 'python')
        st.subheader('Feature selection')
        st.markdown('''We employed **feature selection** methods like **mutual information gain**, **chi-square**, and **variance threshold** to address overfitting. Out of the 217 features, 72 features that passed variance threshold was selected and trained on XGBoost ML algorithm. Unfortunately, we could obtain only 43% accuracy with our method. Due to time constraints, we couldn't improve them with other parameters and options.''')
if page == pages[3]:
    model = load('rf.joblib')
    age_label = {'18-21':0, '22-24':1, '25-29':2, '30-34':3, '35-39':4, '40-44':5, '45-49':6, '50-54':7, '55-59':8, '60-69':9, '70+':10}
    education_label ={'Bachelor\'s degree':0, 'Doctoral degree':1, 'I prefer not to answer':2, 'Master\'s degree':3, 'Highschool':4, 'Professional degree':5, 'College/University without Bachleor\'s degree':6} 
    codingexp_label = {'1-2 years':0, '10-20 years':1, '20+ years':2, '3-5 years':3, '5-10 years':4, '< 1 years':5, 'None':6}

    # Get value from dictionary
    def get_value(val, my_dict):
        for key, value in my_dict.items():
            if val==key:
                return value

    # creating function for prediction
    def datajob_prediction(input_data):
        input_data_as_array = np.array(input_data)
        input_data_reshaped = input_data_as_array.reshape(1, -1)

        prediction = model.predict(input_data_reshaped)
        
        if (prediction[0] == 2):
            return 'You should become a Data Engineer!'
        elif (prediction[0] == 4):
             return 'You should become a Data Scientist!'
        elif (prediction[0] == 5):
            return 'You should become a Machine Learning Engineer!'
        elif (prediction[0] == 8):
            return 'You should become a Research Scientist!'
        else:
            return 'You should become a Software Engineer!'

    st.title('Data Job Prediction')
    
    #getting the input data from the user
    Age = st.selectbox('What is your age', tuple(age_label.keys()))
    Ed = st.selectbox('What is your highest level of education?', tuple(education_label.keys()))
    Codingyears = st.selectbox('For how many years have you been writing code and/or programming?', tuple(codingexp_label.keys()))
    st.write('**What programming languages are you familiar with? Select all that apply.**')
    Language_1 = st.checkbox('Python')
    Language_2 =st.checkbox('R')
    Language_3 = st.checkbox('SQL')
    Language_4 = st.checkbox('C') 
    Language_5 = st.checkbox('C++')
    Language_6 = st.checkbox('Java')
    Language_7 = st.checkbox('Javascript')
    Language_8 = st.checkbox('Julia')
    Language_9 = st.checkbox('Swift')
    Language_10 = st.checkbox('Bash')
    Language_11 = st.checkbox('MATLAB')
    Language_12 = st.checkbox('None of the above languages')
    Language_13 = st.checkbox('Other language(s)')
    st.write('**What IDEs are you familiar with? Select all that apply.**') 
    IDEs_1 = st.checkbox('JupyterLab')
    IDEs_2 = st.checkbox('RStudio')
    IDEs_3 = st.checkbox('Visual Studio')
    IDEs_4 = st.checkbox('VS Code')
    IDEs_5 = st.checkbox('PyCharm')
    IDEs_6 = st.checkbox('Spyder')
    IDEs_7 = st.checkbox('Notepad++')
    IDEs_8 = st.checkbox('Sublime Text')
    IDEs_9 = st.checkbox('Vim, Emacs or similar')
    IDEs_10 = st.checkbox('Matlab')
    IDEs_11 = st.checkbox('None of the above IDEs')
    IDEs_12 = st.checkbox('Other IDEs')
    st.write('**What data visualization tools are you familiar with? Select all that apply.**')
    DataViz_1 = st.checkbox('Matplotlib')
    DataViz_2 = st.checkbox('Seaborn')
    DataViz_3 = st.checkbox('Plotly')
    DataViz_4 = st.checkbox('Ggplot')
    DataViz_5 = st.checkbox('Shiny')
    DataViz_6 = st.checkbox('D3 js')
    DataViz_7 = st.checkbox('Altair')
    DataViz_8 = st.checkbox('Bokeh')
    DataViz_9 = st.checkbox('Geoplotlib')
    DataViz_10 = st.checkbox('Leaflet/Folium')
    DataViz_11 = st.checkbox('None of the above DataViz tools')
    DataViz_12 = st.checkbox('Other tools')
    st.write('**What machine learning frameworks are you familiar with? Select all that apply.**')
    ML_Framework_1 = st.checkbox('Scikit-learn') 
    ML_Framework_2 = st.checkbox('TensorFlow')
    ML_Framework_3 = st.checkbox('Keras')
    ML_Framework_4 = st.checkbox('PyTorch')
    ML_Framework_5 = st.checkbox('Fast.ai')
    ML_Framework_6 = st.checkbox('MXNet')
    ML_Framework_7 = st.checkbox('Xgboost')
    ML_Framework_8 = st.checkbox('LightGBM')
    ML_Framework_9 = st.checkbox('CatBoost')
    ML_Framework_10 = st.checkbox('Prophet')
    ML_Framework_11 = st.checkbox('H2O 3')
    ML_Framework_12 = st.checkbox('Caret')
    ML_Framework_13 = st.checkbox('Tidymodels')
    ML_Framework_14 = st.checkbox('JAX')
    ML_Framework_15 = st.checkbox('None of the above frameworks')
    ML_Framework_16 = st.checkbox('Other machine learning frameworks')
    st.write('**What machine learning algorithms are you familiar with? Select all that apply.**')
    ML_Algorithm_1 = st.checkbox('Linear/Logistic Regression')
    ML_Algorithm_2 = st.checkbox('Decision Trees or Random Forests')
    ML_Algorithm_3 = st.checkbox('Gradient Boosting Machines')
    ML_Algorithm_4 = st.checkbox('Bayesian Approaches')
    ML_Algorithm_5 = st.checkbox('Evolutionary	Approaches')
    ML_Algorithm_6 = st.checkbox('Dense Neural Networks')
    ML_Algorithm_7 = st.checkbox('Convolutional Neural Networks')
    ML_Algorithm_8 = st.checkbox('Generative Adversarial Networks')
    ML_Algorithm_9 = st.checkbox('Recurrent Neural Networks')
    ML_Algorithm_10 = st.checkbox('Transformer Networks')
    ML_Algorithm_11 = st.checkbox('None of the above algorithms')
    ML_Algorithm_12 = st.checkbox('Other algorithms')
   
    k_age = get_value(Age, age_label)
    k_education = get_value(Ed, education_label)
    k_codingexp = get_value(Codingyears, codingexp_label)

    # code for prediction
    jobprediction = ''

    # creating a button for prediction
    if st.button('Get Data job prediction'):
        jobprediction = datajob_prediction([k_age, k_education, k_codingexp, 
                                            Language_1, Language_2, Language_3, Language_4, Language_5, Language_6, Language_7, Language_8, Language_9, Language_10, Language_11, Language_12, Language_13,
                                            IDEs_1, IDEs_2, IDEs_3, IDEs_4, IDEs_5, IDEs_6, IDEs_7, IDEs_8, IDEs_9, IDEs_10, IDEs_11, IDEs_12, 
                                            DataViz_1, DataViz_2, DataViz_3, DataViz_4, DataViz_5, DataViz_6, DataViz_7, DataViz_8, DataViz_9, DataViz_10, DataViz_11, DataViz_12,
                                            ML_Framework_1, ML_Framework_2, ML_Framework_3, ML_Framework_4, ML_Framework_5, ML_Framework_6, ML_Framework_7, ML_Framework_8, ML_Framework_9, ML_Framework_10, ML_Framework_11, ML_Framework_12, ML_Framework_13, ML_Framework_14, ML_Framework_15, ML_Framework_16,
                                            ML_Algorithm_1, ML_Algorithm_2, ML_Algorithm_3, ML_Algorithm_4, ML_Algorithm_5, ML_Algorithm_6, ML_Algorithm_7, ML_Algorithm_8, ML_Algorithm_9, ML_Algorithm_10, ML_Algorithm_11, ML_Algorithm_12])

    st.success(jobprediction)  
    
if page == pages[4]:
    st.header('Conclusions')
    st.markdown('Our goal was to build a multiclass classification model to predict a specific data job.')
    st.markdown('No model performed satisfactory, but we still build a sample recommendation system from the best performing model.')
    st.markdown('The main difficulty during the project was the overfitting of the models and a lot of time was invested in an attempt to correct this issue.')
    st.markdown('Due to the limited time, not all possible approaches were tried out, so there are still perspectives for the future to improve the models.')
    st.header('Perspectives')
    st.markdown('Our next approach to improve our models are -  ')
    st.markdown('(1) Feature selection - Diverse Feature selection methods will be tested to select minimum number of feature variables from our dataset to build a better model')
    st.markdown('(2) Feature Engineering - Creating a new feature (scores) from the existing features')
    st.markdown('(3) Use different methods for dimensionality reduction like t-SNE algorithm or linear discriminant analysis (LDA)')
    st.markdown('(4) Experiment more with hyperparameter tuning, especially for the XGBoost model')
    st.markdown('(5) The most important features and their influence on the model prediction will be determined using SHAP and ' )
    st.markdown('(6) Deep learning neural networks which support multi-label classification problems will be evaluated using the Keras deep learning library.')