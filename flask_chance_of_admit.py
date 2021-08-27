import pickle
import numpy   as np
import pandas  as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import OneHotEncoder
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import RobustScaler
from sklearn.compose         import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing   import PolynomialFeatures

def get_pipeline():
    ##Load data

    df = pd.read_csv('Admission_Predict_Ver1.1.csv')

    ##Data wrangling

    df.drop('Serial No.',axis=1,inplace = True)

    ###Data sampling

    df["Research"] = df["Research"].astype('category')

    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
        
    X_train= train_set.drop("Chance of Admit " , axis=1) # drop labels for training set
    y_train = train_set["Chance of Admit "].copy()
    X_test = test_set.drop('Chance of Admit ',axis = 1)
    y_test = test_set['Chance of Admit '].copy()    

    ##Pipepline

    numerical = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA']
    categorical = ['Research']

    num_pipeline = Pipeline([
            ('robust_scaler', RobustScaler()),
        ])
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))   
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, numerical),
        ("cat", cat_pipeline, categorical),
    ])
    
    X_train_pipelined = full_pipeline.fit_transform(X_train)

    return full_pipeline

def predict_newsample(new_sample):

    new_sample["Research"] = new_sample["Research"].astype('category')

    full_pipeline = get_pipeline()

    new_sample_tr = full_pipeline.transform(new_sample)

    ##load model and predict new sample
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict(new_sample_tr)

    return result