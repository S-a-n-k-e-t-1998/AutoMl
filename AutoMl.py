
class AutoMl():
    def __init__(self,df,target,Estimator=None,test_size=0.2,strategy='mean',std_scale=True,model_train=True,type_model='Classification'):
        """"
        1. df   ::  Data with target column
        2. target ::  target Column Name
        3. Estimator :None :: By default None, here pass the model instance
        4. test_size :0.2 ::Default test size 0.2. Change test size using this parameter
        5. strategy :mean ::Deafault strategy for numeric features `mean` ['mean','median']
                   and use categorical features `mode'
        
        6. std_scale : True :: True >> standard Scalar used for scaling
                            False >> Min-Max Scalar used for scaling
        7. model_train: True ::True >> Model training True return the training result of model
                            False >> Return the data after preprocessing array
                              >> x_train,x_test,y_train,y_test
        8. type_model:'Classification' :: select type of model `Regression` or 'Classification`
        -----------------------------------------------------------------------------------------------
        return ::
            1. train_test_data_split >> split the data into x_train,x_test,y_train,y_test
            2. create_model >> after used all preprocessing and model return
    
        Copyright (c) 2022 Sanket Suresh Bodake
        """
        import warnings
        warnings.filterwarnings('ignore')
        self.df=df.drop(target,axis=1)
        self.target=df[target]
        self.test_size=test_size
        self.strategy=strategy
        self.scale=std_scale
        self.model_train=model_train
        self.Estimator=Estimator
        self.type_model=type_model
    def train_test_data_spilt(self):
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(self.df,self.target,test_size=self.test_size,random_state=10)
        return x_train,x_test,y_train,y_test
    def create_model(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler,MinMaxScaler
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        x_train,x_test,y_train,y_test=self.train_test_data_spilt()
        numeric_feature=list(self.df.select_dtypes(include=['int64','float64']).columns)
#         numeric_feature=["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
        cat_feature=list(self.df.select_dtypes(include=['object']).columns)
#         print(cat_feature)
        std_scalar_pipeline=Pipeline(steps=[('missingvaluehandling',SimpleImputer(strategy=self.strategy)),
                                               ('std_scaler',StandardScaler(with_mean=True))
                   ])
        min_max_scalar_pipeline=Pipeline(steps=[('missingvaluehandling',SimpleImputer(strategy=self.strategy)),
                                                   ('std_scaler',MinMaxScaler())
                   ])   
        categorical_pipeline=Pipeline(steps=[('catmissingvaluehandling',SimpleImputer(strategy="most_frequent")),
                                               ('OneHotencoding', OneHotEncoder(handle_unknown="ignore"))
                   ])  

        def model_evalution(model,X,y_true):
            y_pred=model.predict(X)
            from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
            acc_score=accuracy_score(y_true,y_pred)
            print(f"Accuracy Score of Model ::{acc_score}")
            conf_matrix=confusion_matrix(y_true,y_pred)
            print(f"confusion matrix  of Model ::\n{conf_matrix}")
            class_matrix=classification_report(y_true,y_pred)
            print(f"classification report of Model ::\n{class_matrix}") 
        def model_evalution1(model,X,y_true):
                from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
                import numpy as np
                y_pred=model.predict(X)
                mse=mean_squared_error(y_true,y_pred)
                print(f"The MSE of Model ::{mse}")
                mae=mean_absolute_error(y_true,y_pred)
                print(f"The MAE of Model ::{mae}")
                rsme=np.sqrt(mse)
                print(f"The RMSE of Model ::{rsme}")
                r2_value=r2_score(y_true,y_pred)
                print(f"The R2 of Model ::{r2_value}")
                
        if self.scale==True:
            print("Standard Scalar ...")
            preprocessor=ColumnTransformer(transformers=[('numeric_value',std_scalar_pipeline,numeric_feature),
                                    ("cat", categorical_pipeline, cat_feature)])
            X_train=preprocessor.fit_transform(x_train)
            X_test=preprocessor.transform(x_test)
        else:
            print("Min-Max Scaler ....")
            preprocessor=ColumnTransformer(transformers=[('numeric_value',min_max_scalar_pipeline,numeric_feature),
                                    ("cat", categorical_pipeline, cat_feature)])
            X_train=preprocessor.fit_transform(x_train)
            X_test=preprocessor.transform(x_test) 
        if self.model_train==True:
            from sklearn import set_config
            model_pipe=Pipeline(steps=[('preprcess',preprocessor),
                                  ('model',self.Estimator)])
            model_pipe.fit(x_train,y_train)
            if self.type_model=='Classification':
                print()
                set_config(display="diagram")
                print('*'*20+'Training Data Evalution','*'*20)
                model_evalution(model_pipe,x_train,y_train)
                print('*'*20+'Testing Data Evalution','*'*20)
                model_evalution(model_pipe,x_test,y_test) 
                return model_pipe
            else:
                print()
                set_config(display="diagram")
                print('*'*20+'Training Data Evalution','*'*20)
                model_evalution1(model_pipe,x_train,y_train)
                print('*'*20+'Testing Data Evalution','*'*20)
                model_evalution1(model_pipe,x_test,y_test)
                return model_pipe
        else:
            return X_train,X_test,y_train,y_test

# Dataset1

import pandas as pd
# df1=pd.read_csv(r"F:\New folder\salary.csv")
# df1['salary']=df1['salary'].replace({' <=50K': 0, ' >50K': 1})
# from sklearn.linear_model import LogisticRegression
# lin_model=LogisticRegression()
# ob=AutoMl(df1,'salary',Estimator=lin_model,type_model='Classification')
# ob.create_model()

# Dataset2
df1=pd.read_csv(r"F:\New folder\Pune_rent.csv")
df1['bedroom']=df1['bedroom'].astype('float64')
df1['price']=df1['price'].apply(lambda x:x.replace(',','')).astype('float64')
df1['area']=df1['area'].astype('float64')
from sklearn.linear_model import LinearRegression
lin_model=LinearRegression()
ob=AutoMl(df1,'price',Estimator=lin_model,type_model='Regression')
ob.create_model()
