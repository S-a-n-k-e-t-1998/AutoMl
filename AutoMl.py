class AutoMl():
    def __init__(self,df,target,Estimator=None,test_size=0.2,strategy='mean',std_scale=True,model_train=True,type_model='Classification'):
        """"
          df   ::  Data with target column
        target ::  target Column Name
        Estimator :None :: By default None, here pass the model instance
        test_size :0.2 ::Default test size 0.2. Change test size using this parameter
        strategy :mean ::Deafault strategy for numeric features `mean` ['mean','median']
                   and use categorical features `mode'
        
        std_scale : True :: True >> standard Scalar used for scaling
                            False >> Min-Max Scalar used for scaling
        model_train: True ::True >> Model training True return the training result of model
                            False >> Return the data after preprocessing array
                              >> x_train,x_test,y_train,y_test
        type_model:'Classification' :: select type of model `Regression` or 'Classification`
        -----------------------------------------------------------------------------------------------
        return ::
            1. train_test_data_split >> split the data into x_train,x_test,y_train,y_test
            2. create_model >> after used all preprocessing and model return
    
        Copyright (c) 2022 Sanket Suresh Bodake
        """
        from sklearn.model_selection import train_test_split
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
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
        x_train,x_test,y_train,y_test=train_test_split(self.df,self.target,test_size=self.test_size,random_state=10)
        return x_train,x_test,y_train,y_test
    def create_model(self):
        from sklearn.preprocessing import StandardScaler,MinMaxScaler
        from sklearn.preprocessing import OneHotEncoder
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
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")
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
                                    ("cat", categorical_transformer, cat_feature)])
            X_train=preprocessor.fit_transform(x_train)
            X_test=preprocessor.transform(x_test)
        else:
            print("Min-Max Scaler ....")
            preprocessor=ColumnTransformer(transformers=[('numeric_value',min_max_scalar_pipeline,numeric_feature),
                                    ("cat", categorical_transformer, cat_feature)])
            X_train=preprocessor.fit_transform(x_train)
            X_test=preprocessor.transform(x_test) 
        if self.model_train==True:
            from sklearn import set_config
            model_pipe=Pipeline(steps=[('preprcess',preprocessor),
                                  ('log_model',self.Estimator)])
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
        else:
            return X_train,X_test,y_train,y_test


ob=AutoMl()
