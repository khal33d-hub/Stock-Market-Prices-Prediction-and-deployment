import streamlit as st
import numpy as np
import pandas as pd
from pandas import read_csv
from PIL import Image


from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error


# image = Image.open('GRBT.png')
# st.image(image, caption='Gradient Boost Logo')

st.image('GRBT.PNG')

st.title("STOCK MARKET PREDITION PROJECT")
st.write("""Explore diffrent stocks Open and Closing prices prediction for free! :+1:""")

dataset = st.sidebar.selectbox("Select stock", ("NIO", "GOOGLE", "APPLE", "TESLA", "JUMIA"))

Model = st.sidebar.selectbox("Select Regresion Model", ("Linear", "Ridge", "Lasso"))

#Score = st.sidebar.selectbox("Select Evaluation Method", ("Model Score", "RMSE", "MAE"))

def get_dataset(dataset):
    if dataset == "NIO":
        url = 'https://drive.google.com/file/d/1kMYMOpsg4kId2FIR22KsVd_mOhgtHfDr/view?usp=sharing'
        path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
        data = pd.read_csv(path, index_col=0)

    elif dataset == "GOOGLE":
        url = "https://drive.google.com/file/d/1HUjYavHPdUllwRwUBkqF2civn5Khr5bk/view?usp=sharing"
        path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
        data = pd.read_csv(path, index_col=0)

    elif dataset == "APPLE":
        url = "https://drive.google.com/file/d/1ea719W-k8TIHxYAraNeH-vtxUOn8IEI0/view?usp=sharing"
        path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
        data = pd.read_csv(path, index_col=0)

    elif dataset == "TESLA":
        url = "https://drive.google.com/file/d/1vCROHVMuYu3PeKk65GvljfQk-GfIP5aF/view?usp=sharing"
        path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
        data = pd.read_csv(path, index_col=0)

    else:
        url = "https://drive.google.com/file/d/1OchvWibWEu0aemA32POBeOCTcuBR9CWI/view?usp=sharing"
        path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
        data = pd.read_csv(path, index_col=0)

    X = data.loc[:, ~data.columns.isin(['Close', 'timestamp'])]
    y = data['Close']
    return X, y

X, y = get_dataset(dataset)
st.title(dataset)
st.write("Here are the the independent columns", X.head())
st.write(X.shape)
st.write("Here is the dependent columns", y.head())
st.write(len(y))

def hyperparams(mdl_name):
    params = dict()
    if mdl_name == "Ridge":
        R = st.sidebar.slider("R", min_value=0.0, max_value=1.0, step=0.1)
        params["R"] = R
    elif mdl_name == "Lasso":
        L = st.sidebar.slider("L", min_value=0.0, max_value=1.0, step=0.1)
        params["L"] = L
    else:
        pass
    return params

params = hyperparams(Model)


def get_model(mdl_name, params):
    if mdl_name == "Ridge":
        mdl = Ridge(normalize=True, alpha=params["R"])
    elif mdl_name == "Lasso":
        mdl = Lasso(normalize=True, alpha=params["L"])
    else:
        mdl = LinearRegression()
    return mdl

mdl = get_model(Model, params)

# the dataframe is splitted into two in order to use some for training and the other for testing what we trained
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
mdl.fit(X_train,y_train)
predict = mdl.predict(X_test)
Model_Score=mdl.score(X_test, y_test)
RMSE=np.sqrt(mean_squared_error(y_test,predict))
MAE=mean_absolute_error(y_test, predict)

#compare out tests
test_list=[Model_Score, RMSE, MAE]
best_test = sorted(test_list)[0]
test_dict={"Model_Score":Model_Score, "RMSE":RMSE, "MAE":MAE}

st.header("Evaluations")
st.subheader("Model Score")
st.markdown("**model score** works by comparing our predicted output in y_test against the actual output in y_test and compute the diffrence as a **100% score** and we got a 99%")
st.write("Here is the model score",Model_Score)

st.subheader("RMSE")
st.markdown("In order to understand how well our model works we are going to use a metric known as **RMSE**, in full this refers to the **Root Mean Square Error**. The RMSE will give us what is equivalent to the standard deviation of the unexplained variance by the model, measuring how concentrated the data is to our regression line.** The lower the value of the RMSE, the better the fit.**")
st.write("Here is the model RMSE", RMSE)

st.subheader("MAE")
st.markdown("With the **mean absolute error** we are looking to understand the average error between our predictions and observed house prices. Meaning that if we add up all the errors between our predictions and actual prices we will get an average error of about **$0.17 which is a good score**")
st.write("Here is the model MAE",MAE)
# st.markdown("")
# def evaluation(method, mdl):
#     if method == "Model Score":

st.header("conclusion")
st.write("this is our best score for this model is", best_test)
