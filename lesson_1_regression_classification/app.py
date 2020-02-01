import streamlit as st
from fastai.tabular import load_learner, Path
import pandas as pd
import pickle

st.title("Are you gonna die ?")

dtypes = pickle.load(open("dtypes.pkl","rb"))

dtypes = dtypes.drop(index="Survived")
data = {}

for col,dtype in dtypes.iteritems():
    if isinstance(dtype,pd.core.dtypes.dtypes.CategoricalDtype):
        data[col] = st.selectbox(col,dtype.categories.values)
    else:
        data[col] = st.number_input(col)

path = Path()
learn = load_learner(path=path)

row = pd.Series(data)
prediction = learn.predict(row)
label = prediction[1].item()
prob = prediction[2].max().item()

if label == 1:
    st.write("Congrats you are a survivor !")
else:
    st.write("Hold me Jack !")
st.write(f"This diagnosis had a chance of {prob} to happen")