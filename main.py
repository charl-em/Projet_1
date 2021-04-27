import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
mpl.rcParams.update(mpl.rcParamsDefault)

st.set_page_config(
    page_title="Performances des éléves",
    # page_icon="🧊",
    layout="wide",
    # initial_sidebar_state="expanded",
 )

 #Titre en grand au début
st.write(
    """
    # Bienvenue sur FactorAnalyser 
    ## Il met en évidence l'importance de facteurs socioéconomiques dans la performance à un test en maths , écriture et lecture noté sur 100
    """
)

st.subheader("")
st.subheader("Graphe montrant l'importance de différents facteurs     socioéconomiques")

st.sidebar.header("Upload your CSV data ")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)


    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)


    string_data = stringio.read()
    st.write(string_data)

     # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)


# Code machine learning
df = pd.read_csv('StudentsPerformance.csv') #Reading the data

total_marks = ((df['math score'] + df['reading score'] + df['writing score'])/300)*100 #total marks are score of all subjects out of 100
df['total_marks'] = total_marks
kde_df = df[['math score','reading score','writing score','total_marks']]
df_model = df.drop(['math score','reading score','writing score'],axis=1)
y = df_model['total_marks']
df_model = df_model.drop('total_marks',axis=1)
df_model = pd.get_dummies(df_model)
x_train,y_train,x_test,y_test = train_test_split(df_model,y,test_size=0.2,random_state=42)
model = Ridge()
model.fit(x_train,x_test)
pred = model.predict(y_train)
train_pred = model.predict(x_train)
score =  mean_squared_error(y_test,pred,squared=False)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train,x_test)
feature_importance = np.array(model.feature_importances_)
feature_names = np.array(x_train.columns)
data={'feature_names':feature_names,'feature_importance':feature_importance}
df_plt = pd.DataFrame(data)
df_plt.sort_values(by=['feature_importance'], ascending=False,inplace=True)
plt.figure(figsize=(10,8))
sns.barplot(x=df_plt['feature_importance'], y=df_plt['feature_names'])
plt.xlabel('Importance des facteurs')
plt.ylabel('Noms des facteurs')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()


# source = pd.DataFrame(
#     {'noms des facteurs': np.array([c for c in df_model if c.isalpha()])}
# )

# alt.Chart(source).transform_aggregate(
#     count='count()',
#     groupby=['noms des facteurs']
# ).transform_window(
#     rank='rank(count)',
#     sort=[alt.SortField('importance des facteurs', order='descending')]
# ).transform_filter(
#     alt.datum.rank < 18
# ).mark_bar().encode(
#     y=alt.Y('noms des facteurs:N', sort='-x'),
#     x='count:Q',
# )
