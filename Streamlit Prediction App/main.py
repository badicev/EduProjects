import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle



def collect_data(keys):
    d = []
    for key in keys:
        value = st.session_state[key]
        d.append(value)
    return d


data = sns.load_dataset('iris')


labels = data['species'].unique()
#print(labels)

X = data.drop('species', axis=1)
X.dropna(inplace=True)
keys = X.keys()
#print(keys)

bounds = {}

for key in keys:
    bounds[key]=[]
    bounds[key].append(float(X[key].min()))
    bounds[key].append(float(X[key].max()))
    
#print(bounds)


filename_iris_pca = r'C:\Users\basak\OneDrive\Masaüstü\DeepDreamProject\PersonalProjects\Streamlit Prediction App\models\pca_iris_model.sav'

filename_iris = r'C:\Users\basak\OneDrive\Masaüstü\DeepDreamProject\PersonalProjects\Streamlit Prediction App\models\iris_model.sav'

# pca = PCA(n_components=2) #with 2 principal components
pca = pickle.load(open(filename_iris_pca, 'rb'))
pca.fit(X)
pca_transformed = pca.transform(X)

X["PCA1"] = pca_transformed[:,0]
X["PCA2"] = pca_transformed[:,1]

# model = KMeans(n_clusters=3, random_state=0)
model = pickle.load(open(filename_iris, 'rb'))
model.fit(pca_transformed)
y_pred = model.predict(pca_transformed)



#pickle.dump(model, open(filename, 'wb'))



#pickle.dump(pca, open(filename, 'wb'))



select_box = st.sidebar.selectbox("Which species do you want to predict?", labels)

for i, key in enumerate(keys):
    st.sidebar.slider(label = key, min_value = bounds[key][0], max_value = bounds[key][1],
                      value = (bounds[key][0]+bounds[key][1])/2, step=0.1, key=key)    
    
submit = st.sidebar.button("Predict")
    
    
if submit:
    preds = collect_data(keys)
    data_test = pd.DataFrame(columns=preds)
    data_test = data_test.append(pd.Series(data_test.columns, index=data_test.columns), ignore_index=True)
    
    test_transformed = pca.transform(data_test)
    data_test["PCA1"] = test_transformed[:,0]
    data_test["PCA2"] = test_transformed[:,1]
    
    prediction = model.predict(test_transformed)
    
    
    if prediction[0] == 0:
        st.subheader("The predicted species is: Setosa")
    elif prediction[0] == 1:
        st.subheader("The predicted species is: Versicolor")
    else:
        st.subheader("The predicted species is: Virginica")
        
  
    if select_box == labels[prediction[0]]:
        st.subheader("You are correct!")
        st.balloons()
    else:
        st.subheader("You are wrong!")
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(X["PCA1"], X["PCA2"], c=y_pred, s=50, cmap='viridis')
    plt.scatter(data_test["PCA1"], data_test["PCA2"], c="red", s=150, alpha=0.5)
    st.write(fig)

            

#st.write("The PCA transformed data is shown below:")
#st.write(X)
