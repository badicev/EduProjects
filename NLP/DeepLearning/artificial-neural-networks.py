#feedforward neural networks
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


df = pd.read_csv('NLP/data/bbc_text_cls.csv')

#df.head()

#map classes to integers from 0....K-1
df["labels"].astype("category").cat.codes

df["targets"] = df["labels"].astype("category").cat.codes

df_train, df_test = train_test_split(df, test_size=0.3, random_state=4)

tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
X_train = tfidf.fit_transform(df_train["text"])
X_test = tfidf.transform(df_test["text"])

Y_train = df_train["targets"]
Y_test = df_test["targets"]

#number of classes 
K = df["targets"].max() + 1
K
#input dimensions
D = X_train.shape[1] #columns

#build the model
i = Input(shape=(D,))
x = Dense(100, activation='relu')(i)
x = Dense(100, activation='relu')(x)
x = Dense(K)(x) #softmax included in loss function

model = Model(i, x)

model.summary()

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
                metrics=['accuracy'])

#data shouldn't be a sparse matrix before passing into tensorflow
X_train = X_train.toarray()
X_test = X_test.toarray()

r = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=7, batch_size=32)



'''#plot loss per iteration
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

#plot accuracy per iteration
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()'''

print(df.labels.value_counts())
        
        

