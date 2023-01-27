''' 1)Latent Dirichlet Allocation (LDA)
  -Bayesian model("Bayes net")
  2)Non-Negative Matrix Factoization(NMF)
  -Originates from recommender systems '''
  
  #No labels required
  #"Latent Dirichlet Allocation" by David Blei, Andrew Ng, Micheal Jordan
  
  #x=CountVectorizer().fit_transform(text)
  
  #lda = LatentDirichletAllocation()
  #lda.fit(x) #not fit(X, Y)
  #z = lda.transfrom(X) -> Returns docs x topics matrix
  
  #convention: X = observed data, Z = unobserved variables
  
  #supervised: Yhat = model.predict(X)
  #unsupervised: Z = model.transform(X)
  #both are called "inference"
  
  #z = lda.fit_transform(X)
  
  #where is the topics x words matrix?
  #topics = lda.components_
  
  #the topics are "internal" to the model, documents are not
  #Z_test = lda.transform(X_test)
  
'''Topics are distributions over words.
    Documents are distribution over topics'''
  
#in graphical model notation dark means it is variable we observed, light means it is a variable we didn't observe.
#posterior distribution

'''Bayes Classifier -> p(y|x) = p(x|y)p(y)/p(x)
   Clustering/Mixture Model -> p(z|x) = p(x|z)p(z)/p(x)
'''
#In both cases, we use Bayes' rule to find the posterior. (expectation-maximization)

#Bayesian ML: everything is random - all variables have probability distributions, and distributions have parameters

#we won't be using labels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import textwrap
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download("stopwords")

stops = set(stopwords.words("english")) #make a set of stopwords

stops = stops.union({"said", "also", "one", "us", 
                     "would", "new", "would", 
                     "one", "could", "last", "two",
                     "best", "like", "think"}) #add these words to the set

df =pd.read_csv(r"NLP\data\bbc_text_cls.csv")

df.head()

vectorizer = CountVectorizer(stop_words=stops) #remove stopwords

X = vectorizer.fit_transform(df.text) #fit and transform

lda = LatentDirichletAllocation(n_components=10,#default is 10
                                random_state=44) #10 topics

lda.fit(X) #fit the model

def plot_top_words(model, feature_names, n_top_words, title): # from https://scikit-learn.ru/example/topic-extraction-with-non-negative-matrix-factorization-and-latent-dirichlet-allocation/
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
    
    
feature_names = vectorizer.get_feature_names_out()
plot_top_words(lda, feature_names, 10, 'Topics in LDA model')


Z = lda.transform(X) #transform the data

#Pick a document
#Check which topics it is most likely to belong to

np.random.seed(44)
i = np.random.choice(len(df))
z = Z[i]
topics = np.arange(10) + 1

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.barh(topics, z)
ax.set_yticks(topics)
ax.set_title("True label: %s" % df.iloc[i]["labels"])


print(textwrap.fill(df.iloc[i]["text"], 80))