#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import seaborn as sns
import numpy as np
import pandas as pd
import string
import re
import matplotlib.pyplot as plt
import logging
from nltk.corpus import stopwords


# In[ ]:


df = pd.read_csv('Twitter_Dataset.csv', encoding= 'latin1')
df


# In[ ]:


import re
import emoji

# Function to calculate the word count
def word_count(text):
    words = text.split()
    return len(words)

# Function to calculate the character count
def character_count(text):
    return len(text)

# Function to calculate the average word length
def average_word_length(text):
    words = text.split()
    word_lengths = [len(word) for word in words]
    return sum(word_lengths) / len(words) if len(words) > 0 else 0

# Function to calculate the sentence count
def sentence_count(text):
    sentences = re.split(r'[.!?]+', text)
    return len(sentences)

# Function to calculate the hashtag count
def hashtag_count(text):
    hashtags = re.findall(r"#\w+", text)
    return len(hashtags)

# Function to calculate the mention count
def mention_count(text):
    mentions = re.findall(r"@\w+", text)
    return len(mentions)

# Function to calculate the emoji count
def emoji_count(text):
    text = emoji.demojize(text)
    emojis = re.findall(r":[^:\s]+:", text)
    return len(emojis)

# Function to calculate the URL count
def url_count(text):
    urls = re.findall(r"http\S+|www\S+", text)
    return len(urls)

# Calculate the statistical features for all tweets in the dataset
sum_word_count = 0
sum_character_count = 0
sum_avg_word_length = 0
sum_sentence_count = 0
sum_hashtag_count = 0
sum_mention_count = 0
sum_emoji_count = 0
sum_url_count = 0

for index, row in df.iterrows():
    tweet = row['tweet']
    sum_word_count += word_count(tweet)
    sum_character_count += character_count(tweet)
    sum_avg_word_length += average_word_length(tweet)
    sum_sentence_count += sentence_count(tweet)
    sum_hashtag_count += hashtag_count(tweet)
    sum_mention_count += mention_count(tweet)
    sum_emoji_count += emoji_count(tweet)
    sum_url_count += url_count(tweet)

# Print the statistical features for the whole dataset
print("Statistical Features Before Cleaning the Dataset")
print("Total Word Count:", sum_word_count)
print("Total Character Count:", sum_character_count)
print("Average Word Length:", sum_avg_word_length / len(df))
print("Total Sentence Count:", sum_sentence_count)
print("Total Hashtag Count:", sum_hashtag_count)
print("Total Mention Count:", sum_mention_count)
print("Total Emoji Count:", sum_emoji_count)
print("Total URL Count:", sum_url_count)


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


print(df["Class"].value_counts())
print("total = ",df["Class"].value_counts().sum())


# In[ ]:


my_class = ['neu', 'neg', 'pos']
plt.figure(figsize=(10,4))
df["Class"].value_counts().plot(kind='bar');


# In[ ]:


def print_row(index):
    row = df[df.index == index][['tweet', 'Class']].values[0]
    if len(row) > 0:
        print(row[0])
        print('Class =', row[1])
        print()


# In[ ]:


#before cleaning
for i in range(0,10):
   print_row (i)


# In[ ]:


replaceBySpace = re.compile('[/(){}\[\]\|@,.;#]')
badSymbols = re.compile(r"http\S+|www\S+|[^0-9a-z #+_?]+")

def clean(text):
    
    text = text.lower() #lower text
    text = replaceBySpace.sub(' ',text) # replace with space which symbols are in replaceBySpace
    text = badSymbols.sub('', text) # delete symbols which are in badSymbols
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text    


# In[ ]:


alpha = list(string.ascii_lowercase)
stop_words = []
for i in range(len(alpha)):
    stop_words.append(alpha[i])
    for j in range(len(alpha)):
        stop_words.append(alpha[i]+alpha[j])


# In[ ]:


df["tweet"]= df["tweet"].apply(str)


# In[ ]:


df["tweet"] = df["tweet"].apply(clean)


# In[ ]:


from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stop_words = list(set(stopwords.words('english')))
lemma = WordNetLemmatizer()


# In[ ]:


Class = df['Class']


# In[ ]:


lower = [clean(i).lower() for i in df]


# In[ ]:


list1 = []
for i in lower:
    words = word_tokenize(i)
    list1.append(words)


# In[ ]:


list2 = []
for i in list1:
    l = []
    for j in i:
        if j not in stop_words:
            lem = lemma.lemmatize(j)
            l.append(lem)
    y = ' '.join(l)
    list2.append(y)


# In[ ]:


total_words = df['tweet'].apply(lambda x: len(x.split(' '))).sum()
print(total_words)


# In[ ]:


#before cleaning
for i in range(0,10):
   print_row (i)


# In[ ]:


import re
import emoji

# Function to calculate the word count
def word_count(text):
    words = text.split()
    return len(words)

# Function to calculate the character count
def character_count(text):
    return len(text)

# Function to calculate the average word length
def average_word_length(text):
    words = text.split()
    word_lengths = [len(word) for word in words]
    return sum(word_lengths) / len(words) if len(words) > 0 else 0

# Function to calculate the sentence count
def sentence_count(text):
    sentences = re.split(r'[.!?]+', text)
    return len(sentences)

# Function to calculate the hashtag count
def hashtag_count(text):
    hashtags = re.findall(r"#\w+", text)
    return len(hashtags)

# Function to calculate the mention count
def mention_count(text):
    mentions = re.findall(r"@\w+", text)
    return len(mentions)

# Function to calculate the emoji count
def emoji_count(text):
    text = emoji.demojize(text)
    emojis = re.findall(r":[^:\s]+:", text)
    return len(emojis)

# Function to calculate the URL count
def url_count(text):
    urls = re.findall(r"http\S+|www\S+", text)
    return len(urls)

# Calculate the statistical features for all tweets in the dataset
sum_word_count = 0
sum_character_count = 0
sum_avg_word_length = 0
sum_sentence_count = 0
sum_hashtag_count = 0
sum_mention_count = 0
sum_emoji_count = 0
sum_url_count = 0

for index, row in df.iterrows():
    tweet = row['tweet']
    sum_word_count += word_count(tweet)
    sum_character_count += character_count(tweet)
    sum_avg_word_length += average_word_length(tweet)
    sum_sentence_count += sentence_count(tweet)
    sum_hashtag_count += hashtag_count(tweet)
    sum_mention_count += mention_count(tweet)
    sum_emoji_count += emoji_count(tweet)
    sum_url_count += url_count(tweet)

# Print the statistical features for the whole dataset
print("Statistical Features After Cleaning the Dataset")
print("Total Word Count:", sum_word_count)
print("Total Character Count:", sum_character_count)
print("Average Word Length:", sum_avg_word_length / len(df))
print("Total Sentence Count:", sum_sentence_count)
print("Total Hashtag Count:", sum_hashtag_count)
print("Total Mention Count:", sum_mention_count)
print("Total Emoji Count:", sum_emoji_count)
print("Total URL Count:", sum_url_count)


# In[ ]:


X = df["tweet"]
y = df["Class"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 42)


# In[ ]:


X_train


# In[ ]:


len(X_train)


# In[ ]:


X_test


# In[ ]:


len(X_test)


# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.metrics import classification_report\nfrom sklearn.metrics import accuracy_score, confusion_matrix\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.feature_extraction.text import TfidfTransformer\nfrom sklearn.feature_extraction.text import TfidfVectorizer\n    \nmodel_list = [ Pipeline([\n    ('vect', TfidfVectorizer()),\n    ('clf', MLPClassifier(hidden_layer_sizes=(47), max_iter=100, random_state=42)),\n]),\n             Pipeline([\n    ('vect', TfidfVectorizer()),\n    ('clf', MLPClassifier(hidden_layer_sizes=(47), max_iter=100, random_state=42)),\n]),\n              Pipeline([\n    ('vect', TfidfVectorizer()),\n    ('clf', MLPClassifier(hidden_layer_sizes=(47), max_iter=100, random_state=42)),\n]),\n              Pipeline([\n    ('vect', TfidfVectorizer()),\n    ('clf', MLPClassifier(hidden_layer_sizes=(47), max_iter=100, random_state=42)),\n]),\n             Pipeline([\n    ('vect', TfidfVectorizer()),\n    ('clf', MLPClassifier(hidden_layer_sizes=(47), max_iter=100, random_state=42)),\n]),\n              Pipeline([\n    ('vect', TfidfVectorizer()),\n    ('clf', MLPClassifier(hidden_layer_sizes=(47), max_iter=100, random_state=42)),\n]),\n              Pipeline([\n    ('vect', TfidfVectorizer()),\n    ('clf', MLPClassifier(hidden_layer_sizes=(47), max_iter=100, random_state=42)),\n]),\n              Pipeline([\n    ('vect', TfidfVectorizer()),\n    ('clf', MLPClassifier(hidden_layer_sizes=(47), max_iter=100, random_state=42)),\n]),\n              Pipeline([\n    ('vect', TfidfVectorizer()),\n    ('clf', MLPClassifier(hidden_layer_sizes=(45), max_iter=100, random_state=42)),\n]),\n              Pipeline([\n    ('vect', TfidfVectorizer()),\n    ('clf', MLPClassifier(hidden_layer_sizes=(45), max_iter=100, random_state=42)),\n])\n             ]")


# In[ ]:


def show_values(pc, fmt="%.2f", **kw):

    pc.update_scalarmappable()
    ax = pc.axes# FOR LATEST MATPLOTLIB
    #Use zip BELOW IN PYTHON 3
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):

    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):


    # Plot it out
    fig, ax = plt.subplots()    
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdYlGn', vmin=0.90, vmax=1.0)
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1line.set_visible = False
        t.tick2line.set_visible = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1line.set_visible = False
        t.tick2line.set_visible = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    fig.set_size_inches(cm2inch(35, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    #fig.set_size_inches(cm2inch(figure_width, figure_height))



def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):

    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : 6]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        #print(v)
        plotMat.append(v)

    #print('plotMat: {0}'.format(plotMat))
    #print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)


# In[ ]:


from sklearn.metrics import precision_score, recall_score, f1_score

mlp_log = Pipeline([
    ('vect', TfidfVectorizer()),
    ('clf', MLPClassifier(hidden_layer_sizes=(45), max_iter=100, random_state=42)),
])
mlp_log.fit(X_train, y_train)
y_pred = mlp_log.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy of MLP:", acc)
report = classification_report(y_test, y_pred, labels=my_class)
print("Test Report")
print(report)
plot_classification_report(report)

# Assuming y_true and y_pred are the true and predicted labels, respectively
average_precision = precision_score(y_test, y_pred, average='macro')
average_recall = recall_score(y_test, y_pred, average='macro')
average_f1_score = f1_score(y_test, y_pred, average='macro')

print("Average Precision:", average_precision)
print("Average Recall:", average_recall)
print("Average F1-score:", average_f1_score)


# In[ ]:


def heatmap_cf(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu', mx = 1000):



    # Plot it out
    fig, ax = plt.subplots()    
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdYlGn', vmin=0.0, vmax=mx)
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1line.set_visible = False
        t.tick2line.set_visible = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1line.set_visible = False
        t.tick2line.set_visible = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    fig.set_size_inches(cm2inch(35, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    #fig.set_size_inches(cm2inch(figure_width, figure_height))


# In[ ]:


from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_pred, labels=my_Class)
my_Class_rev = my_Class.reverse
heatmap_cf(cf_matrix, title='Confusion Matrix' , xlabel="True", ylabel="Predicted", xticklabels=my_Class, yticklabels=my_Class)


# In[ ]:


from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ =         learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


# In[ ]:


fig, axes = plt.subplots(3,figsize=(10, 15))
title = "Learning Curve (Multinomial Logistic Regression)"
cv = ShuffleSplit(n_splits=9, test_size=0.3, random_state=42)

estimator = Pipeline([('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(n_jobs=5, C=1e5, solver='lbfgs', multi_class='auto')),
           ])
plot_learning_curve(estimator, title, X, y, axes=axes[:], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=5)


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=10, shuffle=True, random_state=42)
validation_accuracy = []
validation_report = []

for k, (train, test) in enumerate(kf.split(X,y)):
    xtrain, ytrain = X[train], y[train]
    xtest, ytest = X[test], y[test]
    
    model_list[k].fit(xtrain, ytrain)
    pred = model_list[k].predict(xtest)
    accuracy = accuracy_score(ytest, pred)
    report = classification_report(ytest, pred, labels=my_Class)
    validation_accuracy.append(accuracy)
    validation_report.append(report)
    print(f"Fold #{k+1}")
    print(f"Validation Accuracy = {accuracy}")
    print("Validation Report:")
    print(report)

for i in range(9):
    if i < len(validation_report):  # Check if index is within the valid range
        plot_classification_report(validation_report[i])
    
prec_fold = []
reca_fold = []
f1_fold = []
for report in validation_report:
    lines = list(report.split('\n'))
    prec, reca, f1 = 0, 0, 0
    for i in range(2, 6):
        split_result = lines[i].split()
        if len(split_result) >= 4:  # Check if split result has at least 4 parts
            prec += float(split_result[1])
            reca += float(split_result[2])
            f1 += float(split_result[3])        
    prec_fold.append(prec / 4)
    reca_fold.append(reca / 4)
    f1_fold.append(f1 / 4)


# In[ ]:


df_n = pd.DataFrame([['01', '01', '01', '02', '02', '02', '03', '03', '03', '04', '04', '04', '05', '05', '05','06','06','06','07','07','07','08','08','08','09','09','09','10','10','10'],
                     ['Precision','Recall','F1-Score','Precision','Recall','F1-Score','Precision','Recall','F1-Score','Precision','Recall','F1-Score', 'Precision','Recall','F1-Score','Precision','Recall','F1-Score','Precision','Recall','F1-Score','Precision','Recall','F1-Score', 'Precision','Recall','F1-Score','Precision','Recall','F1-Score'], 
                     [prec_fold[i] for i in range(len(prec_fold))] + [reca_fold[i] for i in range(len(reca_fold))] + [f1_fold[i] for i in range(len(f1_fold))]]).T

df_n.columns = ['Fold', 'Type', 'Score']
df_n.set_index(['Fold', 'Type'], inplace=True)
ax.set_ylim(2,10)
df_n.unstack().plot.bar()


# In[ ]:


tweets = df['tweet']
labels = df['Class']

# Generate the IDF weights for all the words in the tweets
vectorizer = TfidfVectorizer()
tfidf_weights = vectorizer.fit_transform(tweets)
idf = vectorizer.idf_
tfidf_dict = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

# Generate the term frequency for all the words in the tweets
tf_vectorizer = CountVectorizer()
tf = tf_vectorizer.fit_transform(tweets)
word_freq = dict(zip(tf_vectorizer.get_feature_names(), np.asarray(tf.sum(axis=0)).ravel()))

# Visualize the IDF weights of the most tweeted words
sorted_idf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:20]
labels, values = zip(*sorted_idf)
plt.figure(figsize=[10, 5])
plt.bar(labels, values)
plt.xticks(rotation=90)
plt.title('IDF weights of most tweeted words')
plt.ylabel('IDF weights')
plt.show()

# Visualize the term frequency of the most tweeted words
sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
labels, values = zip(*sorted_freq)
plt.figure(figsize=[10, 5])
plt.bar(labels, values)
plt.xticks(rotation=90)
plt.title('Term frequency of most tweeted words')
plt.ylabel('Term frequency')
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Define the sentiments and their corresponding metrics
sentiments = ['Neutral', 'Negative', 'Positive']
precision = [0.97, 0.97, 0.97]  # Replace with your actual precision values
recall = [0.91, 0.92, 0.92]  # Replace with your actual recall values
f1_score = [0.88, 0.84, 0.86]  # Replace with your actual F1-score values

# Set the position of each sentiment on the x-axis
x_pos = np.arange(len(sentiments))

# Set the width of the bars
bar_width = 0.1

# Create the bar graph
plt.bar(x_pos, precision, width=bar_width, label='Precision')
plt.bar(x_pos + bar_width, recall, width=bar_width, label='Recall')
plt.bar(x_pos + (2 * bar_width), f1_score, width=bar_width, label='F1 Score')

# Set the labels for the x-axis and y-axis
plt.xlabel('Sentiments')
plt.ylabel('Scores')

# Set the title of the graph
plt.title('Metrics Comparison for Different Sentiments')

# Set the position of the x-axis ticks and label them with the sentiments
plt.xticks(x_pos + bar_width, sentiments)

# Add a legend
plt.legend()

# Show the bar graph
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Prepare your data
training_samples = ['50:50', '60:40', '70:30','80:20','90:10']
precision_values = [89, 90, 92, 92, 92]
recall_values = [87, 89, 91, 90, 90]
f1score_values = [88, 90, 91, 91,  91]

# Create the bar graph
plt.figure(figsize=(7, 5))

# Calculate the percentage values
precision_percentages = np.array(precision_values) / 100
recall_percentages = np.array(recall_values) / 100
f1score_percentages = np.array(f1score_values) / 100

# Set the x-axis positions
x_pos = np.arange(len(training_samples))

# Set the bar width
bar_width = 0.15

# Create the bars for each metric
plt.bar(x_pos, precision_percentages, width=bar_width, label='Precision')
plt.bar(x_pos + bar_width, recall_percentages, width=bar_width, label='Recall')
plt.bar(x_pos + 2 * bar_width, f1score_percentages, width=bar_width, label='F1-score')

# Set the labels for the x-axis and y-axis
plt.xlabel('Tweet Comparison')
plt.ylabel('Performance')
plt.xticks(x_pos + bar_width, training_samples)
plt.yticks(np.arange(0.1, 1.1, 0.1), ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])

# Set the title and legend
plt.title('Metrics for Different Training and Testing Samples')
plt.legend()

# Show the graph
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Model names
models = ['Proposed Model', 'Logistic Regression', 'Graph-based DL Model','BERT and SVM','Random Forest','SVM', 'Other ML Models']

# Accuracy scores
acc_scores = [95.14, 94.7, 90.3, 87.5, 80.9, 85.6, 86.4]

# Define colors for each bar
colors = ['#ff1493', '#00ced1','#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Set the x-coordinates for the bars and labels
x_pos = np.arange(len(models))
x_size = 0.9

# Create a 3D bar chart
fig = plt.figure(figsize=(10, 12))
ax = fig.add_subplot(111, projection='3d')
for i in range(len(models)):
    ax.bar3d(x_pos[i], i, 0, x_size, 0.5, acc_scores[i], color=colors[i], alpha=0.8)
    ax.text(x_pos[i]+x_size/2, i, acc_scores[i]+1, str(acc_scores[i]), ha='center', va='bottom', fontsize=10)

# Set the title and axis labels
ax.set_xlabel('Models')
ax.set_zlabel('Accuracy')

scale_factor = 1.0

# Set the x-axis limits
ax.set_xlim([-1, len(models)])
ax.set_xticks(x_pos)


# Set the y-axis limits
ax.set_ylim([-1, len(models)])
ax.set_yticks(x_pos)
ax.set_yticklabels(models)

# Set the z-axis limits
ax.set_zlim([0, max(acc_scores)+10])

# Customize the legend
handles = [plt.Rectangle((0,0),1,1, color=colors[i], alpha=0.8) for i in range(len(models))]
labels = models
ax.legend(handles, labels, loc='center')

# Rotate the chart for better view
ax.view_init(elev=7, azim=-30)

# Display the plot

plt.yticks(rotation = 50)
plt.show()


# In[ ]:


unlabel = pd.read_csv('COVdataset unlabeled.csv', encoding='latin1')
unlabel


# In[ ]:


unlabel["tweet"].apply(clean)


# In[ ]:


predicted = multinomial_log.predict(unlabel["tweet"])


# In[ ]:


predicted


# In[ ]:


unlabel["Class"] = [cls for cls in predicted]


# In[ ]:


unlabel


# In[ ]:


unlabel.to_csv("classified covid.csv",index=False)


# In[ ]:




