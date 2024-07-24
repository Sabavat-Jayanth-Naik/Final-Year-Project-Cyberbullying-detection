
# README

## Cyberbullying Detection with Machine Learning and Deep Learning Models

This project focuses on detecting cyberbullying on Twitter using a combination of machine learning and deep learning techniques. The dataset contains tweets categorized into various types of cyberbullying, and the goal is to classify tweets as either "bullying" or "not bullying."

### Table of Contents
1. [Installation](#installation)
2. [Data Preprocessing](#data-preprocessing)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Modeling](#modeling)
5. [Evaluation](#evaluation)
6. [Usage](#usage)
7. [Acknowledgements](#acknowledgements)

### Installation

To get started, you need to install the necessary packages. You can do this by running the following commands:

```sh
pip install scikit-learn
pip install --upgrade scikit-learn
pip install emoji
pip install demoji
pip install nltk
pip install wordcloud
pip install keras
pip install tensorflow
```

### Data Preprocessing

1. **Loading the Dataset:**
   ```python
   df = pd.read_csv("path_to_your_dataset/cyberbullying_tweets.csv")
   ```

2. **Handling Missing Values and Duplicates:**
   - Drop rows where `cyberbullying_type` is NaN.
   - Remove any empty strings and duplicates.

3. **Renaming Columns:**
   ```python
   df = df.rename(columns={'tweet_text': 'text', 'cyberbullying_type': 'bully_type'})
   ```

4. **Text Cleaning:**
   - Convert text to lowercase.
   - Remove mentions, hashtags, URLs, and special characters.
   - Remove stop words and apply stemming.

5. **Deep Cleaning:**
   - Remove emojis.
   - Handle contractions.
   - Remove special characters and multiple spaces.
   - Apply stemming.

### Exploratory Data Analysis

Visualize the most frequent words using word clouds for different categories:

```python
from wordcloud import WordCloud

def generate_wordcloud(text, title):
    wordcloud = WordCloud(background_color='black', colormap="Dark2", collocations=False).generate(" ".join(text))
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=40)
    plt.axis('off')
    plt.show()

# Example for non-bullying tweets
generate_wordcloud(df[df['bully_type'] == 'not_cyberbullying']['text'].values, "Not Bully")
```

### Modeling

The project involves training both machine learning and deep learning models:

1. **Bidirectional LSTM (BLSTM):**
   - Architecture:
     - Embedding layer
     - Bidirectional LSTM layers
     - Dense layers with dropout
   - Compile and fit the model:
     ```python
     model5.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
     history5 = model5.fit(X_train, y_train, validation_split=0.1, epochs=40, batch_size=128)
     ```

2. **SimpleRNN:**
   - Architecture:
     - Embedding layer
     - SimpleRNN layers
     - Dense layers with dropout
   - Compile and fit the model:
     ```python
     model_rnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
     history_rnn = model_rnn.fit(X_train, y_train, validation_split=0.1, epochs=40, batch_size=128)
     ```

### Evaluation

Evaluate the models using accuracy, confusion matrix, and classification report:

1. **Accuracy:**
   ```python
   print("Accuracy of the model on Training Data is - ", model.evaluate(X_train, y_train)[1]*100 , "%")
   ```

2. **Confusion Matrix:**
   ```python
   cf = confusion_matrix(y_test, y_pred)
   sns.heatmap(cf, annot=True, cmap='Blues', fmt='g', xticklabels=['not bullying', 'bullying'], yticklabels=['not bullying', 'bullying'])
   plt.show()
   ```

3. **Classification Report:**
   ```python
   print(classification_report(y_test, y_pred, target_names=['not bullying', 'bullying']))
   ```

### Usage

To use the trained models for predictions, ensure you have preprocessed the text data similarly to the training phase. Load the trained model and perform predictions:

```python
from keras.models import load_model

model = load_model('path_to_your_model/model.h5')
predictions = model.predict(X_new)
```

### Acknowledgements

- The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/).
- The text preprocessing and modeling techniques were inspired by various NLP research papers and tutorials.

### class and uml daigrams
![image](https://github.com/Sabavat-Jayanth-Naik/Final-Year-Project-Cyberbullying-detection/assets/130920035/9d4cc743-f354-4af1-a9e6-c1237f7d38de)
![image](https://github.com/Sabavat-Jayanth-Naik/Final-Year-Project-Cyberbullying-detection/assets/130920035/74a59832-d9c5-401a-b089-0bc6a574d974)
![image](https://github.com/Sabavat-Jayanth-Naik/Final-Year-Project-Cyberbullying-detection/assets/130920035/64612c04-09c7-41c4-a829-8df58f4cae83)





