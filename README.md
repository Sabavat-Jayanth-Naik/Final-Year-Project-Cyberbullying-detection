## Cyberbullying Detection Model - README

### Project Overview
This project focuses on detecting cyberbullying in tweets. The dataset used contains tweets labeled with different types of cyberbullying such as religion, gender, age, ethnicity, and non-cyberbullying. The goal is to classify tweets as either "cyberbullying" or "not_cyberbullying."

### Dataset
The dataset used is `cyberbullying_tweets.csv`, which contains the following columns:
- `tweet_text`: The text of the tweet.
- `cyberbullying_type`: The type of cyberbullying (e.g., religion, gender, age, ethnicity, not_cyberbullying).

### Data Preprocessing
Data preprocessing steps include:
1. Removing special characters, URLs, and emojis.
2. Converting text to lowercase.
3. Removing stopwords.
4. Stemming and lemmatizing the text.
5. Removing duplicate tweets and tweets with less than 3 words.

### Model Training
The model used is a Bidirectional LSTM (BiLSTM) neural network. The steps involved in training the model include:
1. Tokenizing the text data.
2. Padding the sequences to a maximum length.
3. Creating the BiLSTM model with embedding, LSTM, and dense layers.
4. Compiling and training the model with a sparse categorical cross-entropy loss function and the Adam optimizer.

### Model Evaluation
The model was evaluated on both training and testing datasets. Below are the key metrics:

#### Training Data Accuracy
- Accuracy: 94.11%

#### Testing Data Accuracy
- Accuracy: 91.96%


### Classification Report
The classification report for the testing data is as follows:

|                | precision | recall | f1-score | support |
|----------------|-----------|--------|----------|---------|
| not bullying   | 0.76      | 0.78   | 0.77     | 960     |
| bullying       | 0.95      | 0.95   | 0.95     | 4622    |
| **accuracy**   |           |        | 0.92     | 5582    |
| **macro avg**  | 0.86      | 0.86   | 0.86     | 5582    |
| **weighted avg**| 0.92     | 0.92   | 0.92     | 5582    |

### Validation Graphs

1. **Training and Validation Accuracy**

   - **X-Axis:** Epochs
   - **Y-Axis:** Accuracy
   
   This graph plots the accuracy of the model over epochs for both the training and validation datasets.

![image](https://github.com/user-attachments/assets/8b43f659-7f26-4d59-ae46-5b89fb46daff)
##
   - **X-Axis:** Epochs
   - **Y-Axis:** Loss
![image](https://github.com/user-attachments/assets/92e8380b-5f55-45a6-a3c2-9a1a99bf024c)


### Conclusion
The BiLSTM model achieves a high accuracy of 91.96% on the testing data, demonstrating its effectiveness in detecting cyberbullying in tweets. The precision, recall, and F1-score are particularly high for the "cyberbullying" class, making it a reliable model for this task.
