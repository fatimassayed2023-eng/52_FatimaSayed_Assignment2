import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

# Load dataset
data = pd.read_csv("train_tweets.csv")
data = pd.read_csv("test_tweets.csv")

# Features and labels
X = data['tweet']
y = data['sentiment']

# Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Models
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(max_iter=200)
}

# Train + Evaluate
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    
    print(f"{name}:")
    print(f"Precision = {precision:.2f}")
    print(f"Recall = {recall:.2f}")
    print("-" * 30)






import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset (upload CSV in Colab first)
data = pd.read_csv("train_tweets.csv")
data = pd.read_csv("test_tweets.csv")

X = data['tweet']
y = data['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Models
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(max_iter=200)
}

accuracies = []

# Loop through models
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=['positive','neutral','negative'])

    # Plot and SHOW (this is the key change)
    plt.figure()
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0,1,2], ['positive','neutral','negative'])
    plt.yticks([0,1,2], ['positive','neutral','negative'])

    # Add numbers
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha='center', va='center')

    plt.show()   

# 📊 Accuracy Comparison Graph
plt.figure()
plt.bar(models.keys(), accuracies)
plt.title("Accuracy Comparison of Models")
plt.xlabel("Models")
plt.ylabel("Accuracy")

# Add values on bars
for i, v in enumerate(accuracies):
    plt.text(i, v, f"{v:.2f}", ha='center')

plt.show()  
