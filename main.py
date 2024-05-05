import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from graph import TextGraph
from knn_graph import GraphKNN
from Training import makeGraph, preprocess
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    confusion_matrix,
)


def main():
    sport = pd.read_csv("Sports.csv", delimiter=";", encoding="latin1")

    healthFitness = pd.read_csv("health_fitness.csv", delimiter=";", encoding="latin1")
    travel = pd.read_csv("Travel.csv", delimiter=";", encoding="latin1")
    tg = TextGraph("output.csv")
    allData = pd.concat([healthFitness, sport, travel], ignore_index=True)
    X_train, X_test, y_train, y_test = train_test_split(
        allData["text"], allData["label"], test_size=0.2, random_state=42
    )

    allData["text"] = allData["text"].apply(preprocess)
    trainTexts = allData["text"].tolist()
    trainLabels = allData["label"].tolist()
    trainGraphs = [makeGraph(text) for text in trainTexts]
    graphClassifier = GraphKNN(k=3)
    graphClassifier.fit(trainGraphs, trainLabels)
    testText = [
        "Travel often seen as an escape from the mundane",
        "Cricket is the most popular sport in Pakistan",
        "How much exercise do I need? How much exercise",
    ]
    testGraphs = [makeGraph(text) for text in testText]
    predictions = [graphClassifier.predict(graph) for graph in testGraphs]
    testLabels = ["Travel", "Sports", "Health & Fitness"]
    accuracy = accuracy_score(testLabels, predictions)
    accuracy_percentage = accuracy * 100
    f1Scores = f1_score(testLabels, predictions, average=None)
    f1ScorePercentage = f1Scores[0] * 100
    jaccard = jaccard_score(testLabels, predictions, average=None)
    jaccard_percentage = jaccard[0] * 100

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100)  # 100 decision trees

    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    # Plot confusion matrix
    confMatrix = confusion_matrix(
        list(testLabels),
        list(predictions),
        labels=list(set(testLabels)),
    )

    tg.divide_data()
    tg.process_training_data()
    common_subgraphs = tg.find_common_subgraphs()
    print("Number of common subgraphs:", len(common_subgraphs))
    tg.visualize_last_graph()

    print("Accuracy: {:.2f}%".format(accuracy_percentage))
    print("Precision: {:.2f}%".format((precision) * 100))
    print("Recall: {:.2f}%".format((recall) * 100))
    print("F1 Score:", "{:.2f}%".format(f1ScorePercentage))
    print("Jaccard Similarity:", "{:.2f}%".format(jaccard_percentage))

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(
        confMatrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=set(testLabels),
        yticklabels=set(testLabels),
    )
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    main()
