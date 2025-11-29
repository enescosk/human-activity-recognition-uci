# Part 1) K-NN Classifier:

import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# 1. Finding the path of dataset, automatically:
basePath = os.path.join(os.path.dirname(__file__), "UCI HAR Dataset")
trainPath = os.path.join(basePath, "train")
testPath = os.path.join(basePath, "test")

# 2. Loading the training/testing data:
trainFeatures = pd.read_csv(os.path.join(trainPath, "X_train.txt"), sep=r"\s+", header=None)
trainLabels = pd.read_csv(os.path.join(trainPath, "y_train.txt"), sep=r"\s+", header=None)
testFeatures = pd.read_csv(os.path.join(testPath, "X_test.txt"), sep=r"\s+", header=None)
testLabels = pd.read_csv(os.path.join(testPath, "y_test.txt"), sep=r"\s+", header=None)

# 3. Implementing normalization on features:
normalizer = StandardScaler()
trainFeatures = normalizer.fit_transform(trainFeatures)
testFeatures = normalizer.transform(testFeatures)

print("\n=== Aggregate Model ===")

# 4. Training a single flobal K-NN model (k = 2):
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(trainFeatures, trainLabels.values.ravel())

# 5. Predicting activity labels on the test set:
predGlobal = knn.predict(testFeatures)

# 6. Calculating and displaying the macro-averaged metrics, where each class contributes equally regardless of sample count:
precision = precision_score(testLabels, predGlobal, average='macro')
recall = recall_score(testLabels, predGlobal, average='macro')
f1 = f1_score(testLabels, predGlobal, average='macro')
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


print("\n=== Individual (Per-User) Models ===")

# 7. Loading user IDs for train and test sets:
trainUsers = pd.read_csv(os.path.join(trainPath, "subject_train.txt"), header=None)
testUsers = pd.read_csv(os.path.join(testPath, "subject_test.txt"), header=None)

# 8. Combining all samples into one dataset for per-user modeling:
totalFeatures = np.concatenate((trainFeatures, testFeatures))
totalLabels = np.concatenate((trainLabels.values.ravel(), testLabels.values.ravel()))
totalUsers = np.concatenate((trainUsers.values.ravel(), testUsers.values.ravel()))


# 9. Iterating through all unique users:
uniqueIds = sorted(np.unique(totalUsers))
userScores = []

for user in uniqueIds:

    # 9.1 Selecting all samples belonging to the current user:
    userId = np.where(totalUsers == user)[0]
    userFeatures = totalFeatures[userId]
    userLabels = totalLabels[userId]
    
    # 9.2 Skipping if the samples for this user are insufficient:
    if len(userFeatures) < 20:
        continue
    
    # 9.3 Splitting the user data into 80% train and 20% test.
    xUserTrain, xUserTest, yUserTrain, yUserTest = train_test_split(userFeatures, userLabels, test_size=0.2, random_state=42, stratify=userLabels)
    
    # 9.4 Training a KNN model for the specific user.
    knn_user = KNeighborsClassifier(n_neighbors=5)
    knn_user.fit(xUserTrain, yUserTrain)
    y_pred_u = knn_user.predict(xUserTest)

    # 9.5 Computing the corresponding user F1 score.
    f1ScoreUser = f1_score(yUserTest, y_pred_u, average='macro')
    userScores.append((user, f1ScoreUser))

# 10. Creating a DataFrame for per-user performance results and displaying the scores:
userDf = pd.DataFrame(userScores, columns=["User ID", "F1 Score"])
print(userDf)
print("\nAverage User F1:", userDf["F1 Score"].mean())
