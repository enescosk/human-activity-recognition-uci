# Part 2) DNN On Raw Data (Inertial Signals):

import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


# 1. Finding the path of dataset, automatically:
basePath = os.path.join(os.path.dirname(__file__), "UCI HAR Dataset")
trainSignalPath = os.path.join(basePath, "train", "Inertial Signals")
testSignalPath  = os.path.join(basePath, "test",  "Inertial Signals")

# 2. Defining the inertial signal file name prefixes (9 channels = 3 acc + 3 gyro + 3 total acc):
inertialChannels = [
    "body_acc_x_", "body_acc_y_", "body_acc_z_",
    "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
    "total_acc_x_", "total_acc_y_", "total_acc_z_",
]

# 3. Loading segmented frames (shape → (numSamples, 128, 9)) from UCI-HAR inertial signals:
def loadFrames():
    Xtr = np.transpose(np.array([np.loadtxt(os.path.join(trainSignalPath, f"{s}train.txt")) for s in inertialChannels]), (1,2,0))
    Xte = np.transpose(np.array([np.loadtxt(os.path.join(testSignalPath,  f"{s}test.txt"))  for s in inertialChannels]), (1,2,0))
    ytr = np.loadtxt(os.path.join(basePath, "train", "y_train.txt")).astype(int) - 1
    yte = np.loadtxt(os.path.join(basePath, "test",  "y_test.txt")).astype(int) - 1
    return Xtr, ytr, Xte, yte

# 4. Loading the dataset and displaying the shapes:
trainFrames, trainLabels, testFrames, testLabels = loadFrames()
numClasses = len(np.unique(trainLabels))
print("Train shape:", trainFrames.shape)  
print("Test shape:",  testFrames.shape) 

# 5. Building a 1D CNN architecture with multiple convolution layers (CNN) and dropout:
def build_cnn(input_shape=(128,9), n_classes=6):
    return models.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
        layers.Conv1D(128, 3, activation='relu'),
        layers.Dropout(0.3),
        layers.Conv1D(256, 3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax')
    ])

print("\n=== Aggregate DNN (train frames -> test frames) ===")

# 6. Training a single global CNN model using train frames and validating on test frames:
agg = build_cnn(n_classes=numClasses)
agg.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
agg.fit(trainFrames, trainLabels, epochs=25, batch_size=32, validation_split=0.20, verbose=1)

# 7. Predicting activity classes on test frames and computing metrics:
y_pred = np.argmax(agg.predict(testFrames), axis=1)
print("Precision:", precision_score(testLabels, y_pred, average='macro'))
print("Recall:   ", recall_score(testLabels, y_pred, average='macro'))
print("F1:       ", f1_score(testLabels, y_pred, average='macro'))

print("\n=== Individual (Per-User) DNNs — stratified 80/20 within user ===")

# 8. Loading subject IDs for both train and test splits:
trainUsers = pd.read_csv(os.path.join(basePath, "train", "subject_train.txt"), header=None).values.ravel()
testUsers = pd.read_csv(os.path.join(basePath, "test",  "subject_test.txt"),  header=None).values.ravel()

# 9. Combining all user frames and labels for per user modeling:
totalFrames = np.concatenate([trainFrames, testFrames], axis=0)
totalLabels = np.concatenate([trainLabels, testLabels], axis=0)
totalUsers = np.concatenate([trainUsers,  testUsers],  axis=0)


# 10. Training user-specific CNNs with stratified 80/20 splits per user:
rows = []
for userID in sorted(np.unique(totalUsers)):

    # 10.1 Selecting all frames belonging to the current user:
    userIndices = np.where(totalUsers == userID)[0]
    userFrames, userLabels = totalFrames[userIndices], totalLabels[userIndices]
    
    # 10.2 Skipping the user if the number of samples is insufficient (under 30!):
    if len(userFrames) < 30:
        continue

    # 10.3 Splitting user's data into 80% train and 20% test (stratified):
    xUserTrain, xUserTest, yUserTrain, yUserTest = train_test_split(
        userFrames, userLabels, test_size=0.20, stratify=userLabels, random_state=42
    )

    # 10.4 Training a CNN model for this specific user:
    m = build_cnn(n_classes=numClasses)
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    m.fit(xUserTrain, yUserTrain, epochs=10, batch_size=32, verbose=0)

    # 10.5 Predicting on the user's test set and computing macro F1 score:
    y_hat_u = np.argmax(m.predict(xUserTest), axis=1)
    f1_u = f1_score(yUserTest, y_hat_u, average='macro')
    rows.append((userID, f1_u))

# 11. Creating a DataFrame to store and display user level results:
userDf = pd.DataFrame(rows, columns=["User ID", "F1 (macro)"])
print(userDf)
print("\nAverage User F1 (macro):", userDf["F1 (macro)"].mean())
