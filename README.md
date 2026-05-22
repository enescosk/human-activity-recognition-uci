# Human Activity Recognition — UCI Smartphones Dataset

Baseline classifiers for the [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) dataset, which contains accelerometer and gyroscope readings from a smartphone worn at the waist while subjects performed six activities (walking, walking upstairs, walking downstairs, sitting, standing, laying).

## Models

- [`project_folder/part1_knn.py`](project_folder/part1_knn.py) — k-Nearest Neighbors classifier
- [`project_folder/part2_dnn.py`](project_folder/part2_dnn.py) — Deep Neural Network classifier
- [`project_folder/part3_perceptrons.py`](project_folder/part3_perceptrons.py) — single-layer perceptron baselines

Each script reads the pre-split `UCI HAR Dataset/` folder (train/test partitions provided by the dataset authors), trains the model, and reports test accuracy + classification metrics.

## Run

```bash
pip install numpy pandas scikit-learn tensorflow
cd project_folder
python part1_knn.py
python part2_dnn.py
python part3_perceptrons.py
```

## Dataset

The original archive is committed under `project_folder/UCI HAR Dataset/` for reproducibility. See `UCI HAR Dataset.names` for full feature descriptions and the activity label encoding.
