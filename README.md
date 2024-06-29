# House Price Prediction using TensorFlow Decision Forests

This project involves predicting house prices using TensorFlow Decision Forests. The dataset used is the Ames Housing dataset, which includes 79 explanatory variables describing various aspects of residential homes in Ames, Iowa. The goal is to predict the final price of each home.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:
- TensorFlow
- TensorFlow Decision Forests
- Pandas
- Seaborn
- Matplotlib

You can install these libraries using pip:

```bash
pip install tensorflow tensorflow_decision_forests pandas seaborn matplotlib
```

### Project Structure

- `project/`
  - `dataset.csv`: The dataset containing the housing data.
  - `train.py`: The script to train and evaluate the model.
  - `README.md`: This file.

### Dataset

The dataset used for this project is `dataset.csv`, which contains 80 columns (features and target).

## Code Explanation

Below is a step-by-step explanation of the code used for this project.

### Import Libraries

```python
import tensorflow_decision_forests as tfdf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
```

### Load the Dataset

Load the dataset and display the shape and the first few rows.

```python
dataset = pd.read_csv("project/dataset.csv")
print("Full dataset shape:", dataset.shape)
print(dataset.head(3))
```

Drop the unnecessary `Id` column.

```python
dataset = dataset.drop('Id', axis=1)
```

### Data Inspection

Inspect the types of feature columns and display basic statistics of the target variable (`SalePrice`).

```python
dataset.info()
print(dataset['SalePrice'].describe())
```

Plot the distribution of the target variable.

```python
plt.figure(figsize=(9, 8))
sns.histplot(dataset['SalePrice'], color='g', bins=100, kde=True)
plt.show()
```

### Split the Dataset

Split the dataset into training and testing datasets.

```python
def split_dataset(dataset, test_ratio=0.30):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(dataset)
print("{} examples in training, {} examples in testing.".format(
    len(train_ds_pd), len(valid_ds_pd)))
```

### Convert to TensorFlow Dataset

Convert the Pandas DataFrames to TensorFlow Datasets.

```python
label = 'SalePrice'
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
```

### Train the Model

Create and train a Random Forest model.

```python
model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
model.compile(metrics=["mse"])
model.fit(train_ds)
```

### Evaluate the Model

Evaluate the model on the validation dataset and plot the training logs.

```python
evaluation = model.evaluate(valid_ds, return_dict=True)
for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

logs = model.make_inspector().training_logs()
plt.plot([log.num_trees for log in logs], [log.evaluation.rmse for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("RMSE (out-of-bag)")
plt.show()
```

### Variable Importances

Display and plot the variable importances.

```python
inspector = model.make_inspector()
print("Variable importances:")
for importance in inspector.variable_importances().keys():
    print("\t", importance)

importance = inspector.variable_importances()["NUM_AS_ROOT"]
print(importance)

plt.figure(figsize=(12, 4))
feature_names = [vi[0].name for vi in importance]
feature_importances = [vi[1] for vi in importance]
plt.barh(feature_names, feature_importances)
plt.xlabel("NUM_AS_ROOT")
plt.title("Variable Importances")
plt.show()
```

### Make Predictions

Predict on the test dataset and save the results to a CSV file.

```python
test_data = pd.read_csv("project/test.csv")
ids = test_data.pop('Id')
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_data, task=tfdf.keras.Task.REGRESSION)
preds = model.predict(test_ds)
output = pd.DataFrame({'Id': ids, 'SalePrice': preds.squeeze()})
output.to_csv('submission.csv', index=False)
print(output.head())
```

## Conclusion

This project demonstrates how to use TensorFlow Decision Forests to predict house prices using a comprehensive dataset. The model utilizes tree-based algorithms which are robust and provide good performance on tabular data. Further tuning and experimentation with different models and hyperparameters can help improve the prediction accuracy.
