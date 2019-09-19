import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import recall_score
import utils
from sklearn.linear_model import LogisticRegression
import pdb
import random
import pickle as pkl


def fit_and_save_test_rf(model_fname, feature_names):
  data = pd.read_csv("newcomb-data.csv")
  random.seed(1)
  dataframes_for_each_study = utils.split_dataset_by_study(data, feature_names, excluded_study_labels=(20,))

  # Merge dataframes
  X = pd.DataFrame({f: [] for f in feature_names})
  y = np.zeros(0)
  for X_and_y in dataframes_for_each_study.values():
    X_, y_ = X_and_y
    X = pd.concat([X, X_])
    y = np.append(y, y_)

  # Fit model on merged dataset
  null_ixs = X.isnull().any(axis=1)
  y = y[np.where(null_ixs == 0)]
  X.dropna(axis=0, how='any', inplace=True)
  clf = RandomForestClassifier(n_estimators=100)
  clf.fit(X, y)

  # Save model
  pkl.dump(clf, open(model_fname, "wb"))

  return


if __name__ == "__main__":
  feature_names = ["fad_unpredictability", "determinism", "faith_intuition", "need_for_cognition"]
  model_fname = "model-4-features-9-19.sav"

  # Fit and save model
  # ToDo: exclude based on comprehension?
  fit_and_save_test_rf(model_fname, feature_names)

  # Load model and data to get test datapoint
  test_model = pkl.load(open(model_fname, "rb"))
  # data = pd.read_csv("newcomb-data.csv")
  x_test = [[30, 20, 3, 0.5]]

  # Run model on test point
  print(test_model.predict(x_test))





