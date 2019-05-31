import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
import utils
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
import pdb
import random


RANDOM_SEED = 3
RESULTS_FNAME = None
SAVE = False


def importance_sampling_analysis(random_seed=RANDOM_SEED, results_fname=RESULTS_FNAME, save=SAVE):
  """
  For each study, fit a predictive model on each of the remaining studies, estimate the joint distribution of the
  features, and combine predictions using importance weighting.

  :param random_seed:
  :param results_fname:
  :param save:
  :return:
  """
  data = pd.read_csv("newcomb-data.csv")
  random.seed(random_seed)
  dataframes_for_each_study = utils.split_dataset_by_study(data)

  # For each study, collect data from other studies with same features, train predictive model, and evaluate
  results = {'study_no': [], 'test_acc': []}
  for test_study_number, X_and_y_test in dataframes_for_each_study.items():
    X_test, y_test = X_and_y_test
    feature_names = X_test.columns
    votes = np.zeros((0, len(y_test)))  # For storing predictions from each estimator; averaged at end to get final

    # Build training data
    for study_number, X_and_y_train in dataframes_for_each_study.items():
      if study_number != test_study_number:
        X_train, y_train = X_and_y_train

        # Fit model on overlapping features
        overlapping_features = np.intersect1d(X_train.columns, feature_names)
        X_train = X_train[overlapping_features]
        clf = RandomForestClassifier(oob_score=True, n_estimators=100)
        selector = RFECV(clf)
        selector.fit(X_train, y_train)

        # Compute prediction on test data and collect
        selected_features = [n for i, n in enumerate(overlapping_features) if selector.support_[i]]
        test_predictions = selector.estimator_.predict_proba(X_test[selected_features])[:, -1]
        votes = np.vstack((votes, test_predictions))

    # Get final predictions and compute accuracy
    final_votes = votes.mean(axis=0)
    final_predictions = (final_votes > 0.5) + 1  # Recode as 1/2
    test_acc = utils.balanced_accuracy_score(y_test, final_predictions)

    # Collect results
    results['study_no'] = test_study_number
    results['test_acc'] = test_acc

  # Display and save to csv
  results_df = pd.DataFrame(results).sort_values(by="study_no")
  print(results_df.to_string())
  if save:
    results_df.to_csv(results_fname)


if __name__ == "__main__":
  importance_sampling_analysis()
