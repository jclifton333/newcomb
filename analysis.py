import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import recall_score
import pdb
import random

RANDOM_SEED = 3
RESULTS_FNAME = "newcomb-oos-results-2.csv"
SAVE = False
EXCLUDED_STUDY_LABELS = (np.nan, 20)
PAYOFF_DICT = {1.0: (3, 0.05), 2.0: (2.55, 0.45), 3.0: (2.25, 0.75), 12.0: (3, 0.50), 13.0: (2.5, 0.25),
               14.0: (2.5, 0.25), 15.0: (4, 0.5), 16.0: (4, 0.5), 17.0: (4, 0.5), 18.0: (4, 0.5), 19.0: (2.23, 0.28),
               20.0: (2.23, 0.11), 21.0: (2.23, 0.11), 22.0: (2, 0.1)}  # Payoffs for (box 1, box 2) in each study



def balanced_accuracy_score(y_true, y_pred):
  """
  Assuming y binary.

  :param y_pred:
  :param y_true:
  :return:
  """
  y_vals = np.unique(y_true)
  recall_1 = recall_score(y_true, y_pred, pos_label=y_vals[0])
  recall_2 = recall_score(y_true, y_pred, pos_label=y_vals[1])
  return (recall_1 + recall_2) / 2


if __name__ == "__main__":
  data = pd.read_csv("newcomb-data.csv")
  random.seed(RANDOM_SEED)

  # Remove columns not needed for analysis
  cols_to_remove = ["StartDate", "EndDate", "Platform", "IPAddress", "workerId", "Duration__in_seconds_",
                    "RecordedDate", "newcomb_explanation", "newcomb_two_explain", "newcomb_confidence",
                    "comprehension", "believability_prediction", "believability_scenario", "believability_1",
                    "believability_2", "believability_3", "knowing_prediction", "decoding", "feedback", "fair",
                    "long", "hard"]
  data.drop(labels=cols_to_remove, axis=1, inplace=True)

  # Create separate dataframes for each study
  dataframes_for_each_study = {}
  for study_number in data.Study.unique():
    if not np.isnan(study_number) and study_number not in EXCLUDED_STUDY_LABELS:
      data_for_study = data[data["Study"] == study_number]

      # Drop empty columns, then take complete cases (null values are coded as ' ', need to change to nan)
      data_for_study.replace(' ', np.nan, regex=True, inplace=True)
      data_for_study.dropna(axis=1, how='all', inplace=True)
      data_for_study.dropna(axis=0, how='any', inplace=True)
      data_for_study.applymap(float)
      data_for_study = data_for_study[data_for_study["newcomb_combined"] != 0.0]
      print("Study {} data size {}".format(study_number, data_for_study.shape))

      if data_for_study.shape[0] > 0:
        X_for_study = data_for_study.drop(labels=["newcomb_combined", "Study"], axis=1)
        X_for_study['payoff1'] = PAYOFF_DICT[study_number][0]
        X_for_study['payoff2'] = PAYOFF_DICT[study_number][1]
        y_for_study = data_for_study.newcomb_combined

        dataframes_for_each_study[study_number] = (X_for_study, y_for_study)

  # For each study, collect data from other studies with same features, train predictive model, and evaluate
  results = {'study_no': [], 'oob_score': [], 'test_acc': [], 'selected_features': [], 'studies_used': [],
             'random_seed': []}
  for test_study_number, X_and_y_test in dataframes_for_each_study.items():
    X_test, y_test = X_and_y_test
    feature_names = X_test.columns
    X_train = pd.DataFrame(columns=feature_names)
    y_train = np.zeros(0)
    overlapping_studies = []

    # Build training data
    for study_number, X_and_y_ in dataframes_for_each_study.items():
      if study_number != test_study_number:
        # Check if train study contains same features as test study; if so, add to dataset
        X_, y_ = X_and_y_
        if np.array_equal(np.intersect1d(X_.columns, feature_names), np.sort(feature_names)):
          X_train = pd.concat([X_train, X_[feature_names]])
          y_train = np.append(y_train, y_)
          overlapping_studies.append(study_number)

    if overlapping_studies:
      print('Fitting model for study {}'.format(test_study_number))

      # Fit model
      clf = RandomForestClassifier(oob_score=True, n_estimators=100)
      selector = RFE(clf)
      selector.fit(X_train, y_train)

      # Train and test accuracy
      selected_features = [n for i, n in enumerate(feature_names) if selector.support_[i]]
      train_acc = selector.estimator_.oob_score_
      test_predictions = selector.estimator_.predict(X_test[selected_features])
      test_acc = balanced_accuracy_score(y_test, test_predictions)

      # Add to results dict
      results['oob_score'].append(train_acc)
      results['test_acc'].append(test_acc)
      results['selected_features'].append(selected_features)
      results['studies_used'].append(overlapping_studies)
      results['study_no'].append(test_study_number)
      results['random_seed'].append(RANDOM_SEED)

  # Display and save to csv
  results_df = pd.DataFrame(results).sort_values(by="study_no")
  print(results_df.to_string())
  if SAVE:
    results_df.to_csv(RESULTS_FNAME)





