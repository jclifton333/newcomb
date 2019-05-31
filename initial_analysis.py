import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import recall_score
import utils
from sklearn.linear_model import LogisticRegression
import pdb
import random

RANDOM_SEED = 4
RESULTS_FNAME = "results/newcomb-oos--rfecv-3-diff-seed-with-payoffs.csv"
SAVE = True


def studywise_bivariate_analysis(feature_name):
  """
  Analyze relationship between single feature and p(one box) across studies.

  :param feature_name:
  :return:
  """
  data = pd.read_csv("newcomb-data.csv")
  results = {'study_no': [], 'logit_coef': [], 'logit_score': [], 'sample_size': []}
  for study_number in data.Study.unique():
    if np.isfinite(study_number):
      # Get complete cases of target and specified feature
      data_for_study = data[data["Study"] == study_number]
      data_for_study = data_for_study[[feature_name, "newcomb_combined"]]
      data_for_study.replace(' ', np.nan, regex=True, inplace=True)
      data_for_study.dropna(axis=0, inplace=True)
      data_for_study = data_for_study.applymap(float)
      data_for_study = data_for_study[data_for_study["newcomb_combined"] != 0.0]


      # If there's any data left, fit a logistic regression
      if data_for_study.shape[0] > 0:
        X = data_for_study[feature_name][:, np.newaxis]
        y = data_for_study.newcomb_combined
        logit = LogisticRegression(C=1e40)
        logit.fit(X, y)

        results['study_no'].append(study_number)
        results['logit_coef'].append(np.round(logit.coef_[0,0], 2))
        bpa = utils.balanced_accuracy_for_optimal_threshold(y, logit.predict_proba(X)[:, -1])
        results['logit_score'].append(bpa)
        results['sample_size'].append(X.shape[0])

  # Display
  results_df = pd.DataFrame(results).sort_values(by="study_no")
  print('\nFeature: {}\n{}'.format(feature_name, results_df.to_string()))

  return


def leave_one_study_out_analysis(random_seed=RANDOM_SEED, results_fname=RESULTS_FNAME, save=SAVE):
  """
  For each study, fit a predictive model on data from previous studies which share the target study's features.

  :param random_seed:
  :param results_fname:
  :param save:
  :return:
  """
  data = pd.read_csv("newcomb-data.csv")
  random.seed(random_seed)
  dataframes_for_each_study = utils.split_dataset_by_study(data)

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
      selector = RFECV(clf, scoring=utils.bpa_scorer)
      selector.fit(X_train, y_train)

      # Train and test accuracy
      selected_features = [n for i, n in enumerate(feature_names) if selector.support_[i]]
      train_acc = selector.estimator_.oob_score_
      test_predictions = selector.estimator_.predict(X_test[selected_features])
      test_acc = utils.balanced_accuracy_score(y_test, test_predictions)

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
  if save:
    results_df.to_csv(results_fname)


if __name__ == "__main__":
  leave_one_study_out_analysis()

