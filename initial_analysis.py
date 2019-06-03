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
RESULTS_FNAME = "results/newcomb-oos--rfecv-estimator={}-restricted-features.csv"
SAVE = True


def studywise_logistic_regression(feature_names):
  """
  Analyze relationship between single feature and p(one box) across studies.

  :param features_name:
  :return:
  """
  PAYOFF_DICT = {1.0: (3, 0.05), 2.0: (2.55, 0.45), 3.0: (2.25, 0.75), 12.0: (3, 0.50), 13.0: (2.5, 0.25),
                 14.0: (2.5, 0.25), 15.0: (4, 0.5), 16.0: (4, 0.5), 17.0: (4, 0.5), 18.0: (4, 0.5), 19.0: (2.23, 0.28),
                 20.0: (2.23, 0.11), 21.0: (2.23, 0.11), 22.0: (2, 0.1)}  # Payoffs for (box 1, box 2) in each study

  data = pd.read_csv("newcomb-data.csv")
  results = {'study_no': [], 'logit_coef': [], 'logit_score': [], 'payoff1': [], 'payoff2': [], 'sample_size': []}
  for study_number in data.Study.unique():
    if np.isfinite(study_number):
      # Get complete cases of target and specified feature
      data_for_study = data[data["Study"] == study_number]
      data_for_study['payoff1'] = PAYOFF_DICT[study_number][0]
      data_for_study['payoff2'] = PAYOFF_DICT[study_number][1]
      payoff1 = np.array(data_for_study.payoff1)[0]
      payoff2 = np.array(data_for_study.payoff2)[0]
      payoffRatio = payoff1 / payoff2
      data_for_study = data_for_study[feature_names + ["newcomb_combined"]]
      data_for_study.replace(' ', np.nan, regex=True, inplace=True)
      data_for_study.dropna(axis=0, inplace=True)
      data_for_study = data_for_study.applymap(float)
      data_for_study = data_for_study[data_for_study["newcomb_combined"] != 0.0]

      # If there's any data left, fit a logistic regression
      if data_for_study.shape[0] > 0:
        X = data_for_study[feature_names]
        y = data_for_study.newcomb_combined
        logit = LogisticRegression(C=1e40)
        logit.fit(X, y)

        # Construct string to display coef for each feature
        coef_str = ''
        for ix, feature_name in enumerate(feature_names):
          coef_str += ' {}: {}'.format(feature_name, np.round(logit.coef_[0, ix], 2))

        results['study_no'].append(study_number)
        results['payoff1'].append(np.round(payoff1, 2))
        results['payoff2'].append(np.round(payoff2, 2))
        results['logit_coef'].append(coef_str)
        bpa = utils.balanced_accuracy_score(y, logit.predict(X))
        # bpa = utils.balanced_accuracy_for_optimal_threshold(y, logit.predict_proba(X)[:, -1])
        results['logit_score'].append(bpa)
        results['sample_size'].append(X.shape[0])

  # Display
  results_df = pd.DataFrame(results).sort_values(by="study_no")
  print('\n{}'.format(results_df.to_string()))

  return results_df


def summary_statistics(feature_names, excluded_studies=(21,)):
  data = pd.read_csv("newcomb-data.csv")
  dataframes_for_each_study = utils.split_dataset_by_study(data, feature_names, excluded_study_labels=excluded_studies)
  for study_no, X_and_y in dataframes_for_each_study.items():
    X, _ = X_and_y
    print('\nStudy {}\n{}'.format(study_no, X.describe().to_string()))
  return


def leave_one_study_out_analysis(feature_names=None, clf=RandomForestClassifier(oob_score=True, n_estimators=100),
                                 excluded_studies=(20,),
                                 random_seed=RANDOM_SEED,
                                 results_fname=RESULTS_FNAME, save=SAVE):
  """
  For each study, fit a predictive model on data from previous studies which share the target study's features.

  :param feature_names: list of strings or None; if provided, use only these features in the analysis.
  :param random_seed:
  :param results_fname:
  :param save:
  :return:
  """
  data = pd.read_csv("newcomb-data.csv")
  random.seed(random_seed)
  dataframes_for_each_study = utils.split_dataset_by_study(data, feature_names, excluded_study_labels=excluded_studies)
  clf_name = clf.__class__.__name__
  results_fname = results_fname.format(clf_name)

  # For each study, collect data from other studies with same features, train predictive model, and evaluate
  results = {'study_no': [], 'train_acc': [], 'test_acc': [], 'selected_features': [], 'studies_used': [],
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
      selector = RFECV(clf)
      selector.fit(X_train, y_train)

      # Train and test accuracy
      selected_features = [n for i, n in enumerate(feature_names) if selector.support_[i]]
      train_predictions = selector.estimator_.predict_proba(X_train[selected_features])[:, -1]
      train_acc, threshold = utils.balanced_accuracy_for_optimal_threshold(y_train, train_predictions)
      test_predictions = (selector.estimator_.predict_proba(X_test[selected_features])[:, -1] > threshold) + 1
      test_acc = utils.balanced_accuracy_score(y_test, test_predictions)

      # Add to results dict
      results['train_acc'].append(train_acc)
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
  feature_names = ["gender", "knights_knaves"]
  excluded_studies = (1, 2, 3, 21)
  summary_statistics(feature_names, excluded_studies=excluded_studies)


