import pdb
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score


def split_dataset_by_study(data, feature_names, excluded_study_labels=(20,),excluded_participants=None):
  """

  :param data: Pandas df containing full newcomb data.
  :param exluded_study_labels:
  :return:
  """
  PAYOFF_DICT = {1.0: (3, 0.05), 2.0: (2.55, 0.45), 3.0: (2.25, 0.75), 12.0: (3, 0.50), 13.0: (2.5, 0.25),
                 14.0: (2.5, 0.25), 15.0: (4, 0.5), 16.0: (4, 0.5), 17.0: (4, 0.5), 18.0: (4, 0.5), 19.0: (2.23, 0.28),
                 20.0: (2.23, 0.11), 21.0: (2.23, 0.11), 22.0: (2, 0.1)}  # Payoffs for (box 1, box 2) in each study

  # Remove columns not needed for analysis
  # cols_to_remove = ["StartDate", "EndDate", "Platform", "IPAddress", "workerId", "Duration__in_seconds_",
  #                   "RecordedDate", "newcomb_explanation", "newcomb_two_explain",
  #                    "believability_3", "knowing_prediction", "decoding", "feedback", "fair",
  #                   "long", "hard","crt_score3_original", "crt_score_5_original"]
  cols_to_keep = ['Study', 'newcomb_combined', 'gender', 'age', 'ethnicity', 'education', 'dancing_physicist',
                   'religiosity', 'cheating_scale_1', 'cheating_scale_2',
                   'cheating_scale_3', 'cheating_scale_4', 'cheated', 'knights_knaves',
                   'extraverted_enthusiastic', 'critical_quarrelsome',
                   'dependable_selfdisciplined', 'anxious', 'opennewexperiences_complex',
                   'reserved_quiet', 'sympathetic_warm', 'disorganized_careless',
                   'calm_emotionallystable', 'conventional_uncreative', 'fad_fatalism',
                   'fad_unpredictability', 'fad_free_will_new',
                   'fad_scientists_figureout_humans', 'fad_human_animal_samelaws_e',
                   'determinism', 'dualism', 'faith_intuition', 'need_for_cognition',
                   'operant_conditioning', 'knight_knave_score', 'superstition_score',
                   'big_five_intellect', 'big_five_modesty', 'logical_reasoning_score_5',
                   'crt_combined_recoded5', 'comprehension'] # ToDo: should drop comprehension in model-fitting

  # Also drop cols that aren't provided in feature_names
  cols_to_remove = []
  for col in data.columns:
    # if col not in cols_to_keep and col not in feature_names:
    if col not in cols_to_keep:
      cols_to_remove.append(col)

  data.drop(labels=cols_to_remove, axis=1, inplace=True)

  if 'ethnicity' in data.columns:
    # data['ethnicity'] = data.ethnicity.astype('category')  # Convert ethnicity coding from numeric to categorical
    data['ethnicity'] = data.ethnicity.astype(str)  # Convert ethnicity coding from numeric to categorical

  # Create separate dataframes for each study
  dataframes_for_each_study = {}
  for study_number in data.Study.unique():
    if study_number not in excluded_study_labels:
      data_for_study = data[data["Study"] == study_number]

      # Drop empty columns, then take complete cases (null values are coded as ' ', need to change to nan)
      data_for_study.replace(' ', np.nan, regex=True, inplace=True)
      data_for_study.dropna(axis=1, how='all', inplace=True)
      data_for_study.dropna(axis=0, how='any', inplace=True)
      data_for_study = data_for_study.applymap(float)  # ToDo: what about categorical vars?

      if data_for_study.shape[0] > 0:
        data_for_study = data_for_study[data_for_study["newcomb_combined"] != 0.0]

        # exclude participants if function for exclusion is given
        if excluded_participants is not None:
          data_for_study.drop(labels=excluded_participants(data_for_study, study_number), axis=0, inplace=True)

        print("Study {} data size {}".format(study_number, data_for_study.shape))

        if data_for_study.shape[0] > 0:
          X_for_study = data_for_study.drop(labels=[x for x in ["newcomb_combined", "Study","comprehension",
                                                                "newcomb_confidence","believability_prediction",
                                                                "believability_scenario", "believability_1",
                                                                "believability_2"] if x in data_for_study.columns],
                                            axis=1)
          X_for_study['payoff1'] = PAYOFF_DICT[float(study_number)][0]
          X_for_study['payoff2'] = PAYOFF_DICT[float(study_number)][1]
          X_for_study['payoffRatio'] = X_for_study.payoff1 / X_for_study.payoff2
          y_for_study = data_for_study.newcomb_combined
          dataframes_for_each_study[float(study_number)] = (X_for_study, y_for_study)
  return dataframes_for_each_study


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


def bpa_scorer(estimator, X, y):
  """

  :param estimator:
  :param X:
  :param y:
  :return:
  """
  y_pred = estimator.predict(X)
  return balanced_accuracy_score(y, y_pred)


def balanced_accuracy_for_optimal_threshold(y_true, phat):
  best_accuracy = 0.0
  best_threshold = None
  for threshold in phat:
    y_pred = (phat > threshold) + 1
    bpa = balanced_accuracy_score(y_true, y_pred)
    if bpa > best_accuracy:
      best_threshold = threshold
      best_accuracy = bpa
  return best_accuracy, best_threshold

