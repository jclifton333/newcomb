import numpy as np
import pandas as pd


if __name__ == "__main__":
  data = pd.read_csv("newcomb-data.csv")

  # Remove columns not needed for analysis
  cols_to_remove = ["StartDate", "EndDate", "Platform", "IPAddress", "workerId", "Duration__in_seconds_",
                    "RecordedDate", "newcomb_explanation", "newcomb_two_explain", "newcomb_confidence",
                    "comprehension", "believability_prediction", "believability_scenario", "believability_1",
                    "believability_2", "believability_3", "knowing_prediction", "decoding", "feedback", "fair",
                    "long", "hard"]
  data.drop(labels=cols_to_remove, axis=1, inplace=True)

  # Create separate dataframes for each study
  dataframes_for_each_study = {}
  exluded_study_labels = (np.nan, 20)
  for study_number in data.Study.unique() if not study_number in exluded_study_labels:
    data_for_study = data[data["Study"] == study_number]

    # Drop empty columns, then take complete cases (null values are coded as ' ', need to change to nan)
    data_for_study.replace(' ', np.nan, regex=True)
    data_for_study.dropna(axis=1, how='all', inplace=True)
    data_for_study.dropna(axis=0, how='any', inplace=True)

    X_for_study = data_for_study.drop(labels=["newcomb_combined"], axis=1)
    y_for_study = data_for_study.newcomb_combined

    dataframes_for_each_study[study_number] = (X_for_study, y_for_study)

  # For each study, collect data from other studies with same features, train predictive model, and evaluate


