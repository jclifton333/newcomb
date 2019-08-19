from initial_analysis import *

RANDOM_SEED = 3
RESULTS_FNAME = "results/newcomb-oos--rfecv-estimator={}-restricted-features-excl-rules.csv"
SAVE = True

def excluded_participants_rule_1(dataframe,studynumber):
    max_comp = np.max(dataframe['comprehension'])
    comp_thr = max_comp*0.70
    excl_list = dataframe.index[dataframe['comprehension']<comp_thr]
    return excl_list

def excluded_participants_rule_2(dataframe,studynumber):
    max_comp = np.max(dataframe['comprehension'])
    comp_thr = max_comp*0.75
    excl_list = dataframe.index[dataframe['comprehension']<comp_thr]
    return excl_list

def excluded_participants_rule_3(dataframe,studynumber):
    max_comp = np.max(dataframe['comprehension'])
    comp_thr = max_comp*0.70
    excl_1= dataframe['comprehension']<comp_thr

    if 'newcomb_confidence' in 'dataframe.columns':
        incon_1 = (dataframe['newcomb_confidence']>4) & (dataframe['newcomb_combined']==1)
        incon_2 = (dataframe['newcomb_confidence'] < 4) & (dataframe['newcomb_combined']==2)
        excl_2 = incon_1 | incon_2
        excl_list = dataframe.index[excl_1|excl_2]
    else:
        excl_list = dataframe.index[excl_1]
    return excl_list

def excluded_participants_rule_4(dataframe,studynumber):

    max_comp = np.max(dataframe['comprehension'])
    comp_thr = max_comp * 0.70
    excl_1 = dataframe['comprehension'] < comp_thr

    if studynumber <4 or studynumber >18:
        excl_2 = (dataframe['believability_prediction']>=4 ) | (dataframe['believability_scenario']<=3 )

    elif studynumber<19 and studynumber>11:
        excl_2 = (dataframe['believability_1']<=2 ) & (dataframe['believability_2']<=2 )

    excl_list = dataframe.index[excl_1 | excl_2]

    return excl_list

def excluded_participants_rule_5(dataframe,studynumber):

    max_comp = np.max(dataframe['comprehension'])
    comp_thr = max_comp * 0.70
    excl_1 = dataframe['comprehension'] < comp_thr

    if studynumber <4 or studynumber >18:
        excl_2 = (dataframe['believability_prediction']>=4 ) | (dataframe['believability_scenario']<=3 )

    elif studynumber<19 and studynumber>11:
        excl_2 = (dataframe['believability_1']<=2 ) & (dataframe['believability_2']<=2 )

    if 'newcomb_confidence' in 'dataframe.columns':
        incon_1 = (dataframe['newcomb_confidence']>4) & (dataframe['newcomb_combined']==1)
        incon_2 = (dataframe['newcomb_confidence'] < 4) & (dataframe['newcomb_combined']==2)
        excl_3 = incon_1 | incon_2
        excl_list = dataframe.index[excl_1 | excl_2|excl_3]
    else:
        excl_list = dataframe.index[excl_1|excl_2]
    return excl_list



list_of_rules = [excluded_participants_rule_5]

def analysis_with_exclusions(feature_names=None,
                        clf=RandomForestClassifier(oob_score=True, n_estimators=100),
                                 excluded_studies=(20,),
                                 random_seed=RANDOM_SEED,
                                 results_fname=RESULTS_FNAME, save=SAVE):
    end_result = []

    for excluded_participants in list_of_rules:


        result = leave_one_study_out_analysis(feature_names=feature_names,
                                         clf=clf,
                                         excluded_studies=excluded_studies,
                                         random_seed=random_seed,
                                         results_fname="foobar", save=False, excluded_participants=excluded_participants)
        end_result.append(result)

    result = pd.concat(end_result,keys=['comp70 + incon + bel'])

    print(result.to_string())
    if save:
        result.to_csv(results_fname)
    return result


def leave_one_study_out_with_many_seeds(feature_names=None,
                                        clf=RandomForestClassifier(oob_score=True, n_estimators=100),
                                        excluded_studies=(20,),
                                        num_random_seeds=100,
                                        excluded_participants=excluded_participants_rule_1):

  mean_test_accuracy = None  # Initialize array of studywise mean test accuracies across seeds
  for seed in range(num_random_seeds):
    # ToDo; inefficient since leave_one_study_out_analysis loads and organizes data for every seed?
    results_df_for_seed = leave_one_study_out_analysis(feature_names=feature_names, clf=clf,
                                                       excluded_studies=excluded_studies, save=False,
                                                       excluded_participants=excluded_participants)
    if mean_test_accuracy is None:
      mean_test_accuracy = results_df_for_seed.test_acc
    else:  # Incremental update to means
      mean_test_accuracy += (results_df_for_seed - mean_test_accuracy) / (seed + 1)

  results_dict = {'mean_test_acc': mean_test_accuracy, 'study_no': results_df_for_seed.study_no}
  results_df = pd.DataFrame(results_dict)
  print(results_df.to_string())


if __name__ == "__main__":
  leave_one_study_out_with_many_seeds(num_random_seeds=1)
