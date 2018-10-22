from model import model
from baseline_model import baseline_model
import pandas as pd
import csv


'''
Notes:
Feature selection: 
1) for SelectKBest, k=23 is the best among all k's

dataset change checklist:
1) dataset_size(model)
2) gender(model and helper)
3) split
4) csv
5) demographic parity def
'''


csv_path = "/dataset.csv"
# Importing the dataset
dataset = pd.read_csv(csv_path)

with open('/Final_Result.csv', 'a', newline="", encoding="utf-8") as result_csv:
    result_csv_writer = csv.writer(result_csv)
    for itr in range(10):

        result_csv_writer.writerow(["*******************************************************************************************************************************************************"])
        result_csv_writer.writerow(["Take - " + str(itr + 1)])
        result_csv_writer.writerow([])
        result_csv_writer.writerow(
            ["Model", "Total_Accuracy", "Male_Accuracy", "Female_Accuracy", "Equal_opportunity", "parity_diff"])


        output = {
            "Model": {"Total_Accuracy": 0, "Male_Accuracy": 0, "Female_Accuracy": 0, "Eq_odds_tpr": 0, "Eq_odds_fpr": 0,
                      "parity_diff": 0},
            "Model_with_feature_selection": {"Total_Accuracy": 0, "Male_Accuracy": 0, "Female_Accuracy": 0,
                                             "Eq_odds_tpr": 0, "Eq_odds_fpr": 0, "parity_diff": 0},
            "Baseline_Model": {"Total_Accuracy": 0, "Male_Accuracy": 0, "Female_Accuracy": 0, "Eq_odds_tpr": 0,
                               "Eq_odds_fpr": 0, "parity_diff": 0},
            "Baseline_Model_with_feature_selection": {"Total_Accuracy": 0, "Male_Accuracy": 0, "Female_Accuracy": 0,
                                                      "Eq_odds_tpr": 0, "Eq_odds_fpr": 0, "parity_diff": 0}
        }
        num_itr = 10
        for i in range(num_itr):
            interim_output = {
                "Model": {"Total_Accuracy": 0, "Male_Accuracy": 0, "Female_Accuracy": 0, "Eq_odds_tpr": 0,
                          "Eq_odds_fpr": 0,
                          "parity_diff": 0},
                "Model_with_feature_selection": {"Total_Accuracy": 0, "Male_Accuracy": 0, "Female_Accuracy": 0,
                                                 "Eq_odds_tpr": 0, "Eq_odds_fpr": 0, "parity_diff": 0},
                "Baseline_Model": {"Total_Accuracy": 0, "Male_Accuracy": 0, "Female_Accuracy": 0, "Eq_odds_tpr": 0,
                                   "Eq_odds_fpr": 0, "parity_diff": 0},
                "Baseline_Model_with_feature_selection": {"Total_Accuracy": 0, "Male_Accuracy": 0, "Female_Accuracy": 0,
                                                          "Eq_odds_tpr": 0, "Eq_odds_fpr": 0, "parity_diff": 0}
            }
            model(dataset, "Model", False, output.get("Model"), interim_output)
            model(dataset, "Model_with_feature_selection", True, output.get("Model_with_feature_selection"), interim_output)
            baseline_model(dataset, "Baseline_Model", False, output.get("Baseline_Model"), interim_output)
            baseline_model(dataset, "Baseline_Model_with_feature_selection", True, output.get("Baseline_Model_with_feature_selection"), interim_output)


            result_csv_writer.writerow(["Iteration - " + str(i+1)])
            for key in ["Model", "Baseline_Model", "Model_with_feature_selection", "Baseline_Model_with_feature_selection"]:
                result_csv_writer.writerow(
                    [key, interim_output.get(key).get("Total_Accuracy"), interim_output.get(key).get("Male_Accuracy"),
                     interim_output.get(key).get("Female_Accuracy"),
                     interim_output.get(key).get("Eq_odds_tpr"),
                     interim_output.get(key).get("parity_diff")])

        output = {key: {k: v / num_itr for k, v in val.items()} for key, val in output.items()}
        result_csv_writer.writerow([""])
        result_csv_writer.writerow(["Averaged - "])
        if (abs(output.get("Model").get("Eq_odds_tpr")) < abs(output.get("Baseline_Model").get("Eq_odds_tpr"))) and (abs(output.get("Model").get("parity_diff")) < abs(output.get("Baseline_Model").get("parity_diff"))):
            print("Achieved at iteration- "+ str(itr+1))
            result_csv_writer.writerow(["Achieved at iteration- " + str(itr+1)])
        for key in ["Model", "Baseline_Model", "Model_with_feature_selection", "Baseline_Model_with_feature_selection"]:
            result_csv_writer.writerow(
                [key, output.get(key).get("Total_Accuracy"), output.get(key).get("Male_Accuracy"),
                 output.get(key).get("Female_Accuracy"),
                 abs(output.get(key).get("Eq_odds_tpr")),
                 abs(output.get(key).get("parity_diff"))])
print("Done")

