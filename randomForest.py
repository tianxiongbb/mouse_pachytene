#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import bb_basic as bb
from random import sample
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def main():
    print_help()
    matrix = pd.read_csv(sys.argv[1], delimiter="\t")
    # fill na in h3k27ac and pol2 at genebody with mean values
    matrix = matrix.fillna(matrix.mean()["h3k27acInSgAtGeneBody":"pol2InTestisAtGeneBody"])
    # fill na in methylation with 90
    matrix = matrix.fillna(90)
    for cutoff in [100, 500, 1000, 3000]:
        for features in [["cgContent","dnameInKitPlusSg","h3k27acInSg","h3k27me3InSg","h3k4me3InSg","h3k27acInSgAtGeneBody"], ["cgContent"], ["dnameInKitPlusSg"], ["h3k27acInSg"], ["h3k27me3InSg"], ["h3k4me3InSg"], ["h3k27acInSgAtGeneBody"], ["amybInTestis3monthOldDeniz"], ["amybInTestis3monthOldDeniz","cgContent"], ["amybInTestis3monthOldDeniz","h3k27acInSg"], ["amybInTestis3monthOldDeniz","h3k27me3InSg"], ["amybInTestis3monthOldDeniz","h3k4me3InSg"], ["amybInTestis3monthOldDeniz","h3k27acInSgAtGeneBody"], ["amybInTestis3monthOldDeniz","cgContent","dnameInKitPlusSg","h3k27acInSg","h3k27me3InSg","h3k4me3InSg","h3k27acInSgAtGeneBody"], ["cgContent","h3k27acInSg"], ["h3k27me3InSg","h3k27acInSg"], ["h3k4me3InSg","h3k27acInSg"], ["h3k27acInSgAtGeneBody","h3k27acInSg"], ["amybInTestis3monthOldDeniz","cgContent","h3k27acInSg"], ["amybInTestis3monthOldDeniz","h3k27me3InSg","h3k27acInSg"], ["amybInTestis3monthOldDeniz","h3k4me3InSg","h3k27acInSg"], ["amybInTestis3monthOldDeniz","h3k27acInSgAtGeneBody","h3k27acInSg"]]:
            training_set, training_labels, test_set, test_labels = get_training_set(matrix, cutoff, features)
            training_set = training_set.fillna(90)
            test_set = test_set.fillna(90)
            run_randomForest(training_set, training_labels, test_set, test_labels, 25, "+".join(features)+"_cutoff_"+str(cutoff))

# --------functions--------
def run_randomForest(training_set, training_labels, test_set, test_labels, rep_times, file_prefix):
    oob_error = []
    feature_imp = np.zeros(training_set.shape[1])
    test_result = np.zeros(len(test_labels))
    for i in range(rep_times):
        rfc = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=16)
        rfc.fit(training_set, training_labels)
        oob_error.append(1-rfc.oob_score_)
        feature_imp = feature_imp + rfc.feature_importances_
        test_result = test_result + rfc.predict_proba(test_set)[:,1]
    # write feature importance
    file_fi = bb.fun_open_file(file_prefix+".FI", "w")
    for i in range(training_set.shape[1]):
        file_fi.write(training_set.columns[i] + "\t" + str(feature_imp[i]/rep_times) + "\n")
    file_fi.close()
    # write out of bag error
    file_oobe = bb.fun_open_file(file_prefix+".OOBE", "w")
    file_oobe.write(str(sum(oob_error)/rep_times))
    # write scores and label for test set
    file_sl = bb.fun_open_file(file_prefix+".SL", "w")
    for i in range(len(test_result)):
        file_sl.write("%s\t%s\n"%(test_result[i], test_labels[i]))
    file_sl.close()
    # write PR curve
    file_pr = bb.fun_open_file(file_prefix+".PR", "w")
    precision, recall, _ = metrics.precision_recall_curve(test_labels, test_result)
    pr_auc = metrics.auc(recall, precision)
    for i in range(len(precision)):
        file_pr.write("%s\t%s\n"%(precision[i], recall[i]))
    file_pr.close()
    #plt.step(recall, precision, color='steelblue', alpha=1,where='post')
    #plt.xlabel('Recall')
    #plt.ylabel('Precision')
    #plt.ylim([0.0, 1.05])
    #plt.xlim([0.0, 1.0])
    #plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(pr_auc))
    #plt.show()


def get_training_set(matrix, cutoff, features):
    index = []
    labels = []
    for i in range(matrix.shape[0]):
        if(matrix.amybBindinTestis3monthOldDeniz[i]==1 and matrix.amybInTestis3monthOldDeniz[i]>3 and (matrix.log2FoldOfAmybMutVsAmybHetInP14[i]<(-1) or matrix.log2FoldOfAmybMutVsAmybHetInP17[i]<(-1))):
            index.append(i)
            if(matrix.smallRNAInSt[i]>=cutoff):
                labels.append(1)
            else:
                labels.append(0)
    training_index = range(0, len(labels), 2)
    test_index = list(set(range(len(labels))).difference(set(training_index)))
    training_labels = np.array(labels)[training_index]
    test_labels = np.array(labels)[test_index]
    test_set = matrix.loc[np.array(index)[test_index]]
    training_set = matrix.loc[np.array(index)[training_index]]
    return training_set[features], training_labels, test_set[features], test_labels

    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version="%(prog)s 1.0")
    parser.add_argument("args", help="", nargs="*")
    args = parser.parse_args()
    return args


def print_help():
    if len(sys.argv) < 2:
        bb.fun_print_help("in.txt", "rpm_cutoff_for_piRNA_gene")

# --------process--------
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        bb.fun_print_error("user interrupted, abort!")
        sys.exit(0)
