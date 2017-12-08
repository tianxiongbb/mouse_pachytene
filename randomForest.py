#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import bb_basic as bb
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
    for cutoff in [100, 500]:
        for features in [["cgContent","dnameInKitPlusSg","h3k27acInSg","h3k27me3InSg","h3k4me3InSg","h3k27acInSgAtGeneBody"], ["amybInTestis3monthOldDeniz","cgContent"], ["amybInTestis3monthOldDeniz","h3k27acInSg"], ["amybInTestis3monthOldDeniz","h3k27me3InSg"], ["amybInTestis3monthOldDeniz","h3k4me3InSg"], ["amybInTestis3monthOldDeniz","h3k27acInSgAtGeneBody"], ["amybInTestis3monthOldDeniz","cgContent","dnameInKitPlusSg","h3k27acInSg","h3k27me3InSg","h3k4me3InSg","h3k27acInSgAtGeneBody"], ["cgContent","h3k27acInSg"], ["h3k27me3InSg","h3k27acInSg"], ["h3k4me3InSg","h3k27acInSg"], ["h3k27acInSgAtGeneBody","h3k27acInSg"], ["amybInTestis3monthOldDeniz","cgContent","h3k27acInSg"], ["amybInTestis3monthOldDeniz","h3k27me3InSg","h3k27acInSg"], ["amybInTestis3monthOldDeniz","h3k4me3InSg","h3k27acInSg"], ["amybInTestis3monthOldDeniz","h3k27acInSgAtGeneBody","h3k27acInSg"], ["fivehmcInSc","cgContent","dnameInKitPlusSg","h3k27acInSg","h3k27me3InSg","h3k4me3InSg","h3k27acInSgAtGeneBody"], ["fivehmcInSc","cgContent","dnameInKitPlusSg","h3k27acInSg","h3k27me3InSg"], ["amybInTestis3monthOldDeniz","fivehmcInSc","cgContent","dnameInKitPlusSg","h3k27acInSg","h3k27me3InSg","h3k4me3InSg","h3k27acInSgAtGeneBody"], ["amybInTestis3monthOldDeniz","fivehmcInSc","cgContent","dnameInKitPlusSg","h3k27acInSg","h3k27me3InSg"], ["amybInTestis3monthOldDeniz","fivehmcInSc","cgContent"]]:
            run_randomForest(matrix, features, 10, 5, "+".join(features)+"_cutoff_"+str(cutoff))

# --------functions--------
def run_randomForest(matrix, features, rep_times, fold_times, file_prefix):
    # get index for train and test, and labels
    index = []
    labels = []
    for i in range(matrix.shape[0]):
        if(matrix.smallRNAInSt[i]>=cutoff):
            labels.append(1)
        else:
            labels.append(0)
        if(matrix.amybBindinTestis3monthOldDeniz[i]==1 and matrix.amybInTestis3monthOldDeniz[i]>3 and (matrix.log2FoldOfAmybMutVsAmybHetInP14[i]<(-1) or matrix.log2FoldOfAmybMutVsAmybHetInP17[i]<(-1))):
            index.append(i)
    # get training_set and test set for different k-fold
    labels = np.array(labels)
    k_fold = KFold(n_splits=fold_times, shuffle=True)
    k = 1
    oob_error = [] # oob error
    feature_imp = np.zeros(training_set.shape[1]) # feature importance
    for i1, i2 in k_fold.split(labels):
        training_index = np.array(index)[i1]
        test_index = np.array(index)[i2]
        training_set = matrix[features].loc(training_index)
        test_set = matrix[features].loc(training_index)
        training_labels = labels[training_index]
        test_labels = labels[test_index]
        # fit random forest
        test_result = np.zeros(len(test_labels)) # prediction of the probability
        for i in range(rep_times):
            rfc = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=16)
            rfc.fit(training_set, training_labels)
            oob_error.append(1-rfc.oob_score_)
            feature_imp = feature_imp + rfc.feature_importances_
            test_result = test_result + rfc.predict_proba(test_set)[:,1]
        # write scores and labels for test set in the k's fold validation
        file_sl = bb.fun_open_file(file_prefix+".SL", "w")
        for i in range(len(test_result)):
            file_sl.write("%s\t%s\t%s\n"%(matrix[geneName].loc[test_index[i]], test_result[i], test_labels[i]))
        # write PR curve
        #file_pr = bb.fun_open_file(file_prefix+".PR", "w")
        #precision, recall, _ = metrics.precision_recall_curve(test_labels, test_result)
        #pr_auc = metrics.auc(recall, precision)
        #for i in range(len(precision)):
        #    file_pr.write("%s\t%s\n"%(precision[i], recall[i]))
        #file_pr.close()
        #plt.step(recall, precision, color='steelblue', alpha=1,where='post')
        #plt.xlabel('Recall')
        #plt.ylabel('Precision')
        #plt.ylim([0.0, 1.05])
        #plt.xlim([0.0, 1.0])
        #plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(pr_auc))
        #plt.show()
        k += 1
    # write feature importance
    file_fi = bb.fun_open_file(file_prefix+".FI", "w")
    for i in range(training_set.shape[1]):
        file_fi.write(training_set.columns[i] + "\t" + str(feature_imp[i]/rep_times) + "\n")
    file_fi.close()
    # write out of bag error
    file_oobe = bb.fun_open_file(file_prefix+".OOBE", "w")
    file_oobe.write(str(sum(oob_error)/rep_times))
    # close scores and labels file
    file_sl.close()


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
