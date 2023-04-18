######################################################
# Evaluation metrics
######################################################

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    multilabel_confusion_matrix,
)
from scipy import stats
from typing import List


def plex_evaluate(
    preds: np.ndarray,
    target_labels: np.ndarray,
    eval_args: dict,
    classes: List[str],
    meta_data: pd.DataFrame = None,
    underdiagnosis_label: str = None,
) -> dict:
    """Evaluate the predictions of a model using the metrics from the PLEX paper.

    Args:
       preds_probs (np.ndarray): The predicted probabilities.
       preds_labels (np.ndarray): The predicted labels.
       target_labels (np.ndarray): The true labels.
       eval_args (dict): The evaluation arguments.
       meta_data (pd.DataFrame, optional): The meta data. Defaults to None.
       multilabel (bool): If multilabel task or multiclass if not. Defaults to True.

    Returns:
       dict: The evaluation metrics.
    """

    metrics = {}
    multilabel = True
    preds_probs = preds
    if multilabel:
        preds_labels = np.array([[1 if p_ > eval_args["decision_threshold"] else 0 for p_ in p] for p in preds])
    else:
        preds_labels = np.argmax(preds, axis=1)

    # Calculate the random guessing accuracy
    distribution_per_class = [
        (sum(target_labels[:, c]) / len(target_labels))
        for c in range(len(target_labels[0]))
    ]
    acc_random_per_class = [1 - d for d in distribution_per_class]

    # Uncertainty metrics
    metrics["auc"] = metric_AUROC(target_labels, preds)
    metrics["avg_auc_prob"] = round(np.mean(metric_AUROC(target_labels, preds)), 4)
    metrics["oracle_auc"] = round(
        selective_prediction(
            preds_probs,
            preds_labels,
            target_labels,
            eval_args["selective_threshold"],
            "auc",
            "oracle",
            multilabel,
        ),
        4,
    )
    metrics["rejection_auc"] = round(
        selective_prediction(
            preds_probs,
            preds_labels,
            target_labels,
            eval_args["selective_threshold"],
            "auc",
            "rejection",
            multilabel,
        ),
        4,
    )
    metrics["calibration_error"] = round(
        expected_calibration_error(
            preds_probs=preds_probs,
            preds_labels=preds_labels,
            target_labels=target_labels,
            num_bins=eval_args["ece_num_bins"],
        ),
        4,
    )

    # Adaptation metrics
    metrics["avg_prob_per_class"] = [
        round(float(e), 4) for e in np.mean(preds_probs, axis=0)
    ]
    metrics["acc_per_class"] = [
        round(c, 4) for c in acc_per_class(target_labels, preds_labels, multilabel)
    ]
    metrics["acc_random_guessing_per_class"] = [
        round(a, 4) for a in acc_random_per_class
    ]
    metrics["disparity"] = round(
        disparity_score(target_labels, preds_labels, multilabel), 4
    )

    # Robust generalisation metrics
    metrics["subset_acc"] = round(accuracy_score(target_labels, preds_labels), 4)
    metrics["mean_acc"] = round(np.mean(metrics["acc_per_class"]), 4)
    metrics["acc_random_guessing"] = round(np.mean(acc_random_per_class), 4)
    metrics["err"] = round(f1_score(target_labels, preds_labels, average="macro"), 4)


    # Subpopulation metrics
    print("Calculating subpopulation metrics...")
    if meta_data is not None:
        if eval_args["independent_reg_variable"] in meta_data.columns:
            metrics["temporal_err"] = regression_metrics(
                target_labels,
                preds_labels,
                meta_data[eval_args["independent_reg_variable"]],
                "err",
            )
        metrics["subpopulation_err"] = {
            g: {
                str(c): round(
                    f1_score(
                        target_labels[meta_data[g] == c],
                        preds_labels[meta_data[g] == c],
                        average="macro",
                    ),
                    4,
                )
                for c in meta_data[g].unique()
                if c is not np.nan
            }
            for g in eval_args["subpopulation_groups"]
        }
        if underdiagnosis_label is not None:
            metrics["underdiagnosis"] = {
                g: {
                    str(c): round(
                        underdiagnosis(
                            target_labels[meta_data[g] == c],
                            preds_labels[meta_data[g] == c],
                            classes=classes,
                            underdiagnosis_label=underdiagnosis_label,
                        ),
                        4,
                    )
                    for c in meta_data[g].unique()
                    if c is not np.nan
                }
                for g in eval_args["subpopulation_groups"]
            }

    return metrics


def metric_AUROC(target, output):
    outAUROC = []

    nb_classes = output.shape[-1]

    if not isinstance(target, np.ndarray):
        target = target.cpu().numpy()
    if not isinstance(output, np.ndarray):
        output = output.cpu().numpy()

    for i in range(nb_classes):
        if target[:, i].sum() != 0:
            outAUROC.append(roc_auc_score(target[:, i], output[:, i]))
        else:
            outAUROC.append(0)

    return np.array(outAUROC)


def underdiagnosis(
    ytrue: np.ndarray, ypred: np.ndarray, classes: list, underdiagnosis_label: str
) -> float:
    """Calculate underdisagnosis score as false positive rate for class "No Finding" in classes.

    Args:
       ytrue (np.ndarray): The true labels.
       ypred (np.ndarray): The predicted labels.
       classes (list): The classes.
       underdiagnosis_label (str): Underdiagnosis label

    Returns:
       float: The underdiagnosis score.
    """

    # Calculate all false positive scores for all classes:
    fp = np.sum((ypred == 1) & (ytrue == 0), axis=0)

    # Select fp score for class "No Finding" from classes:
    fp_no_finding = fp[classes.index(underdiagnosis_label)]

    # Calculate underdiagnosis score:
    underdiagnosis = fp_no_finding / np.sum(ytrue == 0)

    return underdiagnosis


def acc_per_class(ytrue: np.ndarray, ypred: np.ndarray, multilabel: bool) -> np.ndarray:
    """Calculate the accuracy per class.

    Args:
       ytrue (np.ndarray): The true labels.
       ypred (np.ndarray): The predicted labels.
       multilabel (bool): Whether multilabel or multiclass predictions as supplied.

    Returns:
       np.ndarray: The accuracy per class.
    """

    if multilabel:
        cm = multilabel_confusion_matrix(ytrue, ypred)
        acc_per_class = [c.diagonal().sum() / c.sum() for c in cm]
    else:
        cm = confusion_matrix(ytrue, ypred)
        acc_per_class = cm.diagonal() / cm.sum(axis=1)

    return acc_per_class


def disparity_score(ytrue: np.ndarray, ypred: np.ndarray, multilabel: bool) -> float:
    """Difference of the maximum and minimum accuracies across all classes (Disp)

    Args:
       ytrue (np.ndarray): The true labels.
       ypred (np.ndarray): The predicted labels.
       multilabel (bool): Whether multilabel or multiclass predictions as supplied.

    Returns:
       float: Disparity score.
    """

    all_acc = acc_per_class(ytrue, ypred, multilabel)
    return max(all_acc) - min(all_acc)


def selective_prediction(
    preds_probs: np.ndarray,
    preds_labels: np.ndarray,
    target_labels: np.ndarray,
    threshold: float,
    metric: str,
    method: str,
    multilabel: bool,
) -> float:
    """Calculate the selective prediction metric for a given set of predictions and targets.
    The selective prediction metric is calculated as the accuracy, error or AUC of the predictions for which the maximum probability is above a given threshold.
    The metric can be "acc", "err" or "auc".

    Args:
       preds_probs (np.ndarray): The predicted probabilities.
       preds_labels (np.ndarray): The predicted labels.
       target_labels (np.ndarray): The target labels.
       threshold (float): The threshold for the maximum probability.
       metric (str): The metric to use for the selective prediction. Can be "acc", "err" or "auc".
       method (str): The method to use for the selective prediction. Can be "oracle" or "rejection".
       multilabel (bool): If multilabel task or multiclass if not. Defaults to True.

    Returns:
       float: The selective prediction metric.

    Raises:
       ValueError: If the metric or method is unknown.
    """

    no_values = int(len(preds_probs) * threshold)
    if multilabel:
        idx_most_uncertain = np.argpartition(
            [min(o) for o in abs(preds_probs - 0.5)], no_values
        )[:no_values]
    else:
        idx_most_uncertain = np.argpartition([max(o) for o in preds_probs], no_values)[
            :no_values
        ]

    if method == "oracle":
        preds_labels[idx_most_uncertain] = target_labels[idx_most_uncertain]
        preds_probs[idx_most_uncertain] = target_labels[idx_most_uncertain]
    elif method == "rejection":
        preds_labels = [
            preds_labels[i]
            for i in range(len(preds_labels))
            if i not in idx_most_uncertain
        ]
        preds_probs = [
            preds_probs[i]
            for i in range(len(preds_probs))
            if i not in idx_most_uncertain
        ]
        target_labels = [
            target_labels[i]
            for i in range(len(target_labels))
            if i not in idx_most_uncertain
        ]
    else:
        raise ValueError("Unknown method: {}".format(method))

    if metric == "acc":
        value = accuracy_score(target_labels, preds_labels)
    elif metric == "err":
        value = f1_score(target_labels, preds_labels, average="macro")
    elif metric == "auc":
        value = roc_auc_score(
            target_labels, preds_probs, average="macro", multi_class="ovr"
        )
    else:
        raise ValueError("Unknown metric: {}".format(metric))

    return value


def regression_metrics(
    target_labels: np.ndarray,
    preds_labels: np.ndarray,
    independent_reg_variable: pd.Series,
    metric: str,
) -> dict:
    """Calculate the regression metrics for a given set of predictions and targets.
    The regression metrics are calculated as the slope and p-value of a linear regression with the independent_reg_variable as independent variable and the metric as dependent variable.
    The metric can be "acc", "err" or "auc".

    Args:
       target_labels (np.ndarray): The target labels.
       preds_labels (np.ndarray): The predicted labels.
       independent_reg_variable (pd.Series): The independent variable for the regression.
       metric (str): The metric to use for the regression. Can be "acc", "err" or "auc".

    Returns:
       dict: The regression metrics.

    Raises:
       ValueError: If the metric is unknown.
    """

    # Calculate the metric for each value of the independent variable
    independent_values = independent_reg_variable.unique()
    metric_values = {}
    for value in independent_values:
        if metric == "acc":
            metric_values[value] = accuracy_score(
                target_labels[independent_reg_variable == value],
                preds_labels[independent_reg_variable == value],
            )
        elif metric == "err":
            metric_values[value] = f1_score(
                target_labels[independent_reg_variable == value],
                preds_labels[independent_reg_variable == value],
                average="macro",
            )
        elif metric == "auc":
            metric_values[value] = roc_auc_score(
                target_labels[independent_reg_variable == value],
                preds_labels[independent_reg_variable == value],
                average="macro",
                multi_class="ovr",
            )
        else:
            raise ValueError("Unknown metric: {}".format(metric))

    # Calculate regression slope and p-value from regression with metric.keys() as independent variable and metric.values() as dependent variable
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        list(metric_values.keys()), list(metric_values.values())
    )

    return {
        "intercept": round(intercept, 4),
        "slope": round(slope, 4),
        "p_value": round(p_value, 4),
        "r_value": round(r_value, 4),
    }


def expected_calibration_error(
    preds_probs: np.ndarray,
    preds_labels: np.ndarray,
    target_labels: np.ndarray,
    num_bins: int,
) -> float:
    """Calculate the expected calibration error (ECE) for a given set of predictions and targets.
    The ECE is calculated as the weighted accuracy of the differences between the mean confidence and mean accuracy per bin, weighted by the number of samples in the bin.
    The ECE is calculated for each label separately and the average is returned.

    Args:
       preds_probs (np.ndarray): The predicted probabilities for each label.
       preds_labels (np.ndarray): The predicted labels.
       target_labels (np.ndarray): The target labels.
       num_bins (int): The number of bins to use.

    Returns:
       float: The expected calibration error.
    """

    # Loop through each label
    no_labels = len(target_labels[0])
    eces = []
    for i in range(no_labels):
        # Create bins
        bins = {
            start_value: {"prob": [], "acc": []}
            for start_value in np.linspace(0, 1, num_bins + 1)[:-1]
        }

        # Loop through each prediction and add to the correct bin the probability and accuracy
        for prob, pred, target in zip(
            preds_probs[:, i], preds_labels[:, i], target_labels[:, i]
        ):
            for start_value in bins.keys():
                if prob >= start_value and prob < start_value + 1 / num_bins:
                    bins[start_value]["prob"].append(prob)
                    bins[start_value]["acc"].append(int(pred == target))
                    break

        # Calculate the difference between the mean confidence and mean accuracy
        for bin, bin_dict in bins.items():
            bins[bin]["difference"] = np.abs(
                np.mean(bin_dict["prob"]) - np.mean(bin_dict["acc"])
            )

        # Calculate the ECE as weighted accuracy of the differences between the mean confidence and mean accuracy per bin, weighted by the number of samples in the bin
        eces.append(
            np.sum(
                [
                    bin_dict["difference"] * len(bin_dict["prob"])
                    for bin_dict in bins.values()
                    if abs(bin_dict["difference"]) > 0
                ]
            )
            / len(preds_probs)
        )

    return np.mean(eces)
