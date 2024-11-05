from typing import Any

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_F1_sklearn(hypo: list[str], ref1: list[str]) -> dict[str, Any]:
    # check
    assert len(hypo) == len(ref1), "len not match"

    acc1 = accuracy_score(y_true=ref1, y_pred=hypo)
    rec1 = recall_score(y_true=ref1, y_pred=hypo, average="weighted")
    pre1 = precision_score(
        y_true=ref1, y_pred=hypo, average="weighted", zero_division=0
    )
    f1 = f1_score(y_true=ref1, y_pred=hypo, average="weighted")
    qwk1 = cohen_kappa_score(y1=ref1, y2=hypo, weights="quadratic")
    results = {
        "overall_accuracy": acc1,
        "overall_precision": pre1,
        "overall_recall": rec1,
        "overall_f1": f1,
        "overall_qwk": qwk1,
    }
    results["overall_number"] = len(hypo)

    # *100 and round 2
    results = {
        k: (
            round(v * 100, 2)
            if k.split("_")[-1] in {"f1", "precision", "recall", "accuracy", "qwk"}
            else v
        )
        for k, v in results.items()
    }

    return results
