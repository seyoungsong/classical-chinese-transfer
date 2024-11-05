import unicodedata
from typing import Any

import evaluate
from evaluate.module import EvaluationModule

_SEQEVAL: Any = None


def _char2iob(c: str) -> str:
    if c == "":
        return "O"
    c_name = unicodedata.name(c).strip().title().replace(" ", "")
    iob_label = f"B-{c_name}"
    return iob_label


def _encode_labels(s: str, punc: str) -> tuple[list[str], list[str]]:
    tokens: list[str] = []
    pmarks: list[str] = []
    n = len(s)
    for i in range(n):
        c_curr = s[i]
        c_next = s[i + 1] if i + 1 < n else None
        if c_curr not in punc:
            tokens.append(c_curr)
            if c_next is not None and c_next in punc:
                pmarks.append(c_next)
            else:
                pmarks.append("")
    assert len(tokens) == len(pmarks), "err-261"
    labels = [_char2iob(c) for c in pmarks]

    return tokens, labels


def old_compute_punc_f1(hypo: list[str], ref1: list[str], punc: str) -> dict[str, Any]:
    # encode to IOB2 labels
    hypo_labels = [_encode_labels(s=s, punc=punc)[1] for s in hypo]
    ref1_labels = [_encode_labels(s=s, punc=punc)[1] for s in ref1]

    # check
    for i, l1l2 in enumerate(zip(hypo_labels, ref1_labels)):
        l1, l2 = l1l2
        assert len(l1) == len(l2), f"len mismatch: {len(l1)} vs {len(l2)} ({i})"
    if 0:
        etool: Any = None
        s1 = hypo[i]
        s2 = ref1[i]
        _encode_labels(s=s1, punc=punc)[0]
        _encode_labels(s=s2, punc=punc)[0]
        etool.remove_punc(s=hypo[i], punc=punc)
        etool.remove_punc(s=ref1[i], punc=punc)
        hypo_labels[i]
        ref1_labels[i]

    # lazy load
    global _SEQEVAL
    if _SEQEVAL is None:
        _SEQEVAL = evaluate.load("seqeval")
    assert isinstance(_SEQEVAL, EvaluationModule), "seqeval load error"

    # compute
    results_seqeval: dict[str, Any] = _SEQEVAL.compute(
        predictions=hypo_labels, references=ref1_labels, zero_division=0
    )

    # change format
    results: dict[str, float] = {}
    label_names = [s for s in results_seqeval.keys() if "overall_" not in s]
    label_name = label_names[0]
    for label_name in label_names:
        d1 = results_seqeval[label_name]
        assert isinstance(d1, dict)
        d2 = {f"{label_name}_{s}": i for s, i in d1.items()}
        results.update(d2)
    d3 = {k: v for k, v in results_seqeval.items() if "overall_" in k}
    results.update(d3)
    results["overall_number"] = len(hypo)

    # *100 and round 2
    results = {
        k: (
            round(v * 100, 2)
            if k.split("_")[-1] in {"f1", "precision", "recall", "accuracy"}
            else v
        )
        for k, v in results.items()
    }

    return results
