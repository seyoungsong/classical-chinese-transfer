from collections import Counter

import numpy as np


def _ruzicka_similarity(x: list[int], y: list[int]) -> float:
    # Ruzicka similarity (Weighted Jaccard similarity)
    # x, y: predicted and ground-truth labels of the same token
    # represented by a vector consisting of the count of
    # all single-character punctuation marks in that label
    assert len(x) == len(y), "err-025: len mismatch"
    rs: float = np.sum(np.minimum(x, y)) / np.sum(np.maximum(x, y))
    return rs


def _encode_vector(s: str, punc: str) -> tuple[list[str], list[list[int]]]:
    # remove starting punc
    s1 = s.lstrip(punc)

    # init
    tokens: list[str] = []
    pmarks: list[str] = []

    # interate
    i = 0
    while i < len(s1):
        if s1[i] not in punc:
            tokens.append(s1[i])
            pmarks.append("")
            i += 1
        else:
            pmarks[-1] += s1[i]
            i += 1
    # check
    assert len(tokens) == len(pmarks), "err-046: len mismatch"
    assert (
        "".join(["".join(t) for t in zip(tokens, pmarks, strict=True)]) == s1
    ), "err-047: str bad"

    # label: punc count vector
    counters = [Counter(s) for s in pmarks]
    labels = [[counter[p] for p in punc] for counter in counters]

    return tokens, labels


def compute_mRS(
    hypo: list[str], ref1: list[str], hypo_punc: str, ref1_punc: str
) -> float:
    # check
    assert len(hypo) == len(ref1), "err-060: len mismatch"

    punc = "".join(sorted(set(ref1_punc + hypo_punc)))

    # encode
    hypo_encode = [_encode_vector(s=s, punc=punc) for s in hypo]
    ref1_encode = [_encode_vector(s=s, punc=punc) for s in ref1]

    # check
    for _i, hr in enumerate(zip(hypo_encode, ref1_encode, strict=True)):
        h, r = hr
        assert "".join(h[0]) == "".join(r[0]), "err-074: tokens mismatch"

    # labels
    hypo_labels = [labels for tokens, labels in hypo_encode]
    ref1_labels = [labels for tokens, labels in ref1_encode]

    # compute
    sim_list = []
    for sent_hl, sent_rl in zip(hypo_labels, ref1_labels, strict=True):
        for char_hl, char_rl in zip(sent_hl, sent_rl, strict=True):
            if sum(char_rl) == 0:
                continue
            sim1 = _ruzicka_similarity(x=char_hl, y=char_rl)
            sim_list.append(sim1)
    mean_sim = float(round(np.mean(sim_list) * 100, 2))
    return mean_sim
