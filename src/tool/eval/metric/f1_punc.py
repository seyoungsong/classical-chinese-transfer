import random
from typing import Any

import pandas as pd
from seqeval.metrics import classification_report

import src.tool.eval as etool
from src import utils


def compute_punc_f1(
    hypo: list[str], ref1: list[str], mode: str = "reduce"
) -> dict[str, Any]:
    # check
    assert len(hypo) == len(ref1), "len not match"
    if 0:
        utils.temp_diff("\n\n".join(hypo), "\n\n".join(ref1))

    if 0:
        punc_xml = random.choice(hypo)
        punc_xml
        etool.punc_xml2punc_iob2(punc_xml)
        punc_xml2 = etool.punc_xml_reduce(punc_xml)
        if punc_xml != punc_xml2:
            utils.temp_diff(punc_xml, punc_xml2)

    # reduce if needed
    if mode == "reduce":
        hyp2 = [etool.punc_xml_reduce(punc_xml=s) for s in hypo]
        ref2 = [etool.punc_xml_reduce(punc_xml=s) for s in ref1]
    elif mode == "exact":
        hyp2, ref2 = hypo, ref1
    else:
        raise ValueError(f"mode {mode} not supported")
    if 0:
        utils.temp_diff("\n\n".join(hyp2), "\n\n".join(ref2))
        utils.temp_diff("\n\n".join(hypo), "\n\n".join(hyp2))

    # convert
    if 0:
        _i = random.choice(range(len(hyp2)))
        h1, r1 = hyp2[_i], ref2[_i]
    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []
    for h1, r1 in zip(hyp2, ref2, strict=True):
        ref_toks, ref_tags = etool.punc_xml2punc_iob2(r1)
        hyp_toks, hyp_tags = etool.punc_xml2punc_iob2(h1)
        assert "".join(ref_toks) == "".join(hyp_toks), "tokens not match"
        assert len(ref_tags) == len(hyp_tags), "len not match"
        y_true.append(ref_tags)
        y_pred.append(hyp_tags)

    report1 = classification_report(
        y_true=y_true, y_pred=y_pred, output_dict=True, zero_division=0
    )
    results: dict[str, Any] = pd.json_normalize(report1).iloc[0].to_dict()
    results["num_sample"] = len(hypo)
    results["f1"] = results["weighted avg.f1-score"]

    return results
