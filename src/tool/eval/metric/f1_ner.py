import random
from typing import Any

import pandas as pd
from seqeval.metrics import classification_report

import src.tool.eval as etool
from src import utils


def old_compute_F1_ner(
    hypo: list[str],
    ref1: list[str],
    hypo_label_map: dict[str, str],
    ref1_label_map: dict[str, str],
) -> dict[str, Any]:
    # check
    assert len(hypo) == len(ref1), "len not match"
    if 0:
        utils.temp_diff("\n\n".join(hypo), "\n\n".join(ref1))

    # convert per, loc to other
    if 0:
        s_xml = random.choice(hypo)
        s_xml
        etool.xml_change_label(s_xml=s_xml, label_map=hypo_label_map)
    hyp2 = [etool.xml_change_label(s_xml=s, label_map=hypo_label_map) for s in hypo]
    ref2 = [etool.xml_change_label(s_xml=s, label_map=ref1_label_map) for s in ref1]
    if 0:
        utils.temp_diff("\n\n".join(hyp2), "\n\n".join(ref1))
        utils.temp_diff("\n\n".join(hypo), "\n\n".join(hyp2))

    # convert
    if 0:
        _i = random.choice(range(len(hyp2)))
        h1, r1 = hyp2[_i], ref2[_i]
    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []
    for h1, r1 in zip(hyp2, ref2, strict=True):
        ref_toks, ref_tags = etool.xml2iob(r1)
        hyp_toks, hyp_tags = etool.xml2iob(h1)
        assert "".join(ref_toks) == "".join(hyp_toks), "tokens not match"
        assert len(ref_tags) == len(hyp_tags), "len not match"
        y_true.append(ref_tags)
        y_pred.append(hyp_tags)

    report1 = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
    results: dict[str, Any] = pd.json_normalize(report1).iloc[0].to_dict()
    results["overall_number"] = len(hypo)
    results["overall_f1"] = results["weighted avg.f1-score"]

    return results


def compute_ner_f1(
    hypo: list[str], ref1: list[str], mode: str = "binary"
) -> dict[str, Any]:
    # check
    assert len(hypo) == len(ref1), "len not match"
    if 0:
        utils.temp_diff("\n\n".join(hypo), "\n\n".join(ref1))

    label_map: dict[str, str]
    if mode == "binary":
        # binary ner: convert all to other
        label_map = {
            "ajd_location": "other",
            "ajd_other": "other",
            "ajd_person": "other",
            "klc_other": "other",
            "wyweb_bookname": "other",
            "wyweb_other": "other",
            "other": "other",
        }
    elif mode == "entity":
        # entity ner: convert all to other except per, loc
        label_map = {
            "ajd_location": "location",
            "ajd_person": "person",
            "ajd_other": "other",
            "klc_other": "other",
            "wyweb_bookname": "other",
            "wyweb_other": "other",
            "other": "other",
        }
    else:
        raise ValueError(f"mode={mode} not supported")

    if 0:
        s_xml = random.choice(hypo)
        s_xml
        etool.xml_change_label(s_xml=s_xml, label_map=label_map)

    # convert per, loc to other
    hyp2 = [etool.xml_change_label(s_xml=s, label_map=label_map) for s in hypo]
    ref2 = [etool.xml_change_label(s_xml=s, label_map=label_map) for s in ref1]
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
        ref_toks, ref_tags = etool.xml2iob(r1)
        hyp_toks, hyp_tags = etool.xml2iob(h1)
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
