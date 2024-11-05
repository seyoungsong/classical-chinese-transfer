import os
from typing import Any

from sacrebleu.metrics.bleu import BLEU, BLEUScore, BLEUSignature
from sacrebleu.significance import PairedTest, Result


def compute_bleu_mt(
    hypo: list[str], ref1: list[str], lang: str, mode: str = "bleu"
) -> dict[str, Any]:
    assert mode in {"bleu", "spbleu"}, f"mode={mode}"
    lang2 = lang.lower().strip()
    if mode == "spbleu":
        bleu_obj = BLEU(tokenize="flores200")
    elif lang2 in {"ko", "oko", "cko"}:
        bleu_obj = BLEU(tokenize="ko-mecab")
    elif lang2 in {"hj", "zh", "lzh", "cc"}:
        bleu_obj = BLEU(tokenize="zh")
    elif lang2 in {"en"}:
        bleu_obj = BLEU(tokenize="13a")
    else:
        raise ValueError(f"lang={lang}")

    # compute BLEU
    score1: BLEUScore = bleu_obj.corpus_score(hypotheses=hypo, references=[ref1])
    signature1: BLEUSignature = bleu_obj.get_signature()  # type: ignore

    # output
    results = {
        "score": score1.score,
        "score.str": str(score1),
        "signature": str(signature1),
        "lang": lang,
        "num_sample": len(hypo),
        "hypo_len": sum(map(len, hypo)),
        "ref1_len": sum(map(len, ref1)),
    }

    return results


def compute_bleu_paired_bs(
    hyp1: list[str],  # baseline
    hyp2: list[str],
    ref1: list[str],
    lang: str,
    mode: str = "bleu",
    n_samples: int = 2000,
) -> dict[str, Any]:
    assert mode in {"bleu", "spbleu"}, f"mode={mode}"
    lang2 = lang.lower().strip()
    if mode == "spbleu":
        tokenize = "flores200"
    elif lang2 in {"ko", "oko", "cko"}:
        tokenize = "ko-mecab"
    elif lang2 in {"hj", "zh", "lzh", "cc"}:
        tokenize = "zh"
    elif lang2 in {"en"}:
        tokenize = "13a"
    else:
        raise ValueError(f"lang={lang}")
    metrics = {"BLEU": BLEU(tokenize=tokenize)}
    named_systems = [("hyp1", hyp1), ("hyp2", hyp2)]

    # compute
    os.environ["SACREBLEU_SEED"] = str(42)
    bs_scores = PairedTest(
        named_systems=named_systems,  # type: ignore
        metrics=metrics,
        references=[ref1],
        test_type="bs",
        n_samples=n_samples,
    )()
    signature1: BLEUSignature = bs_scores[0]["BLEU"]  # type: ignore
    score1: Result = bs_scores[1]["BLEU"][0]  # type: ignore
    score2: Result = bs_scores[1]["BLEU"][1]  # type: ignore

    # output
    results = {
        "score1": score1.score,
        "score1.p_value": score1.p_value,
        "score1.mean": score1.mean,
        "score1.ci": score1.ci,
        "score1.str": str(score1),
        "score2": score2.score,
        "score2.p_value": score2.p_value,
        "score2.p_value_is_significant": score2.p_value < 0.05,  # type: ignore
        "score2.mean": score2.mean,
        "score2.ci": score2.ci,
        "score2.str": str(score2),
        "signature": str(signature1),
        "lang": lang,
        "num_sample": len(hyp1),
        "hyp1_len": sum(map(len, hyp1)),
        "hyp2_len": sum(map(len, hyp2)),
        "ref1_len": sum(map(len, ref1)),
    }

    return results
