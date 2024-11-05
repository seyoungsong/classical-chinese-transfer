from typing import Any

from sacrebleu.metrics.bleu import BLEU, BLEUScore, BLEUSignature


def fix_flores200() -> None:
    cmd = '''mkdir -p ~/.sacrebleu/models && wget -O ~/.sacrebleu/models/flores200sacrebleuspm "https://dl.fbaipublicfiles.com/large_objects/nllb/models/spm_200/flores200_sacrebleu_tokenizer_spm.model"'''
    print(cmd)


def compute_BLEU(hypo: list[str], ref1: list[str], lang: str) -> dict[str, Any]:
    lang2 = lang.lower().strip()
    if lang2 in {"ko", "oko", "cko"}:
        bleu_obj = BLEU(tokenize="ko-mecab")
    elif lang2 in {"hj", "zh", "lzh", "cc"}:
        bleu_obj = BLEU(tokenize="zh")
    elif lang2 in {"en"}:
        bleu_obj = BLEU(tokenize="13a")
    else:
        raise ValueError(f"lang={lang}")

    # compute BLEU
    BLEU_score_obj: BLEUScore = bleu_obj.corpus_score(
        hypotheses=hypo, references=[ref1]
    )
    BLEU_signature: BLEUSignature = bleu_obj.get_signature()  # type: ignore
    BLEU_score = round(BLEU_score_obj.score, 2)

    # compute spBLEU
    spBLEU_obj = BLEU(tokenize="flores200")
    spBLEU_score_obj: BLEUScore = spBLEU_obj.corpus_score(
        hypotheses=hypo, references=[ref1]
    )
    spBLEU_signature: BLEUSignature = spBLEU_obj.get_signature()  # type: ignore
    spBLEU_score = round(spBLEU_score_obj.score, 2)

    # output
    result = {
        "BLEU_score": BLEU_score,
        "BLEU_score_raw": str(BLEU_score_obj),
        "BLEU_signature": str(BLEU_signature),
        "spBLEU_score": spBLEU_score,
        "spBLEU_score_raw": str(spBLEU_score_obj),
        "spBLEU_signature": str(spBLEU_signature),
        "lang": lang,
        "num_sample": len(hypo),
        "hypo_len": sum(map(len, hypo)),
        "ref1_len": sum(map(len, ref1)),
    }

    return result
