import random
import unicodedata
from collections import Counter

import pandas as pd
from tqdm import tqdm

from src import utils

# punct reduction rules & simple label precedence
REDUCTION_RULE: dict[str, str] = utils.read_json2(utils.PUNC_REDUCTION_JSON)
PRECEDENCE_RULE: list[str] = utils.read_json2(utils.PUNC_PRECEDENCE_JSON)


# example global punc set from punc_selection.json
__corpus_punc: dict[str, str] = utils.read_json2(utils.PUNC_CORPUS_JSON)
GLOBAL_PUNC = "".join(sorted(set("".join(list(__corpus_punc.values())))))
REDUCE4_PUNC = ",?。"


def edit_punc_for_google(s: str) -> str:
    s = s.replace(" ", "[S]")
    s = s.replace("'", "ʼ")
    s = s.replace('"', "ˮ")
    return s


def name_punc_label(label1: str) -> str:
    if 0:
        label1 = ",!"
    names = [unicodedata.name(c).title().replace(" ", "") for c in label1]
    name1 = "_".join(names)
    return name1


def count_punc_label(
    texts: list[str], not_punc: str, ignore_whites: bool
) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for s1 in tqdm(texts, desc="count_punc_label"):
        if s1 is None:
            continue
        if ignore_whites:
            s2 = utils.remove_whites(s1)
        else:
            s2 = utils.squeeze_whites(s1)
        items = utils.chunk_by_classifier(
            s=s2, f=lambda x: utils.is_punc_unicode(c=x, not_punc=not_punc)
        )
        labels = [d["text"] for d in items if d["label"] is True]
        counter.update(labels)
    df_label = pd.DataFrame(counter.most_common(), columns=["label", "count"])
    return df_label


def old_remove_punc(s: str, punc: str = GLOBAL_PUNC) -> str:
    trans_table = str.maketrans({c: "" for c in punc})
    return s.translate(trans_table)


def remove_punc(s: str) -> str:
    return "".join(c for c in s if not utils.is_punc_unicode(c))


def separate_punc(
    s: str, punc: str, trim_left: bool = False
) -> tuple[list[str], list[str]]:
    if 0:
        s = "hello, world!"
        s = "..안녕!, 세계???  test1..? test2, dfdfhe ."
        punc = GLOBAL_PUNC + "."

    # separate
    tokens: list[str] = []
    pmarks: list[str] = []
    for i in range(len(s)):
        c = s[i]
        if c not in punc:
            tokens.append(c)
            pmarks.append("")
        elif c in punc and i == 0:
            tokens.append("")
            pmarks.append(c)
        else:
            pmarks[-1] += c

    # check
    assert len(tokens) == len(pmarks), "len not same!"
    s_reconst = "".join(["".join(t) for t in zip(tokens, pmarks, strict=True)])
    assert s == s_reconst, "reconst not equal"

    # trim
    if trim_left:
        tokens, pmarks = _trim_left(tokens=tokens, pmarks=pmarks)

    return tokens, pmarks


def separate_pred(h1: str, z1: str) -> tuple[list[str], list[str]]:
    if 0:
        h1 = "癸未/上命召對, 講《詩傳。"
        z1 = "癸未上命召對講詩傳"

    # check
    assert utils.is_subset_with_count(z1, h1), "not subset"

    # separate
    tokens: list[str] = []
    pmarks: list[str] = []
    z1_copy = z1
    for c in h1:
        if z1_copy.startswith(c):
            tokens.append(c)
            pmarks.append("")
            z1_copy = z1_copy[1:]
        else:
            pmarks[-1] += c

    # check
    assert len(tokens) == len(pmarks), "len not same!"
    s_reconst = "".join(["".join(t) for t in zip(tokens, pmarks, strict=True)])
    assert h1 == s_reconst, "reconst not equal"

    return tokens, pmarks


def _trim_left(tokens: list[str], pmarks: list[str]) -> tuple[list[str], list[str]]:
    # remove starting puncts since they cannot be predicted)
    for i in range(len(tokens)):
        if tokens[i] != "":
            break
    tokens = tokens[i:]
    pmarks = pmarks[i:]
    return tokens, pmarks


def reduce4_punc(
    s: str,
    punc: str = GLOBAL_PUNC,
    reduction_rule: dict[str, str] = REDUCTION_RULE,
) -> str:
    if 0:
        s = "?。戊寅。/詣『孝禧』殿,,、 行端午祭。。?"
        s = "".join(random.sample(s, len(s)))
        punc = " ,。\":、?'/《》;·!〈〉"
        reduction_rule = REDUCTION_RULE

    # split
    tokens, pmarks = separate_punc(s=s, punc=punc)
    if 0:
        list(zip(tokens, pmarks, strict=True))

    # trim reduction rule
    pruned_reduction = {c: reduction_rule[c] for c in punc}
    pruned_reduction = {k: {"O": ""}.get(v, v) for k, v in pruned_reduction.items()}
    pruned_reduction = {k: v for k, v in pruned_reduction.items() if k != v}
    pruned_trans_table = str.maketrans(pruned_reduction)

    # apply reduction
    pmarks2 = [s.translate(pruned_trans_table) for s in pmarks]
    if 0:
        list(zip(pmarks, pmarks2, strict=True))

    # apply precedence
    pmarks3 = [__apply_precedence_rule(segment=segment) for segment in pmarks2]
    if 0:
        list(zip(pmarks2, pmarks3, strict=True))

    # remove starting puncts since they cannot be predicted)
    pairs = list(zip(tokens, pmarks3, strict=True))
    for i in range(len(pairs)):
        if pairs[i][0] != "":
            break
    pairs = pairs[i:]

    # merge
    s2 = "".join(["".join(t) for t in pairs])
    if 0:
        utils.temp_diff(s1=s, s2=s2)

    return s2


def __apply_precedence_rule(segment: str) -> str:
    if 0:
        segment = ",,?"
        segment = ",?"

    # If no base label was found, OTHER class is assigned.
    if len(set(segment).intersection(PRECEDENCE_RULE)) == 0:
        return ""

    # inclusion criteria. more frequently represented one.
    counts = {c: segment.count(c) for c in PRECEDENCE_RULE}
    max_count = max(counts.values())
    max_keys = {k for k, v in counts.items() if v == max_count}
    if len(max_keys) == 1:
        return max_keys.pop()

    # precedence order (strong to weak): question, period, comma
    for c in PRECEDENCE_RULE:
        if c in max_keys:
            return c

    # should not reach here
    raise ValueError(f"invalid segment: {segment}")


def encode_ner_direct(
    s: str, label: str, label2id: dict[str, int]
) -> tuple[list[str], list[str]]:
    if 0:
        s = "二月甲寅诏吏犯赃至流按察官失举者并劾之庚午置西界和市场"
        label = "OOO,OOOOO,OOOOO,OO。O,OOOOO。"

    punc = "".join(sorted(set("".join(list(label2id.keys())))))
    assert set(label).issubset("O" + punc), "invalid label"

    tokens = list(s)
    pmarks = list(label)
    assert len(tokens) == len(pmarks), "len not same!0"

    ids = [label2id.get(c, 0) for c in pmarks]
    ner_tags = ["O" if i == 0 else f"B-L{i}" for i in ids]
    assert len(tokens) == len(ner_tags), "len not same!1"
    return tokens, ner_tags


def encode_ner(s: str, label2id: dict[str, int]) -> tuple[list[str], list[str]]:
    if 0:
        s = "안녕, 하세요! 세계?"

    punc = "".join(sorted(set("".join(list(label2id.keys())))))
    tokens, pmarks = separate_punc(s=s, punc=punc, trim_left=True)
    ids = [label2id.get(c, 0) for c in pmarks]
    ner_tags = ["O" if i == 0 else f"B-L{i}" for i in ids]
    assert len(tokens) == len(ner_tags), "len not same!1"
    return tokens, ner_tags


def decode_ner(tokens: list[str], ner_tags: list[str], label2id: dict[str, int]) -> str:
    assert len(tokens) == len(ner_tags), "len not same!2"
    assert all([c == "O" or c.startswith("B-L") for c in ner_tags]), "invalid tag"
    id2label = {v: k for k, v in label2id.items()}
    ids = [0 if c == "O" else int(c.split("B-L")[1]) for c in ner_tags]
    pmarks = [id2label.get(i, "") for i in ids]
    s = "".join([x for lx in zip(tokens, pmarks, strict=True) for x in lx])
    return s
