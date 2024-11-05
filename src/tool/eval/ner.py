import re

from src import utils
from src.tool.eval.punc import name_punc_label, old_remove_punc

_PREF = utils.NER_PREF  # prefix for inline XML tags


def is_valid_xml(s_xml: str) -> bool:
    count1 = s_xml.count(f"{_PREF}")
    items = xml2items(s_xml=s_xml)
    count2 = len([d["name"] for d in items if d["name"] is not None])
    is_valid = count1 == count2 * 2
    return is_valid


def iob2xml(tokens: list[str], ner_tags: list[str]) -> str:
    assert len(tokens) == len(ner_tags), "not equal in length"

    result: list[str] = []
    open_tag: str | None = None

    for token, tag in zip(tokens, ner_tags, strict=True):
        if tag.startswith("B-"):
            if open_tag:
                result.append(f"</{open_tag}>")
            # Remove the "B-" part and add unique emoji
            open_tag = _PREF + tag[2:]
            result.append(f"<{open_tag}>{token}")
        elif tag.startswith("I-") and open_tag:
            result.append(f"{token}")
        else:
            if open_tag:
                result.append(f"</{open_tag}>{token}")
                open_tag = None
            else:
                result.append(token)

    # Close any remaining open tag
    if open_tag:
        result.append(f"</{open_tag}>")

    s_xml = "".join(result)
    return s_xml


def xml2items(s_xml: str) -> list[dict[str, str | None]]:
    # Define patterns
    split_pattern = re.compile(rf"(<{_PREF}[^>]+>.+?</{_PREF}[^>]+>)")
    item_pattern = re.compile(rf"^<{_PREF}([^>]+)>(.+?)</{_PREF}([^>]+)>$")
    delete_pattern = re.compile(rf"<{_PREF}([^>]+)></{_PREF}([^>]+)>")

    # Delete empty tags
    s_xml = delete_pattern.sub("", s_xml)

    # Split the XML into parts
    substrs: list[str] = split_pattern.split(s_xml)

    # Trim empty strings
    substrs = [s for s in substrs if len(s) > 0]

    # Process each part
    items: list[dict[str, str | None]] = []
    for substr1 in substrs:
        assert isinstance(substr1, str), f"not a string1: {substr1}"
        match1 = item_pattern.match(substr1)
        if match1:
            for s1 in match1.groups():
                assert isinstance(s1, str), f"not a string2: {s1}"
                assert len(s1) > 0, f"empty string3: {s1}"
            name1, text1, name2 = match1.groups()
            assert name1 == name2, f"tags do not match4: {name1} != {name2}"
            elem1 = {"text": text1, "name": name1}
        else:
            elem1 = {"text": substr1, "name": None}
        items.append(elem1)

    return items


def names_in_xml(s_xml: str) -> list[str]:
    items = xml2items(s_xml=s_xml)
    names = {item["name"] for item in items if item["name"] is not None}
    names2 = sorted(names)
    return names2


def items2xml(items: list[dict[str, str | None]]) -> str:
    s = ""
    for item in items:
        text = item["text"]
        assert isinstance(text, str), f"not a string4: {text}"
        name = item["name"]
        if name is None:
            s += text
        else:
            assert len(text) > 0, f"empty string5: {text}"
            assert len(name) > 0, f"empty string6: {name}"
            s += f"<{_PREF}{name}>{text}</{_PREF}{name}>"
    return s


def xml2iob(s_xml: str) -> tuple[list[str], list[str]]:
    items = xml2items(s_xml=s_xml)
    tokens: list[str] = []
    ner_tags: list[str] = []
    for item1 in items:
        text1 = item1["text"]
        name1 = item1["name"]

        # Split text into individual characters
        assert isinstance(text1, str), f"not a string5: {text1}"
        tokens.extend(list(text1))

        if name1 is None:
            # If name is None, tag all characters as 'O'
            ner_tags.extend(["O"] * len(text1))
        else:
            # If name is not None, tag the first character as 'B-{name}' and the rest as 'I-{name}'
            ner_tags.append(f"B-{name1}")
            ner_tags.extend([f"I-{name1}"] * (len(text1) - 1))

    # sanity check
    assert len(tokens) == len(ner_tags), "not equal in length6"

    # final check
    s2 = iob2xml(tokens=tokens, ner_tags=ner_tags)
    assert s_xml == s2, "not equal in xml7"

    return tokens, ner_tags


def xml2plaintext(s_xml: str) -> str:
    items = xml2items(s_xml=s_xml)
    s = "".join([item["text"] for item in items])  # type: ignore
    return s


def align_char_and_puncs(s_punc: str, s_orig: str) -> list[tuple[str, str]]:
    if 0:
        from src import utils

        utils.temp_diff(s_orig, s_punc)
        s_orig = "注博广至似此 疏注云孑句孑凡戟而无刃秦晋之间谓之孑或谓之鏕吴扬之间谓之伐东齐秦晋之间其大者谓之曼胡其曲者谓之句孑曼胡 ○凡戟以下至末并方言文孑今本作𫓦鏕作鏔曼并作镘"
        s_punc = "注博广至似此 疏注云: '孑、句孑, 凡戟而无刃, 秦、晋之间谓之孑, 或谓之鏕, 吴、扬之间谓之伐。 东齐、秦、晋之间, 其大者谓之曼胡, 其曲者谓之句孑、曼胡 ○凡戟以下至末, 并方言文, 孑, 今本作𫓦, 鏕作鏔, 曼并作镘。"
    pairs: list[list[str]] = [["", ""]]

    # Split
    s_temp = s_orig
    for c in s_punc:
        if s_temp.startswith(c):
            pairs.append([c, ""])
            s_temp = s_temp[1:]
        else:
            pairs[-1][1] += c
    assert len(s_temp) == 0, f"Remaining: {s_temp}"

    # Trim
    pairs = [t for t in pairs if len(t[0]) >= 1 or len(t[1]) >= 1]

    # check?
    assert len(pairs) == len(s_orig), f"len mismatch: {len(pairs)} != {len(s_orig)}"

    # check
    s_punc2 = "".join(["".join(t) for t in pairs])
    assert s_punc == s_punc2, f"Expected: {s_punc}, Got: {s_punc2}"

    # convert to tuple
    pairs2 = [(t[0], t[1]) for t in pairs]

    return pairs2


def xml_remove_punc(s_xml: str, punc: str) -> str:
    # convert to items
    items = xml2items(s_xml=s_xml)

    # remove punc
    texts2 = [old_remove_punc(s=d["text"], punc=punc) for d in items]  # type: ignore

    # update items
    items2 = [dict(d, text=s) for d, s in zip(items, texts2, strict=True)]

    # filter out empty text
    items3 = [d for d in items2 if len(d["text"]) >= 1]  # type: ignore

    # convert to xml
    s_xml_result = items2xml(items=items3)
    return s_xml_result


def xml_add_punc(s_xml: str, s_punc: str) -> str:
    if 0:
        s_xml = "<▪other>中和</▪other><▪other>中和</▪other>二年七月二十七日某官某乙奉太尉處分為故<▪other>昭義</▪other>僕射於<▪other>法雲寺</▪other>設三百僧齋竝寫金光明經五部法華經一部永充供養蓋聆佛修慧力普濟群迷人發信心終成善願"
        s_punc = "中和,中和二年七月二十七日, 某官某乙奉太尉處分, 為故昭義僕射於法雲寺, 設三百僧齋, 竝寫《金光明經》五部法華經》一部, 永充供養。 蓋聆佛修慧力, 普濟群迷; 人發信心, 終成善願。 "
        s_xml_punc = "<▪other>中和</▪other>,<▪other>中和</▪other>二年七月二十七日, 某官某乙奉太尉處分, 為故<▪other>昭義</▪other>僕射於<▪other>法雲寺</▪other>, 設三百僧齋, 竝寫《金光明經》五部法華經》一部, 永充供養。 蓋聆佛修慧力, 普濟群迷; 人發信心, 終成善願。 "

    # convert
    items = xml2items(s_xml=s_xml)
    s_orig: str = "".join([d["text"] for d in items])  # type: ignore

    # align
    pairs = align_char_and_puncs(s_punc=s_punc, s_orig=s_orig)

    # process
    pair_idx = 0
    item_idx = 0
    items_punc: list[dict[str, str | None]] = []
    residual_punc = ""
    while item_idx < len(items):
        # unpack
        item1 = items[item_idx]
        text1 = item1["text"]
        name1 = item1["name"]
        assert isinstance(text1, str), f"not a string8: {text1}"

        snippet = pairs[pair_idx : pair_idx + len(text1)]
        text1_punc = ""

        if name1:
            if len(residual_punc) >= 1:
                items_punc.append({"text": residual_punc, "name": None})
                residual_punc = ""
            last_char, last_punc = snippet[-1]
            snippet[-1] = (last_char, "")
            residual_punc = last_punc
            text1_punc += "".join(["".join(t) for t in snippet])
        else:
            if len(residual_punc) >= 1:
                text1_punc = residual_punc + text1_punc
                residual_punc = ""
            text1_punc += "".join(["".join(t) for t in snippet])

        items_punc.append({"text": text1_punc, "name": name1})
        pair_idx += len(text1)
        item_idx += 1

    # residual
    if len(residual_punc) >= 1:
        items_punc.append({"text": residual_punc, "name": None})
        residual_punc = ""

    s_xml_punc = items2xml(items=items_punc)

    # should check more?

    return s_xml_punc


def xml_change_label(s_xml: str, label_map: dict[str, str]) -> str:
    items = xml2items(s_xml=s_xml)
    items2 = [dict(d, name=label_map.get(d["name"], d["name"])) for d in items]  # type: ignore
    s_xml2 = items2xml(items2)
    return s_xml2


def text2punc_xml(s: str, not_punc: str, remove_whites: bool) -> str:
    # count
    if remove_whites:
        s2 = utils.remove_whites(s)
    else:
        s2 = utils.squeeze_whites(s)
    items = utils.chunk_by_classifier(
        s=s2, f=lambda x: utils.is_punc_unicode(c=x, not_punc=not_punc)
    )
    for d in items:
        d["name"] = "punc" if d["label"] else None
    punc_xml = items2xml(items)
    return punc_xml


def punc_iob2punc_xml(
    tokens: list[str], ner_tags: list[str], label2id: dict[str, str]
) -> str:
    assert len(tokens) == len(ner_tags), "not equal in length"
    id2label = {v: k for k, v in label2id.items()}
    s_xml = iob2xml(tokens=tokens, ner_tags=ner_tags)
    items = xml2items(s_xml=s_xml)
    items2 = []
    for item1 in items:
        text1 = item1["text"]
        name1 = item1["name"]
        if name1 is None:
            items2.append(item1)
        else:
            items2.append({"text": text1, "name": None})
            label1 = id2label[name1]
            items2.append({"text": label1, "name": "punc"})
    punc_xml = items2xml(items2)
    return punc_xml


def _reduce_punc(text: str) -> str:  # noqa: C901
    if 0:
        text = "'、。"
    reduce_map = {
        ",": ",",
        "-": ",",
        "/": ",",
        ":": ",",
        "|": ",",
        "·": ",",
        "、": ",",
        "?": "?",
        "!": "。",
        ".": "。",
        ";": "。",
        "。": "。",
    }
    if 0:
        _ = ",。、\":?》『』《/;'!「」·)(|-][*‖〈〉.{}"
        reduce_rule = {
            ",": ",-/:|·、",
            "?": "?",
            "。": "!.;。",
            "O": " \"'()*[]{}‖〈〉《》「」『』",
        }
        {k: "".join(sorted(set(v))) for k, v in reduce_rule.items()}
        reduce_map = {}
        for k, v in reduce_rule.items():
            if k == "O":
                continue
            for c in v:
                reduce_map[c] = k

    # reduce to basic punctuation marks
    text = "".join([reduce_map.get(c, "") for c in text])

    # If no base punc was found, OTHER class is assigned.
    punc_order = "?。,"
    if len(set(text).intersection(punc_order)) == 0:
        return ""

    # inclusion criteria. more frequently represented one.
    counts = {c: text.count(c) for c in punc_order}
    max_count = max(counts.values())
    max_keys = {k for k, v in counts.items() if v == max_count}
    if len(max_keys) == 1:
        return max_keys.pop()

    # precedence order (strong to weak): question, period, comma
    for c in punc_order:
        if c in max_keys:
            return c

    # should not reach here
    raise ValueError(f"invalid segment: {text}")


def punc_xml_reduce(punc_xml: str) -> str:
    if 0:
        punc_xml = "太宗<▪punc>、</▪punc>高宗之世<▪punc>,</▪punc>屡欲立明堂<▪punc>,</▪punc>诸儒议其制度<▪punc>,</▪punc>不决而止<▪punc>。</▪punc>及太后称制<▪punc>,</▪punc>独与北门学士议其制<▪punc>,</▪punc>不问诸儒<▪punc>。</▪punc>诸儒以为明堂当在国阳丙己之地<▪punc>,</▪punc>三里之外<▪punc>,</▪punc>七里之內<▪punc>。</▪punc>太后以为去宫太远<▪punc>。</▪punc>二月<▪punc>,</▪punc>庚午<▪punc>,</▪punc>毁乾元殿<▪punc>,</▪punc>于其地作明堂<▪punc>,</▪punc>以僧怀义为之使<▪punc>,</▪punc>凡役数万人<▪punc>。</▪punc>"
    items = xml2items(s_xml=punc_xml)
    items2 = []
    for item1 in items:
        text1 = item1["text"]
        name1 = item1["name"]
        if name1 is None:
            items2.append(item1)
        else:
            assert text1 is not None, "text is None"
            text2 = _reduce_punc(text1)
            if len(text2) >= 1:
                items2.append({"text": text2, "name": "punc"})
    punc_xml_reduced = items2xml(items2)
    return punc_xml_reduced


def punc_xml2text_nopunc(punc_xml: str) -> str:
    if 0:
        punc_xml = "太宗<▪punc>、</▪punc>高宗之世<▪punc>,</▪punc>屡欲立明堂<▪punc>,</▪punc>诸儒议其制度<▪punc>,</▪punc>不决而止<▪punc>。</▪punc>及太后称制<▪punc>,</▪punc>独与北门学士议其制<▪punc>,</▪punc>不问诸儒<▪punc>。</▪punc>诸儒以为明堂当在国阳丙己之地<▪punc>,</▪punc>三里之外<▪punc>,</▪punc>七里之內<▪punc>。</▪punc>太后以为去宫太远<▪punc>。</▪punc>二月<▪punc>,</▪punc>庚午<▪punc>,</▪punc>毁乾元殿<▪punc>,</▪punc>于其地作明堂<▪punc>,</▪punc>以僧怀义为之使<▪punc>,</▪punc>凡役数万人<▪punc>。</▪punc>"
    items = xml2items(s_xml=punc_xml)
    s_nopunc = "".join([d["text"] for d in items if d["name"] is None])  # type: ignore
    return s_nopunc


def punc_xml2punc_iob(
    punc_xml: str, label2id: dict[str, str]
) -> tuple[list[str], list[str]]:
    items = xml2items(s_xml=punc_xml)
    tokens: list[str] = []
    ner_tags: list[str] = []
    if 0:
        item1 = items[0]
        item1 = items[1]
    for item1 in items:
        text1 = item1["text"]
        name1 = item1["name"]

        # check
        assert isinstance(text1, str), f"not a string5: {text1}"

        # skip punc before first text
        if len(tokens) == 0 and name1 == "punc":
            continue

        if name1 is None:
            # Split text into individual characters
            tokens.extend(list(text1))
            # If name is None, tag all characters as 'O' for now
            ner_tags.extend(["O"] * len(text1))
        elif name1 == "punc":
            # If name is punc, tag the last character as 'B-{label}' (or 'O' if it's not a target punc label)
            id1 = label2id.get(text1, "")
            ner_tags[-1] = f"B-{id1}" if id1 else "O"
        else:
            raise ValueError(f"unexpected name: {name1}")

    # sanity check
    assert len(tokens) == len(ner_tags), "not equal in length6"

    return tokens, ner_tags


def punc_xml2punc_iob2(punc_xml: str) -> tuple[list[str], list[str]]:
    # We don't need label2id here
    items = xml2items(s_xml=punc_xml)
    tokens: list[str] = []
    ner_tags: list[str] = []
    if 0:
        item1 = items[0]
        item1 = items[1]
    for item1 in items:
        text1 = item1["text"]
        name1 = item1["name"]

        # check
        assert isinstance(text1, str), f"not a string5: {text1}"

        # skip punc before first text
        if len(tokens) == 0 and name1 == "punc":
            continue

        if name1 is None:
            # Split text into individual characters
            tokens.extend(list(text1))
            # If name is None, tag all characters as 'O' for now
            ner_tags.extend(["O"] * len(text1))
        elif name1 == "punc":
            # If name is punc, tag the last character as 'B-{label}' (or 'O' if it's not a target punc label)
            id1 = name_punc_label(text1)
            ner_tags[-1] = f"B-{id1}" if id1 else "O"
        else:
            raise ValueError(f"unexpected name: {name1}")

    # sanity check
    assert len(tokens) == len(ner_tags), "not equal in length6"

    return tokens, ner_tags


def text2punc_iob(
    s: str, not_punc: str, label2id: dict[str, str]
) -> tuple[list[str], list[str]]:
    punc_xml = text2punc_xml(s=s, not_punc=not_punc, remove_whites=True)
    tokens, ner_tags = punc_xml2punc_iob(punc_xml=punc_xml, label2id=label2id)
    return tokens, ner_tags
