def split_punc2(s: str, punc: str) -> list[str]:
    # separate
    tokens: list[str] = []
    for i in range(len(s)):
        c = s[i]
        if c not in punc:
            tokens.append(c)
        else:
            tokens.append(c)
            tokens.append("")

    # check
    s_reconst = "".join(tokens)
    assert s == s_reconst, "reconst not equal"

    return tokens
