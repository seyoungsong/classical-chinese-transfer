from pathlib import Path

from bs4 import BeautifulSoup

from src import utils


def find_special_tags(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    found_selectors = [
        t for t in SPECIAL_SELECTOR_LIST if soup.select_one(t) is not None
    ]
    s = " ; ".join(sorted(found_selectors))
    return s


def find_xml_tags(html: str) -> str:
    # DRRI
    soup = BeautifulSoup(html, "xml")
    tags = list(soup.find_all())
    found_selectors = {t.name for t in tags}
    s = " ; ".join(sorted(found_selectors))
    return s


def report_sample(d: Path, f: Path) -> Path:
    # read
    df = utils.read_df(f)
    if 0:
        df.sample(1).iloc[0].to_dict()

    # sample
    n_safe = min(30, len(df))
    df_sample = df.sample(n=n_safe, random_state=42).sort_index(ignore_index=True)

    # save
    f2 = d / f.name / "sample.json"
    f2.parent.mkdir(parents=True, exist_ok=True)
    utils.write_json(f2, df_sample.to_dict(orient="records"))

    return f2


SPECIAL_SELECTOR_LIST = [
    "a.footnote_super",
    "a.idx_annotation00",
    "a.idx_annotation01",
    "a.idx_annotation02",
    "a.idx_annotation03",
    "a.idx_annotation04",
    "a.idx_annotation05",
    "a.idx_annotation06",
    "a.sup",
    "br",
    "em.book",
    "em.era",
    "em.etc",
    "em.name",
    "em.person",
    "em.place",
    "img.newchar",
    "img.xsl_page_icon",
    "img",
    "p.paragraph",
    "span.idx_annotation00",
    "span.idx_annotation01",
    "span.idx_annotation02",
    "span.idx_annotation03",
    "span.idx_annotation04",
    "span.idx_annotation05",
    "span.idx_annotation06",  # https://sillok.history.go.kr/mc/id/qsilok_012_2680_0010_0010_0010_0050
    "span.idx_book",
    "span.idx_era",
    "span.idx_etc",
    "span.idx_name",
    "span.idx_person",
    "span.idx_place",
    "span.idx_proofreading00",
    "span.idx_proofreading01",
    "span.idx_proofreading02",
    "span.idx_proofreading03",
    "span.idx_proofreading04",
    "span.idx_proofreading05",
    "span.idx_proofreading06",
    "span.jusok",  # https://db.itkc.or.kr/dir/item?itemId=BT#/dir/node?dataId=ITKC_BT_1456A_0050_000_0290&viewSync=OT&viewSync2=KP
    "span.xsl_img_open",
    "span.xsl_tbl_open",  # https://db.itkc.or.kr/dir/item?itemId=MP#/dir/node?dataId=ITKC_MP_0597A_1190_010_0020
    "span.xsl_wonju",
    "sup",
    "table",  # https://db.itkc.or.kr/dir/item?itemId=MP#/dir/node?dataId=ITKC_MP_0597A_1190_010_0020
    "ul.ins_footnote",
    "ul.ins_source",
]
