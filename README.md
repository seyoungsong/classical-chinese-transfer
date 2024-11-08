# Classical Chinese Transfer Analysis

Code and data for the paper ["When Does Classical Chinese Help? Quantifying Cross-Lingual Transfer in Hanja and Kanbun"](https://arxiv.org/abs/2411.04822) (preprint)

## Abstract

Historical and linguistic connections within the Sinosphere have led researchers to use Classical Chinese resources for cross-lingual transfer when processing historical documents from Korea and Japan. In this paper, we question the assumption of cross-lingual transferability from Classical Chinese to Hanja and Kanbun, the ancient written languages of Korea and Japan, respectively. Our experiments across machine translation, named entity recognition, and punctuation restoration tasks show minimal impact of Classical Chinese datasets on language model performance for ancient Korean documents written in Hanja, with performance differences within ±0.0068 F1-score for sequence labeling tasks and up to +0.84 BLEU score for translation. These limitations persist consistently across various model sizes, architectures, and domain-specific datasets. Our analysis reveals that the benefits of Classical Chinese resources diminish rapidly as local language data increases for Hanja, while showing substantial improvements only in extremely low-resource scenarios for both Korean and Japanese historical documents. These mixed results emphasize the need for careful empirical validation rather than assuming benefits from indiscriminate cross-lingual transfer.

## Data

Our experiments use the following datasets:

- **Hanja (Korean Historical Documents)**
  - Royal Records: AJD, DRS, DRRI
  - Literary Works: KLC (Korean Literary Collections)
- **Classical Chinese**
  - NiuTrans Classical Chinese to Modern Chinese dataset
  - C2MChn dataset
  - WYWEB evaluation benchmark
  - OCDB (Oriental Classics Database)

For access to the datasets, please refer to the data sources section in our paper.

## Models

We evaluate models across three tasks:

- Machine Translation (MT)
- Named Entity Recognition (NER)
- Punctuation Restoration (PR)

Our implementations use:

- Qwen2-7B for MT
- SikuRoBERTa for NER and PR

## Citation

```bibtex
@misc{song2024doesclassicalchinesehelp,
    archiveprefix = {arXiv},
    author        = {Seyoung Song and Haneul Yoo and Jiho Jin and Kyunghyun Cho and Alice Oh},
    eprint        = {2411.04822},
    primaryclass  = {cs.CL},
    title         = {When Does Classical Chinese Help? Quantifying Cross-Lingual Transfer in Hanja and Kanbun},
    url           = {https://arxiv.org/abs/2411.04822},
    year          = {2024},
}
```
