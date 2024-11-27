<h1 align="center">LLMs to Replace Crowdsourcing For Parallel Data Creation? The Case of Text Detoxification</h1>


<div align="center">
<img src="static/images/illustration.drawio_page.jpg" alt="ImageTranscreation">

[![EMNLP](https://img.shields.io/badge/EMNLP-Findings%202024-b31b1b)](https://2024.emnlp.org)
[![Web Page](https://img.shields.io/badge/🌎-Website-blue.svg)](https://s-nlp.github.io/pseudoparadetox/)
</div>


This is the official implementation of the paper [LLMs to Replace Crowdsourcing For Parallel Data Creation? The Case of Text Detoxification](https://aclanthology.org/2024.findings-emnlp.839.pdf) by Daniil Moskovskiy, Sergey Pletenev and Alexander Panchenko.

## Abstract

The lack of high-quality training data remains a significant challenge in NLP. Manual annotation methods, such as crowdsourcing, are costly, require intricate task design skills, and, if used incorrectly, may result in poor data quality. From the other hand, LLMs have demonstrated proficiency in many NLP tasks, including zero-shot and few-shot data annotation. However, they often struggle with text detoxification due to alignment constraints and fail to generate the required detoxified text. This work explores the potential of modern open source LLMs to annotate parallel data for text detoxification. Using the recent technique of activation patching, we generate a pseudo-parallel detoxification dataset based on ParaDetox. The detoxification model trained on our generated data shows comparable performance to the original dataset in automatic detoxification evaluation metrics and superior quality in manual evaluation and side-by-side comparisons.

## Results



## Test data

Test part of ParaDetox dataset is available upon [request](mailto:daniil.moskovskiy@skoltech.ru).

## Citation

If you find this work useful, please cite this paper:

```bibtex
@inproceedings{moskovskiy-etal-2024-llms,
    title = "{LLM}s to Replace Crowdsourcing For Parallel Data Creation? The Case of Text Detoxification",
    author = "Moskovskiy, Daniil  and
      Pletenev, Sergey  and
      Panchenko, Alexander",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.839",
    pages = "14361--14373",
    abstract = "The lack of high-quality training data remains a significant challenge in NLP. Manual annotation methods, such as crowdsourcing, are costly, require intricate task design skills, and, if used incorrectly, may result in poor data quality. From the other hand, LLMs have demonstrated proficiency in many NLP tasks, including zero-shot and few-shot data annotation. However, they often struggle with text detoxification due to alignment constraints and fail to generate the required detoxified text. This work explores the potential of modern open source LLMs to annotate parallel data for text detoxification. Using the recent technique of activation patching, we generate a pseudo-parallel detoxification dataset based on ParaDetox. The detoxification model trained on our generated data shows comparable performance to the original dataset in automatic detoxification evaluation metrics and superior quality in manual evaluation and side-by-side comparisons.",
}
```