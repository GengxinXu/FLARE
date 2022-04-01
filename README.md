## This is a pytorch implementation of the model FLARE (Fair cross-domain adaptation with LAtent REpresentations) - [TMI2022](https://ieeexplore.ieee.org/abstract/document/9512047/), [arXiV](https://arxiv.org/abs/2109.03478)

## Environment
- pytorch 1.0.1
- numpy 1.16.3
- pandas 0.24.2
- imbalanced-learn 0.5.0
- python 3.6.8

## Dataset

```
Data
|- feature_total237_4sites.xlsx
|- feature_detail.xlsx
```

## Usage

`python main.py`

## Citation

If you use this project in your research or wish to refer to the FLARE model, please use the following BibTeX entry.

```bash
@ARTICLE{Xu2022TMI,
  author={Xu, Geng-Xin and Liu, Chen and Liu, Jun and Ding, Zhongxiang and Shi, Feng and Guo, Man and Zhao, Wei and Li, Xiaoming and Wei, Ying and Gao, Yaozong and Ren, Chuan-Xian and Shen, Dinggang},
  journal={IEEE Transactions on Medical Imaging},
  title={Cross-Site Severity Assessment of COVID-19 From CT Images via Domain Adaptation},
  year={2022},
  volume={41},
  number={1},
  pages={88-102},
  doi={10.1109/TMI.2021.3104474}
}
```
