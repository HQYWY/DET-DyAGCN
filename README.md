# Readme

This is the repository for the paper "[A Novel Dynamic Graph Attention Aggregation Network for Multivariate Time Series Classification.](https://github.com/ywyhq/DET-DyAGCN)"

This repository contains:

1. data: Available sample datasets are provided here. For the complete datasets, please visit the [Time Series Machine Learning Website](https://www.timeseriesclassification.com/index.php). 
Due to size limitations, this project also offers partial datasets, accessible via [UCR_dataset](https://www.kaggle.com/datasets/hanqiuyeweiyang/ucr-dataset).
2. log: The log folder for recording each model training session.
3. src: The complete code, including model training and evaluation, etc.
4. environment.yml: A usable example environment configuration. In principle, minimal dependencies need to be installed based on the code's import statements, but this example can be used directly to run the code.


## Install
You can directly use a conda virtual environment to load the example environment. \
Please note that this project defaults to relying on a GPU environment. You can refer to **[pytorch](https://pytorch.org/)** for installing the basic PyTorch runtime environment.

```sh
$ conda env create -f environment.yml
```

## Usage
You can directly use the sample data in the `data` folder to complete the model training and evaluation process. \
Accessing the complete dataset requires further visiting [Time Series Machine Learning Website](https://www.timeseriesclassification.com/index.php) or [UCR_dataset](https://www.kaggle.com/datasets/hanqiuyeweiyang/ucr-dataset).

```sh
$ cd src
$ python train.py
```

## Maintainers
[@HQYWY](https://github.com/HQYWY).

## Citation

If you use this code or find our work helpful, please cite our paper:

```bibtex
@article{gui2025novel,
  title={A Novel Dynamic Graph Attention Aggregation Network for Multivariate Time Series Classification},
  author={Gui, Haoyu and Tang, Xianghong and Li, Guanjun and Wang, Chaobin and Lu, Jianguang},
  journal={Pattern Recognition},
  pages={112732},
  year={2025},
  publisher={Elsevier}
}
```

## License
Apache-2.0 license