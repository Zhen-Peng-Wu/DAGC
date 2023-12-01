## Requirements

```shell
python == 3.7.4
torch == 1.6.0
torch-cluster == 1.5.7  
torch_scatter == 2.0.6  
torch-sparse == 0.6.9
torch-geometric == 2.0.0
pandas == 1.2.5
numpy == 1.17.2  
```

## Installation

```shell
conda create -n DAGC python==3.7.4
conda activate DAGC
pip install torch==1.6.0 -f https://download.pytorch.org/whl/cu101/torch-1.6.0%2Bcu101-cp37-cp37m-linux_x86_64.whl
pip install torch-cluster==1.5.7 -f https://data.pyg.org/whl/torch-1.6.0%2Bcu101.html
pip install torch_scatter==2.0.6 -f https://data.pyg.org/whl/torch-1.6.0%2Bcu101.html
pip install torch-sparse==0.6.9 -f https://data.pyg.org/whl/torch-1.6.0%2Bcu101.html
pip install torch-geometric==2.0.0
pip install pandas==1.2.5
pip install numpy==1.17.2
```

## Quick Start

A quick start example is given by:

```shell
$ python auto_test.py --data_name MUTAG --gpu 0
```

An example of auto search is as follows:

```shell
$ python auto_main.py --data_name MUTAG --gpu 0
or
$ python auto_main.py --data_name COX2 --gpu 0
```
