## Requirements


If user want to run the scaffold splitting code, should extra install python packages including _networkx_, _rdkit_, and _scikit-learn_.

The complete python packages are as follows: 

```shell
python == 3.7.4
torch == 1.6.0
torch-cluster == 1.5.7  
torch_scatter == 2.0.6  
torch-sparse == 0.6.9
torch-geometric == 2.0.0
pandas == 1.2.5
numpy == 1.17.2  
networkx == 2.6.3
rdkit == 2023.3.2
scikit-learn == 0.22.2
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
pip install networkx==2.6.3
pip install rdkit==2023.3.2
pip install scikit-learn==0.22.2
```

## Quick Start

A quick start example is given by:

```shell
$ python auto_test.py --data_name clintox --gpu 0
```

An example of auto search is as follows:

```shell
$ python auto_main.py --data_name clintox --gpu 0
or
$ python auto_main.py --data_name bbbp --gpu 0
```





## Arguments Description
### auto_main.py/auto_test.py
|          Name          | Default value |                                                               Description                                                                |
|:----------------------:|:-------------:|:----------------------------------------------------------------------------------------------------------------------------------------:|
|       data_name        |    clintox    |                                                   the name of molecular graph dataset                                                    |
|          gpu           |       0       |                                                              gpu device id                                                               |
|         epochs         |      100      |        the num of training epochs of each GNN architecture to obtain the reward value of each GNN architecture during auto-search        |
|      epochs_test       |      100      | the num of training epochs during testing the optimal GNN architecture, which will be trained from scratch with 20-fold cross-validation |
| controller_train_epoch |      200      |                                    the train epoch of learnable agent based on reinforcement learning                                    |
|      search_scale      |      100      |                               the number of good GNN architecture predicted by the trained learnable agent                               |


## User-defined
DAGC is very friendly for users to implement customization, users can freely define their own functional components as long as they follow the custom specification. 
Users can know the custom specification of each functional component in the following list, which is very simple. The list of definable components is as follows:


### Graph Data

#### More Benchmark Preparation

If user want to run more molecular graph benchmarks, can download the datasets as follows: 


```shell
$ cd ./scaffold_splitting_codes_version
$ wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
$ unzip chem_dataset.zip
$ rm -rf dataset/*/processed
$ mv dataset chem_data
```

#### Non-Benchmark Preparation
Then, user can define the input graph data if user use non benchmark datasets. The data format is like torch-geometric:

```shell
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch

class PyGDataset(InMemoryDataset):
    """
    data: list
    every element is a sample, in which contain feature, torch_edge_index, label
    
    dataset_save_path: str
    the saving path for dataset
    """
    def __init__(self, data, dataset_save_path):
        self.dataset = self.convert_pyg_data(data)
        self.dataset_save_path = dataset_save_path
        if not os.path.exists(self.dataset_save_path):
            os.mkdir(self.dataset_save_path)
        if os.path.exists(self.dataset_save_path + 'processed/' + self.processed_file_names[0]):
            os.remove(self.dataset_save_path + 'processed/' + self.processed_file_names[0])
        
        super(PyGDataset, self).__init__(root=self.dataset_save_path)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    def convert_pyg_data(self, data):
        converted_data = []
        for d in data:
            feature = d[0]
            torch_edge_index = d[1]
            label = d[2]
            c_d = Data(x=feature, edge_index=torch_edge_index, y=label)
            converted_data.append(c_d)
        return converted_data
  
    @property
    def processed_file_names(self):
        return ['data.pt']
        
    def process(self):
        data_list = self.dataset
        data, slices = self.collate(data_list)
        print("Saving processed files...")
        torch.save((data, slices), self.processed_paths[0])
        print('Saving complete!')
```


### Search Space
Then, user can modify the search space according to the need. The default setting of search space is as follows:
#### [Default Configuration](https://github.com/Zhen-Peng-Wu/DAGC/blob/master/scaffold_splitting_codes_version/core/search_space/search_space_config.py)
|    Search Component    |                        Candidate Operations                         |
|:----------------------:|:-------------------------------------------------------------------:|
|       GNN Depth        |                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10                    |
|  Aggregation Function  |     GCNConv, GATConv, SAGEConv, GINConv, GraphConv, GeneralConv     |
|     Local Pooling      | TopKPool, SAGPool, ASAPool, PANPool, HopPool, GCPool, GAPPool, None |
|     Global Pooling     |            GlobalMaxPool, GlobalMeanPool, GlobalSumPool             |

#### Modify GNN Depth Component
modify the _'gnn_layers'_ key in the [_search_space_config.py_](https://github.com/Zhen-Peng-Wu/DAGC/blob/master/scaffold_splitting_codes_version/core/search_space/search_space_config.py#L15)

#### Modify Aggregation Function Component
modify the _'conv'_ key in the [_search_space_config.py_](https://github.com/Zhen-Peng-Wu/DAGC/blob/master/scaffold_splitting_codes_version/core/search_space/search_space_config.py#L16)

and, define the added aggregation function like [_convolution.py_](https://github.com/Zhen-Peng-Wu/DAGC/blob/master/scaffold_splitting_codes_version/core/search_space/convolution.py)

last, register to the _'conv_map'_ function in [_utils.py_](https://github.com/Zhen-Peng-Wu/DAGC/blob/master/scaffold_splitting_codes_version/core/search_space/utils.py#L20)

#### Modify Local Pooling Component
modify the _'local_pooling'_ key in the [_search_space_config.py_](https://github.com/Zhen-Peng-Wu/DAGC/blob/master/scaffold_splitting_codes_version/core/search_space/search_space_config.py#L17)

and, define the added local pooling function like [_local_pooling.py_](https://github.com/Zhen-Peng-Wu/DAGC/blob/master/scaffold_splitting_codes_version/core/search_space/local_pooling.py)

last, register to the _'local_pooling_map'_ function in [_utils.py_](https://github.com/Zhen-Peng-Wu/DAGC/blob/master/scaffold_splitting_codes_version/core/search_space/utils.py#L39)

#### Modify Global Pooling Component
modify the _'global_pooling'_ key in the [_search_space_config.py_](https://github.com/Zhen-Peng-Wu/DAGC/blob/master/scaffold_splitting_codes_version/core/search_space/search_space_config.py#L18)

and, define the added global pooling function like [_torch_geometric.nn.pool.glob_](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/pool/glob.html)

last, register to the _'global_pooling_map'_ function in [_utils.py_](https://github.com/Zhen-Peng-Wu/DAGC/blob/master/scaffold_splitting_codes_version/core/search_space/utils.py#L8)
