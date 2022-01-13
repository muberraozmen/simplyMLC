# Multi Label Classification using Transformer

### Description
This repository contains a simple adaptation of [[Transformer]](https://arxiv.org/abs/1706.03762) layers to multi-label classification problem.
Basically, the encoder stays same, and decoder attention layers are modified to obtain multi-label likelihoods output.

### Requirements
- Python 3.6
- PyTorch 1.10
- Numpy 1.19.1
- tqdm 4.62.3

### Usage
0. install the required packages and their dependencies, if your environment does not have them already
   ```
   pip install -r requirements.txt
   ```
1. bring your dataset into following format: 
   ```
   data = {'train': {'features': List[List[int]], 'labels': List[List[int]]},
           'test' : {'features': List[List[int]], 'labels': List[List[int]]}, 
           'vocab': {'id2feature': Dict[int, str], 'id2label': Dict[int, str]}
          }
   ```
   see `readers.py` for an example of converting from [[mulan]](http://mulan.sourceforge.net/datasets.html) repository dataset format.
2. modify `config.json` according to your needs
3. run `python main.py -configuration config.json`

### Results on benchmark datasets from [[scikit-multilearn]](http://scikit.ml/index.html)
If you would like reproduce these results include `scikit-multilearn~=0.2.0` in your virtual environment for downloading the datasets. 


### Notes
`transformer.py` is licensed under the [Apache License](https://github.com/huggingface/transformers/blob/9a94bb8e218033cffa1ef380010b528410ba3ca7/LICENSE).