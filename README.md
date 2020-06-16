# IRNQA
 
This is a repository for the implementation of the paper-
[Interpretable Reasoning Network for Multi-Relation Question Answering](https://www.aclweb.org/anthology/C18-1171/)
by Mantong Zhou, Minlie Huang, Xiaoyan Zhu


## Citation

If you find the IRNQA useful, please cite the paper:
```
@inproceedings{zhou-etal-2018-interpretable,
    title = "An Interpretable Reasoning Network for Multi-Relation Question Answering",
    author = "Zhou, Mantong  and
      Huang, Minlie  and
      Zhu, Xiaoyan",
    booktitle = "Proceedings of the 27th International Conference on Computational Linguistics",
    month = aug,
    year = "2018",
    address = "Santa Fe, New Mexico, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/C18-1171",
    pages = "2010--2022"
}
```
* Open-domain Question Answering (QA) has always been a hot topic in AI and this task has recently been facilitated by large-scale Knowledge Bases (KBs)
* Question answering over knowledge bases falls into two types, namely single-relation QA and multi- relation QA
* In comparison, reasoning over multiple fact triples is required to answer multi-relation questions such as “Name a soccer player who plays at forward position at the club Borussia Dortmund.” where more than one entity and relation are mentioned. Compared to single-relation QA, multi-relation QA is yet to be addressed.
* The paper has two major contributions:
    * An Interpretable Reasoning Network which can make reasoning on multi-relation ques- tions with multiple triples in KB. Results show that our model obtains state-of-the-art performance.
    * The model is more interpretable than existing reasoning networks in that the intermediate entities and relations predicted by the hop-by-hop reasoning process construct traceable reasoning paths to clearly reveal how the answer is derived.
    
## Implementation
For implementation I have used following major libraries:
* Pytorch
* Numpy
* Scikit-Learn

The code structure is as follows:
* model.py : This file contains the model and architecture for implementation
* preprocess.py : Loads and reads the data and formats it in for input to model.
* train.py : Used to train the model and predict the various summary parameters as described below:
```
-----------------------
Epoch 10000
Train Accuracy: 80.31
Validation Accuracy: 72.6721
-----------------------
```





