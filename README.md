# GCN_for_Node_Classification_Task

## Goal

Leverage GCN for node classification tasks on datasets Cora, Citeseer and Pubmed.

## Datasets

|Dataset|#Nodes|#Edges|#Features|#Classes|
|:----:|:----:|:----:|:----:|:-----:|
|Cora|2,708|10,556|1,433|7|
|Citeseer|3,327|9,104|3,703|6|
|PubMed|19,717|88,648|500|3|

### Data Source

https://github.com/kimiyoung/planetoid/tree/master/data

Datasets for Cora, Citeseet, and PubMed are available in the directory `Cora`, `Citeseet`,  and PubMed, in a preprocessed format stored as numpy/scipy files.

> - `allx`, the feature vectors of both labeled and unlabeled training instances (a superset of  `x`),
> 
> - `ally`, the labels for instances in  `allx`,
> 
> - `x`, the feature vectors of the labeled training instances,
> 
> - `y`, the one-hot labels of the labeled training instances,
> 
> - `tx`, the feature vectors of the test instances,
> 
> - `ty`, the one-hot labels of the test instances,
> 
> - `test.index`, the indices of test instances in  `graph`, for the inductive setting,
> 
> - `graph`, a `dict` in the format `{index: [index_of_neighbor_nodes]}`, where the neighbor nodes are organized as a list. The current version only supports binary graphs.

### Datasets Introduction
- **Cora**  
The Cora dataset consists of machine learning papers, which are classified into the following seven categories:
	- Case Based
	- Genetic Algorithms
	- Neural Networks
	- Probabilistic Methods
	- Reinforcement Learning
	- Rule Learning
	- Theory 

The papers were selected in such a way that each paper in the final corpus cited or was cited by at least one other paper. There are 2708 papers in the entire corpus. After stemming and removing stop words, we were left with only 1433 unique words in size. All words with document frequency less than 10 were removed.

- **Citeseer**  
The Citeseer dataset is a selection of papers from the CiteSeer digital papers library, classified into the following six categories:
	- Agents
	- AI
	- DB
	- IR
	- ML
	- HCI

The papers were selected in such a way that each paper in the final corpus cited or was cited by at least one other paper. There are 3327 papers in the entire corpus.

- **PubMed**  
The PubMed dataset includes 19,717 scientific publications on diabetes from the Pubmed database, divided into three categories:
	- Diabetes Mellitus, Experimental
	- Diabetes Mellitus Type 1
	- Diabetes Mellitus Type 2 

The citation network consists of 44338 links. Each publication in the dataset is described by a TF/IDF-weighted word vector from a dictionary of 500 unique words. TF-IDF (term frequency–inverse document frequency) is a commonly used weighting technique for information retrieval and data mining. TF is Term Frequency, and IDF is Inverse Document Frequency. TF-IDF is a statistical method for evaluating the importance of a word to a document set or a document in a corpus. The importance of a word increases proportionally to the number of times it appears in the document, but decreases inversely proportional to the frequency it appears in the corpus.

## Environment

python == 3.8.16
 
pytorch == 1.12.1

scikit-learn == 1.2.2

matplotlib == 3.7.1


## Usage


## Result

|Dataset|Accuracy|F1_score|
|:----:|:----:|:----:|
|Cora|0.8120|0.8051|
|Citeseer|0.6930|0.6670|
|PubMed|0.7940|0.7893|

## Reference
- Zhilin Yang, William Cohen, Ruslan Salakhudinov. [Proceedings of Machine Learning Research.](http://proceedings.mlr.press/v48/yanga16) *Proceedings of The 33rd International Conference on Machine Learning, PMLR 48*:40-48, 2016.

- [GRAPH CONVOLUTIONAL NETWORKS.](https://tkipf.github.io/graph-convolutional-networks/)
