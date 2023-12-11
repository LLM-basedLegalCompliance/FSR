# Replication Package for RE'24: Classifying Requirements-related Provisions in Food-safety Regulations: An LLM-based Approach


## Content description

```bash
.
├── Code
│   ├── RQ1
│   │   ├── BERT
│   │   └── GPT
│   ├── RQ2
│   └── Supplementary Code
├── Data
└── Evaluation Results
    ├── RQ1
    │   └── dfboxplots
    └── RQ2
        └── dfboxplots
```
        
* Code: implementations of all the elements discussed in the paper. The folder is divided into three subfolders: RQ1, RQ2, and Supplementary Code.

    * RQ1: is divided into two subfolders: BERT and GPT.
        * BERT: contains code related to the BERT-based Language Models implemented for answering RQ1. The prerequisite packages are listed in the requirement.txt file.
        *  GPT: contains code related to GPT-3.5 Language Model implemented for answering RQ1. The prerequisite packages are listed in the requirement.txt file.
    * RQ2: contains code for the BiLSTM and Keyword Search baselines and implementation details of these baselines. The prerequisite packages are again listed in the requirement.txt file. 
    * Supplementary Code: contains auxilliary scripts that support the main analyses presented in the paper. The scripts implement 1) the calculation of summary statistics for food-safety provisions (retrieving the content of SFCR and FSRG URLs and deriving various statistics on the number of sentences found in them), 2) the claculation of Accuracy metrics for classification and significance testing (taking classification reports as input and generating a dataframe with Precision, Recall, and F-score values for each label, followed by statistical significance testing between model pairs), 3) boxplot visualization (based on the Accuracy calculation dataframes).

* Evaluation Results: contains two subfolders named RQ1 and RQ2. 
    * RQ1: contains Precision, Recall and F-Score for different BERT variants and zero-shot prompting with GPT, along with BERT variants hyperparameters used in the experiments, statistical significance tests for comparing BERT base against other variants. The dfboxplots subfolder contains the dataframes used for creating boxplots. We also provide details of results for FDA and Non-FDA part of our test data.
    
    * RQ2: contains Precision, Recall and F-Score for baselines, BiLSTM hyperparameters used in the experiment, statistical significance tests comparing BERT and GPT against baselines. Like in RQ1 the dfboxplots subfolder contains the dataframes used for creating boxplots.
    
* Data: contains our datasets including qualitative data derived from our qualitative coding as well as data curated from third-parties and used as test data in our evaluation. This folder further contains the keywordsets used for classifying scarce labels. Here we also provide a list of SFCR parts excluded form our qualitative study because these parts were not related to food products. We also provide here the prompt for our zero-shot learning experiments.

### Execution Instructions

* Create a python environment with the packages listed in: FSR/Code/RQ1/requirement.txt
* Open the environment and proceed to FSR main folder FSR/Code/RQ1 
* Set the path to data 
* Execute the code BERTbase.py

## Version History

Initial Release

## Acknowledgments
Redacted for double-anonymous review
