# Replication Package for ASE'23 Submission #431: Automated Classification of Requirements-related Provisions in Food-safety Regulations


## Content description

```bash
.
├── Code
│   ├── RQ1
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

    * RQ1: contains code related to the BERT-based Language Models implemented for answering RQ1. The prerequisite packages are listed in the requirement.txt file.
    * RQ2: contains code for the BiLSTM and Keyword Search baselines. The prerequisite packages are again listed in the requirement.txt file.
    * Supplementary Code: contains auxilliary scripts that support the main analyses presented in the paper. The scripts implement 1) the calculation of summary statistics for food-safety provisions (retrieving the content of SFCR and FSRG URLs and deriving various statistics on the number of sentences found in them), 2) the claculation of Accuracy metrics for classification and significance testing (taking classification reports as input and generating a dataframe with Precision, Recall, and F-score values for each label, followed by statistical significance testing between model pairs), 3) Boxplot Visualization (based on the Accuracy calculation dataframes).

* Evaluation Results: contains two subfolder namely RQ1 and RQ2. 
    * RQ1: contains Precision, Recall and F-Score for the BERT variants classification results, along with the model hyperparameters used in the experiments, statistical significance tests of comparing BERT base approach against these BERT variants, and dfboxplots folder containing BERT variants dataframes used for visualization.
    
    * RQ2: includes the BiLSTM hyperparameters used in the experiment, statistical significance tests comparing our approach against baselines, and dfboxplots folder containing baseline dataframes used for visualization.
    
* Data: contains datasets including qualitative data derived from qualitative coding and evaluation data annotated by third-part annotators.

### Instructions to run the proposed algorithms

* Create a python environment with the packages listed in: FSR/Code/RQ1/Requirement.txt
* Open the environment and proceed to FSR main folder FSR/Code/RQ1 and execute the code BERTbase.py

## Version History

Initial Release

## Acknowledgments
