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

    * RQ1: contains code related to the BERT-based Language Models implemented for answering RQ1 and the prerequisite packages in the requirement.txt file.
    * RQ2: contains code for BiLSTM and Keyword Search baseline models, as well as the prerequisite packages specified in the requirement.txt file.
    * Supplementary: contains subsidiary codes that support the main analyses presented in this study. This includes scripts for Food Safety Sentence Statistics (retrieves the content of food safety-related URLs and calculates various statistics on the number of sentences found in them, such as the mean, standard deviation, and other relevant metrics), scripts for Classification Evaluation and Significance Testing (takes classification reports as input and generates a dataframe with precision, recall, and f-score values for each label. It then performs statistical significance testing between two models using the generated dataframe) and Boxplot Visualization (uses the second dataframe generated in the previous notebook to create boxplots showcasing the distribution of results).

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
