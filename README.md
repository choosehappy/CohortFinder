# CohortFinder
---
Intelligent data partitioning using quality control metrics



# Purpose

---
Batch effects (BE) (e.g., scanner, stain variances), are systematic technical differences in data creation unrelated to biological variation. BEs have been shown to negatively impact machine learning (ML) model generalizability. Since they can result in the worst case when partioning patients into training/validation set, where patients in training set come from totally different BE groups from those in validation set. The purpose of the CohortFinder is to provide an intelligent data partition strategy trying to avoid the worst case situation without any manual effort.

CohortFinder has the following two functions:

1. Cluster patients into different BE groups using quality control metrics
2. Partition patients into training/validation set, making sure the patients in training or validation set come from all the BE groups

This tool can increase the performance and generalizability of machine learning model.



# Requirements

---
Tested with Python 3.7.10

Requires:
1. Python 
2. pip

And the following additional python package:
1. matplotlib

2. numpy

3. opencv_python_headless

4. scikit_learn

5. scipy

6. umap

7. umap_learn

   

You can install the python requirements using the following command line:

```
pip3 install -r requirements.txt
```



# Quality Control Metrics Generation

Please see  [Histoqc](https://github.com/choosehappy/HistoQC), this is an open-source quality control tool for digital pathology slides. We used the quality control metrics it generated.



# Basic Usage

---
The parameters CohortFinder used are as below:

![Screen Shot 2022-06-15 at 10.15.02 PM](/Users/fanfan/Desktop/Screen Shot 2022-06-15 at 10.15.02 PM.png)



### Run

Go to the cohortfinder repository. Download or simply git-cloned, using the following typical command line for running the tool like:

```python
 python3 cohortfinder_colormod_original.py -f "{the path of the result.tsv file of HistoQC}" -n 3
```
 For the histoqc result file, please input the complete file path, for example:
 ```python
'Histoqc-Master/histoqc_output_20220612-171953/results.tsv'
 ```
So it can generate the correct visual grouping results.



# Outputs of CohortFinder

##### 1. Result file and running log

The cohortfinder_result.tsv has two more columns than the histoqc tsc file. One is called 'groupid' represents which BE group the patient belongs to. One is called 'testind', where '1' represents the patients were partitioned into testing set, and '0' represents the patients were partitioned into training set.



##### 2. Embeded plots

Each point represents a patient and different colors represent different batch effect groups

<img src="/Users/fanfan/Desktop/cohortfinder/script_git/histoqc_output_20220615-202926/embed.png" alt="embed" style="zoom:25%;" />

##### 3. Patient partition plot

 'x' represents the patients were split into training set and '+' means the patients were partitioned into testing set. You can also find the patients information detail in the ***results_cohortfinder.tsv*** file.

<img src="/Users/fanfan/Desktop/cohortfinder/script_git/histoqc_output_20220615-202926/embed_split.png" alt="embed_split" style="zoom:25%;" />

##### 4. The visual cluster results

![groupingresults](/Users/fanfan/Desktop/groupingresults.png)

