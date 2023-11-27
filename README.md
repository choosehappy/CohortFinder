

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

Please see  [Histoqc](https://github.com/choosehappy/HistoQC), this is an open-source quality control tool for digital pathology slides. We use the quality control metrics it generates.



# Basic Usage

---
The parameters CohortFinder used are as below:
```bash
python3 cohortfinder_colormod_original.py --help
```

```
usage: cohortfinder_colormod_original.py [-h] [-c COLS] [-l LABELCOLUMN] [-s SITECOLUMN] [-p PATIENTIDCOLUMN] [-t TESTPERCENT] [-b] [-y] [-r RANDOMSEED] [-o OUTDIR] [-n NCLUSTERS] histoqcdir

Split histoqc tsv into training and testing

positional arguments:
  histoqcdir            The directory containing the output of HistoQC. This argument is required.

optional arguments:
  -h, --help            show this help message and exit
  -c COLS, --cols COLS  columnts to use for clustering, comma seperated
  -l LABELCOLUMN, --labelcolumn LABELCOLUMN
                        column name associated with a 0,1 label
  -s SITECOLUMN, --sitecolumn SITECOLUMN
                        column name associated with site variable
  -p PATIENTIDCOLUMN, --patientidcolumn PATIENTIDCOLUMN
                        column name associated with patient id, ensuring slides are grouped
  -t TESTPERCENT, --testpercent TESTPERCENT
  -b, --batcheffectsitetest
  -y, --batcheffectlabeltest
  -r RANDOMSEED, --randomseed RANDOMSEED
  -o OUTDIR, --outdir OUTDIR
  -n NCLUSTERS, --nclusters NCLUSTERS
                        Number of clusters to attempt to divide data into before splitting into cohorts, default -1 of negative 1 makes best guess
```



### Parameter meaning:



```
-c: metrics calculated by HistoQC we used for batch effect group generation, the default metrics are:
"mpp_x,mpp_y,michelson_contrast,rms_contrast,grayscale_brightness,chan1_brightness,chan2_brightness,chan3_brightness,chan1_brightness_YUV,chan2_brightness_YUV,chan3_brightness_YUV"
```

This is the description of the metrics used to identify the batch effects in our previous [work](https://pubmed.ncbi.nlm.nih.gov/33197281/), as they quantify chromatic artifacts imparted during the staining and cutting of the tissue samples steps conducted at individual laboratories before central scanning. And you can also try other metrics if they have some influence during the scanning or the staining process for your slides.

| Quality control metric | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| Mpp_x                  | Microns per pixel in the X dimension at base magnification   |
| Mpp_y                  | Microns per pixel in the Y dimension at base magnification   |
| Michelson_constrast    | Measurement of image contrast deﬁned by luminance difference over average luminance |
| Rms_contrast           | Root mean square (RMS) contrast, deﬁned as the standard deviation of the pixel intensities across the pixels of interests |
| Grayscale_brightness   | Mean pixel intensity of the image after converting the image to grayscale |
| Chan1_brightness       | Mean pixel intensity of the red color channel of the image   |
| Chan2_brightness       | Mean pixel intensity of the green color channel of the image |
| Chan3_brightness       | Mean pixel intensity of the blue color channel of the image  |
| Chan1_brightness_YUV   | Mean channel brightness of red color channel of image after converting to YUV color space |
| Chan2_brightness_YUV   | Mean channel brightness of green color channel of image after converting to YUV color space |
| Chan3_brightness_YUV   | Mean channel brightness of blue color channel of image after converting to YUV color space |

```
-l: the column contains the patient label information (0,1), this column needs to be manually added into the input data. The default value is None.
```

```
-s: the column contains the site id information, this column needs to be manually added into the input data. The default value is None.
```

```
-p: the column contains the patient id information, this column needs to be manually added into the input data. The default value is None.
```

```
-t: the percentage of data you want to set as testing set for your machine learning model, default number = 0.2
```

```
-b: statistical tests for variable confounding, at site level.
```

```
-y: statistical tests for variable confounding, at label level
```

```
-r: random seed
```

```
-o: the output directory. The default value of None causes cohortfinder to use histoqcdir as the output directory.
```

```
-n: Number of clusters to attempt to divide data into. Positive interger represents the number of batch effect group you want to generate and '-1' will make each batch effect group has average certain number of patients, the default average number of patients is  6.
```

```
histoqcdir: the directory containing histoqc's output.
```
histoqcdir should be the same directory that was passed to the "--output" flag in histoqc. See "--output" in [HistoQC's basic usage section](https://github.com/choosehappy/HistoQC#basic-usage)

### Run

Go to the cohortfinder repository. Download or simply git-cloned, using the following typical command line for running the tool like. And you can also try the other function using the above parameters.

```python
 python3 cohortfinder_colormod_original.py -n 1 "{the HistoQC output path}"
```


### Use the results of CohortFinder

Once you run the CohortFinder, you will get a cohortfinder result file called 'results_cohortfinder.tsv'. You will see two columns, one is called 'groupid', and the other is called 'testind', the testind == 1 represents the patients is partitioned into testing set and testind == 0 represents the patient is partitioned into the training set. You can simply use that patient partitioning results  to set up the training set and test/val set for your machine learning model!

CohortFinder produces the following ouput file structure:
```
outputdir/ (default is histoqc output directory)
    ... (histoqc output, including results.tsv)
    cohortfinder_output_DATE_TIME/
        results_cohortfinder.tsv
        cohortfinder.log
        plots/
            embed.png
            embed_split.png
            embed_by_label.png (conditional)
            embed_by_site.png (conditional)
            group_0.png
            ...
            group_N.png
            allgroups.png
```


# Outputs of CohortFinder

#### 1. Result file and running log

The results_cohortfinder.tsv has four more columns than the histoqc results.tsv file:
1. **groupid**: the batch effect group assigned to the patient by cohortfinder.
2. **testind**: the testing/training set assignment, where "1" patients were assigned to the testing set and "0" patients were assigned to the training set.
3. **embed_x**: the UMAP embedding x coordinates.
4. **embed_y**: the UMAP embedding y coordinates.


#### 2. Embeded plots

Each point represents a patient and different colors represent different batch effect groups

<img src="/figs/embed.png" alt="embed" style="zoom:25%;" />

#### 3. Patient partition plot

 'x' represents the patients were split into training set and '+' means the patients were partitioned into testing set. You can also find the patients information detail in the ***results_cohortfinder.tsv*** file.

<img src="/figs/embed_split.png" alt="embed_split" style="zoom:25%;" />

#### 4. The visual cluster results

![groupingresults](/figs/groupingresults.png)


# Citation
---
Please use below to cite this paper if you find this repository useful or if you use the software shared here in your research.

```
  @misc{fan2023cohortfinder,
      title={CohortFinder: an open-source tool for data-driven partitioning of biomedical image cohorts to yield robust machine learning models}, 
      author={Fan Fan and Georgia Martinez and Thomas Desilvio and John Shin and Yijiang Chen and Bangchen Wang and Takaya Ozeki and Maxime W. Lafarge and Viktor H. Koelzer and Laura Barisoni and Anant Madabhushi and Satish E. Viswanath and Andrew Janowczyk},
      year={2023},
      eprint={2307.08673},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

[CohortFinder Paper link](https://arxiv.org/abs/2307.08673)
