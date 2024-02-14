

# CohortFinder

---
Intelligent data partitioning using quality control metrics



# Purpose

---
Batch effects (BE) (e.g., scanner, stain variances), are systematic technical differences in data creation unrelated to biological variation. BEs have been shown to negatively impact machine learning (ML) model generalizability. Since they can result in the worst case when partioning patients into training/validation set, where patients in training set come from totally different BE groups from those in validation set. The purpose of the CohortFinder is to provide an intelligent data partition strategy trying to avoid the worst case situation without any manual effort.

CohortFinder has the following functionality:

1. Cluster patients into different BE groups using quality control metrics
2. Partition patients into training/validation set, making sure the patients in training or validation set come from all the BE groups

This tool can increase the performance and generalizability of machine learning model.



# Requirements

---
Tested with Python 3.8.18 and 3.9.18

Requires:
1. Python 
2. pip

And the following python packages:
1. matplotlib

2. numpy

3. opencv-python-headless

4. scikit-learn

5. scipy

6. umap_learn

7. pandas

   

# Installation
### 1. Clone the CohortFinder github repository
```
git clone https://github.com/choosehappy/CohortFinder.git
```

### 2. (optional) Create a virtual environment
A python virtual environment (https://docs.python.org/3/library/venv.html) is the recommended dependency manager for CohortFinder.
```
cd CohortFinder
python3 -m venv cf_env
source cf_env/bin/activate
```

### 3. Install CohortFinder as a python package
```
pip install .
```



# Quality Control Metrics Generation

Please see  [Histoqc](https://github.com/choosehappy/HistoQC) and [MRQy](https://github.com/ccipd/MRQy/), these are 2 open-source quality control tools for digital pathology slides and imaging data. We use the quality control metrics it generates.



# Basic Usage

---
The parameters CohortFinder used are as below:
```
python3 -m cohortfinder --help
```

```
usage: __main__.py [-h] [-c COLS] [-l LABELCOLUMN] [-s SITECOLUMN] [-p PATIENTIDCOLUMN] [-t TESTPERCENT] [-b] [-y] [-r RANDOMSEED] [-q] [-n NCLUSTERS]
                   resultsfilepath

Split histoqc/mrqy tsv into training and testing

positional arguments:
  resultsfilepath       The full path to the HistoQC/MRQy output file. This argument is required.

options:
  -h, --help            show this help message and exit
  -c COLS, --cols COLS  columns to use for clustering, comma seperated
  -l LABELCOLUMN, --labelcolumn LABELCOLUMN
                        column name associated with a 0,1 label
  -s SITECOLUMN, --sitecolumn SITECOLUMN
                        column name associated with site variable
  -p PATIENTIDCOLUMN, --patientidcolumn PATIENTIDCOLUMN
                        column name associated with patient id, ensuring slides are grouped
  -t TESTPERCENT, --testpercent TESTPERCENT
  -b, --batcheffectsitetest
  -y, --batcheffectlabeltest
  -r RANDOMSEED, --randomseed RANDOMSEED, for reproducing the same results for UMAP, k-means and data partitioning
  -q, --disable_save    Run silently, do not save any files.
  -d, --quality_control_tool Which quality tool is used here: HistoQC or MRQy (--histoqc/ --mrqy)
  -n NCLUSTERS, --nclusters NCLUSTERS
                        Number of clusters to attempt to divide data into before splitting into cohorts, default -1 of negative 1 makes best guess
```

Example run command:
``` python
python3 -m cohortfinder -n 4 -t 0.3 "/full/path/to/your/results.tsv"
```
Replace the filepath with a real file path, for example, we upload some sample data into the path "/test/histoqc_outdir/".
You can do a quick test by using the following command

``` python
python3 -m cohortfinder -n 3 -t 0.3 -r 200  "/cohortfinder/test/histoqc_outdir/results.tsv"
```



### Parameter meaning :
#### HistoQC
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

#### MRQy
```
-c: metrics calculated by MRQy we used for batch effect group generation, the default metrics are:
"MEAN,RNG,VAR,CV,CPP,PSNR,SNR1,SNR2,SNR3,SNR4,CNR,CVP,CJV,EFC,FBER"
```
| Quality control metric | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| MEAN                   | Mean of the foreground   |
| RNG                    | Range of the foreground   |
| VAR                    | Variance of the foreground |
| CV                     | Coefficient of variation of the foreground for shadowing and inhomogeneity artifacts |
| CPP                    | Contrast  per  pixel:  mean  of  the  foreground  filtered  by  a  3×3  2D  Laplacian  kernel  for  shadowing  artifacts |
| PSNR                   | Peak signal to noise ratio of the foreground  |
| SNR1                   | Foreground standard deviation (SD) divided by background SD |
| SNR2                   | Mean of the foreground patch divided by background SD |
| SNR3                   | Foreground patch SD divided by the centered foreground patch SD |
| SNR4                   | Mean of the foreground patch divided by mean of the background patch|
| CNR                    | Contrast to noise ratio for shadowing and noise artifacts |
| CVP                    | Coefficient of variation of the foreground patch for shading artifacts: foreground patch SD divided by foreground patch mean |
| CJV                    | Coefficient of joint variation between the foreground and background for aliasing and inhomogeneity artifacts |
| EFC                    | Entropy focus criterion for motion artifacts |
| FBER                   | Foreground-background energy ratio for ringing artifacts |


# CohortFinder Output File Structure

Once you run the CohortFinder, you will get a cohortfinder result file called 'results_cohortfinder.tsv'. You will see two columns, one is called 'groupid', and the other is called 'testind', the testind == 1 represents the patients is partitioned into testing set and testind == 0 represents the patient is partitioned into the training set. You can simply use that patient partitioning results  to set up the training set and test/val set for your machine learning model!

CohortFinder produces the following ouput file structure:
```
outputdir/ (default is histoqc/mrqy output directory)
    ... (histoqc/mrqy output, including results.tsv)
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

The results_cohortfinder.tsv has four more columns than the histoqc/mrqy results.tsv file:
1. **groupid**: the batch effect group assigned to the patient by cohortfinder.
2. **testind**: the testing/training set assignment, where "1" patients were assigned to the testing set and "0" patients were assigned to the training set.
3. **embed_x**: the UMAP embedding x coordinates.
4. **embed_y**: the UMAP embedding y coordinates.


#### 2. Embedding plots

Each point represents a patient and different colors represent different batch effect groups

<img src="/cohortfinder/figs/embed.png" alt="embed" style="zoom:25%;" />

#### 3. Patient partition plot

 'x' represents the patients were split into training set and '+' means the patients were partitioned into testing set. You can also find the patients information detail in the ***results_cohortfinder.tsv*** file.

<img src="/cohortfinder/figs/embed_split.png" alt="embed_split" style="zoom:25%;" />

#### 4. The visual cluster results

<img src="/cohortfinder/figs/groupingresults.png" alt="embed_split" style="zoom:25%;" />


#### 5. BE score

We also introduce three clustering metrics: the silhouette coefficient, the Davies-Bouldin index, and the Calinski-Harabasz index as BE scores. Here are the description of these 3 measurements. The measurements can be found in both cohortfinder tsv file and log file. Better score represents the cohort has severe batch-effect.

| Quality control metric | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| Silhouette Coefficient(mean Silhouette Coefficient over all samples)                   | Measures how similar an object is to its own cluster compared to other clusters. The value ranges from -1 to 1. A high value indicates appropriate clustering.                                     |
| Davies-Bouldin index                    | Measures how similar an object is to its own cluster compared to other clusters. The value ranges from -1 to 1. A high value indicates appropriate clustering.   |
| Calinski-Harabasz index                    | 1. Between-Cluster Dispersion: It measures how far the clusters are from each other. For good clustering, this should be as large as possible. 2.Within-Cluster Dispersion: It measures how compact the clusters are internally. For good clustering, this should be as small as possible. |


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
