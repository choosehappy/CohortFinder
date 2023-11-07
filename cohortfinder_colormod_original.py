from warnings import simplefilter
from sklearn.exceptions import UndefinedMetricWarning
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UndefinedMetricWarning)

import argparse
import glob
import logging
import os
import random
import sys
import time
from collections import Counter

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap.umap_ as umap
from random import shuffle

from scipy.stats import ttest_ind
from sklearn import preprocessing
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix

def batcheffecttester(data, columnName, datasub, name):
    col = data[columnName]
    logging.info(f"Starting  Batch Effect {name} test...") #--- write to file 
    df_true=pd.DataFrame(index=set([str(s) for s in col])) #convert sites to strings and then use set to get unique list
    df_rand=pd.DataFrame(index=set([str(s) for s in col])) #convert sites to strings and then use set to get unique list

    featimport={}
    for df,ver in zip([df_true,df_rand],['true','random']):
        # split train and testing sets
        featimport[ver]=np.zeros([100,datasub.shape[1]])
        for i,fi in zip(range(100),featimport[ver]):
            clf = RandomForestClassifier()
            y_all=list(col)
            if ver=='random':
                shuffle(y_all)

            X_train, X_test, y_train, y_test = train_test_split(datasub, y_all, test_size=0.20)

            #train the classifier
            clf.fit(X_train, y_train)

            #make predictions on the testing set
            preds=clf.predict(X_test)
            cr=classification_report(y_test, preds,output_dict=True)

            newcol={k:cr[k]['f1-score'] for k in cr.keys() if k not in ["accuracy","macro avg","weighted avg"]}
            df[i]=df.index.map(newcol)
            fi[:]=np.argsort(clf.feature_importances_)

    logging.info(f"--------- Batch Effect {name} test results -------") #--- write to file 
    logging.info(f"{name} \t num slides \t p-value  \t sig-ind") #--- write to file 
    for key in df_true.index.sort_values():
        ttestres=ttest_ind(df_true.loc[key],df_rand.loc[key], nan_policy='omit')[1]
        logging.info(f"{key}\t{sum(col==key)}\t{ttestres}\t{'***' if ttestres<.05 else ''}") # line assumes sitecol exists 

    logging.info(f"--------- Batch Effect {name} feature importance results -------") 
    logging.info(f"\t Average+Variance \t Feature name") 
    meanfeatimport=np.mean(featimport['true'],axis=0)
    varfeatimport=np.var(featimport['true'],axis=0)
    for fi in np.argsort(meanfeatimport)[::-1]:
        logging.info(f"\t{meanfeatimport[fi]:.2f} ({varfeatimport[fi]:.2f}) \t\t{datasub.columns[fi]}")     

    logging.info(f"------------------------------------------------") #--- write to file 


def runCohortFinder(args):
    """
    Using the output tsv file from HistoQC, produce a new tsv file containing four new columns: embed_x, embed_y, groupid, testind

    args.outputdir is assumed to be the output directory of HistoQC. The output of this function will be written to args.outputdir/cohortFinder_output

    outputdir/
        ... (histoqc output files)
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


    Args:
        args: A namespace (like that returned by argparse) containing the arguments for CohortFinder.
    Returns:
        output: A pandas dataframe containing the input data and the cohortfinder results.
        preds: A list of integers representing the cluster assignments.
    """


    # --- set up output file structure
    hqc_results_tsv_path = os.path.join(args.histoqcdir, "results.tsv")

    if args.outdir is None: # default behavior is to write to the same directory as the histoqc output
        args.outdir = args.histoqcdir

    cf_outdir = os.path.join(args.outdir, "cohortfinder_output_" + time.strftime("%Y%m%d-%H%M%S"))

    os.mkdir(cf_outdir)
    results_outdir = os.path.join(cf_outdir, "results_cohortfinder.tsv")
    log_outdir = os.path.join(cf_outdir, "cohortfinder.log")
    plots_outdir = os.path.join(cf_outdir, "plots")
    os.mkdir(plots_outdir)
    


    # --- setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    file = logging.FileHandler(filename=log_outdir)
    file.setLevel(logging.INFO)
    file.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(file)


    # --- setup seed for reproducability
    seed = args.randomseed if (args.randomseed) else random.randrange(
        sys.maxsize)  # get a random seed so that we can reproducibly do the cross validation setup
    random.seed(seed)  # set the seed
    np, random.seed(seed)
    logging.info(f"random seed (note down for reproducibility): {seed}")

    coluse = args.cols.split(",")

    data = pd.read_csv(hqc_results_tsv_path, sep='\t', header=5)
    logging.info(data)
    logging.info(f'Number of slides:\t{len(data)}')

    labelcol = None
    if args.labelcolumn:
        if args.labelcolumn in data.columns:
            labelcol = args.labelcolumn
            logging.info(f"Label column {labelcol} found")
        else:
            logging.warning(f"Label column {labelcol} *NOT* found")

    sitecol = None
    if args.sitecolumn:
        if args.sitecolumn in data.columns:
            sitecol = args.sitecolumn
            data[sitecol] = data[sitecol].astype(str)
            logging.info(f"Site column {sitecol} found")
        else:
            logging.warning(f"Site column {sitecol} *NOT* found")


    pidcol = None
    if args.patiendidcolumn and  args.patiendidcolumn in data.columns:
            pidcol = args.patiendidcolumn
            pids = data[pidcol]
            logging.info(f"Patient column {pidcol} found")
    else:
            logging.warning(f"Patient column {pidcol} *NOT* found, assuming no duplicates")
            pids=np.arange(len(data))


    # --- choose number of clusters
    if args.nclusters == -1:
        nslides = len(data)
        nclusters = int(nslides // 6)  # every group has average three patients
        logging.info(f"Number of clusters implicitly computed to be:\t{nclusters}")    
    else:
        nclusters = args.nclusters
        logging.info(f"Number of clusters explicitly set on command line to:\t{nclusters}")    


    logging.info(data[coluse].describe())

    datasub = data[coluse]
    datasub = preprocessing.scale(datasub)


    # --- add patch embeddings from quick annotator
    if args.batcheffectsitetest and sitecol:
        batcheffecttester(data, sitecol, datasub, 'Site')
    if args.batcheffectlabeltest and labelcol:
        batcheffecttester(data, labelcol, datasub, 'Label')


    # --- back to cohort finder
    embedding = umap.UMAP(metric="correlation").fit_transform(datasub)
    clustered = KMeans(n_clusters=nclusters).fit(embedding)
    #clustered = SpectralClustering(n_clusters=nclusters).fit(embedding)
    preds = clustered.labels_


    # --- setup colormap
    cmap=matplotlib.colors.ListedColormap( matplotlib.cm.get_cmap('Set1').colors+ matplotlib.cm.get_cmap('Set2').colors+  matplotlib.cm.get_cmap('Set3').colors  )

    plt.figure(figsize=(20, 20))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=preds, cmap=cmap, linewidths=5)
    plt.savefig(os.path.join(plots_outdir, 'embed.png'))
    logging.info(Counter(preds))


    # --- setup output dataframe with histoqc header
    output = pd.DataFrame(data=datasub, columns=coluse)
    output.insert(0, "#dataset:filename", data["#dataset:filename"])


    if labelcol:
        # --- add label information to the dataframe
        output["label"] = data[labelcol]


        # --- convert labels to integers starting from 0 for plotting
        labellookup={ v:i for i,v in enumerate(set(data[labelcol]))}
        labelids = [labellookup[s] for s in data[labelcol]]
        nlabels = len(labellookup)
        
        
        plt.figure(figsize=(20, 20))
        plt.scatter(embedding[:, 0], embedding[:, 1], c=labelids, cmap=cmap)
        plt.savefig(os.path.join(plots_outdir, 'embed_by_label.png'))
        logging.info(Counter(data[labelcol]))
        
        
    if sitecol:
        # --- add site information to the dataframe
        output["site"] = data[sitecol]


        # --- convert sites to integers starting from 0 for plotting
        sitelookup={ v:i for i,v in enumerate(set(data[sitecol]))}
        siteids = [sitelookup[s] for s in data[sitecol]]
        nsites = len(sitelookup)
        
        plt.figure(figsize=(20, 20))
        plt.scatter(embedding[:, 0], embedding[:, 1], c=siteids, cmap=cmap)
        plt.savefig(os.path.join(plots_outdir, 'embed_by_site.png'))
        logging.info(Counter(data[sitecol]))
        

    # --- add new data to the dataframe
    output["embed_x"] = embedding[:, 0]
    output["embed_y"] = embedding[:, 1]
    output["groupid"] = preds
    output["testind"] = None


    # --- assign test or train status
    labels = data[labelcol] if labelcol else pd.Series(np.zeros(len(preds)))
    sites = data[sitecol] if sitecol else pd.Series(np.zeros(len(preds)))
    patient_lookup={}

    for gid in np.unique(preds):
        for sid in pd.unique(sites):
            for lid in pd.unique(labels):
                idx = np.where((gid == preds) & (labels == lid) & (sid == sites))[0]
                if len(idx)==0:
                    continue

                #testidx = np.random.rand(len(idx))<args.testpercent
                sortidx=np.argsort(np.random.rand(len(idx)))
                testidx = np.zeros(len(sortidx),dtype=bool)
                testids = sortidx[0:int(np.ceil(len(sortidx)*args.testpercent))]
                testidx[testids]=True
                
                testidx = np.asarray([patient_lookup.get(pids[i],t) for i,t in zip(idx,testidx) ])
                
                output.loc[idx[~testidx], "testind"] = 0
                output.loc[idx[testidx], "testind"] = 1
                
                patient_lookup.update({pids[i]:t for i,t in zip(idx,testidx)})

    logging.info(f'Num in training set: {np.sum(output["testind"]==0)}')
    logging.info(f'Num in testing set: {np.sum(output["testind"]==1)}')
    logging.info(f'Percent in testing set: {np.mean(output["testind"])}')


    # --- plot training testing split. + is test, x is train
    plt.figure(figsize=(20, 20))
    testind = output["testind"] == True
    plt.scatter(embedding[testind, 0], embedding[testind, 1], c=preds[testind], cmap=cmap, marker='+')
    plt.scatter(embedding[~testind, 0], embedding[~testind, 1], c=preds[~testind], cmap=cmap, marker='x')
    plt.savefig(os.path.join(plots_outdir, 'embed_split.png'))


    # ------------------------- WRITE TSV OUTPUT ------------------------- #
    with open(results_outdir, 'w') as cf_out:
        with open(hqc_results_tsv_path, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    cf_out.write(line.replace("dataset:filename", "prevcols:filename"))
                else:
                    break
        cf_out.write(f"#{args}\n")
        cf_out.write(f"#sitecol:{sitecol}\n")
        cf_out.write(f"#labelcol:{labelcol}\n")

        colorlist = ['#%02x%02x%02x' % (x[0], x[1], x[2]) for x in (np.asarray(cmap.colors) * 255).astype(np.uint8)]
        cf_out.write(f"#colorlist:{','.join(list(colorlist))}\n")
        output.to_csv(cf_out, sep="\t", line_terminator='\n', index=False)


    # ------------------------- MAKE GROUP PLOTS ------------------------- #
    basedir = os.path.dirname(hqc_results_tsv_path)
    ngroupsof5 = 3
    for gid in np.unique(preds):
        fig, axs = plt.subplots(ngroupsof5, 5, figsize=(20, 20))
        axs = list(axs.flatten())

        fnamessub = list(output["#dataset:filename"][gid == preds])
        fnamessub = random.sample(fnamessub, ngroupsof5 * 5) if len(fnamessub) > ngroupsof5 * 5 else fnamessub

        for fname in fnamessub:
            print(fname)
            fullfname = glob.glob(f"{basedir}/**/{fname}*thumb*small*")
            # print(hqc_results_tsv)
            print(f"This is the filename name: {basedir}")
            io = cv2.cvtColor(cv2.imread(fullfname[0]), cv2.COLOR_BGR2RGB)
            axs.pop().imshow(io)

        plt.savefig(os.path.join(plots_outdir, f'group_{gid}.png'))
        plt.close(fig)


    # ------------------------- MAKE OVERVIEW PLOT ------------------------- #
    basedir = os.path.dirname(hqc_results_tsv_path)

    fig, axs = plt.subplots(int(np.ceil(len(np.unique(preds)) / 5)), 5, figsize=(20, 20))
    axs = list(axs.flatten())
    for gid in np.unique(preds):
        fnamessub = list(output["#dataset:filename"][gid == preds])
        fname = random.sample(fnamessub, 1)[0]

        fullfname = glob.glob(f"{basedir}/**/{fname}*thumb*small*")
        io = cv2.cvtColor(cv2.imread(fullfname[0]), cv2.COLOR_BGR2RGB)
        axs.pop().imshow(io)

    plt.savefig(os.path.join(plots_outdir, f'allgroups.png'))
    plt.close(fig)


    return output, preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split histoqc tsv into training and testing')
    parser.add_argument('-c', '--cols', help="columnts to use for clustering, comma seperated", type=str,
                        default="mpp_x,mpp_y,michelson_contrast,rms_contrast,grayscale_brightness,chan1_brightness,chan2_brightness,"
                                "chan3_brightness,chan1_brightness_YUV,chan2_brightness_YUV,chan3_brightness_YUV")
    parser.add_argument('-l', '--labelcolumn', help="column name associated with a 0,1 label", type=str, default=None)
    parser.add_argument('-s', '--sitecolumn', help="column name associated with site variable", type=str, default=None)
    parser.add_argument('-p', '--patiendidcolumn', help="column name associated with patient id, ensuring slides are grouped", type=str, default=None)
    parser.add_argument('-t', '--testpercent', type=float, default=.2)
    parser.add_argument('-b', '--batcheffectsitetest', action="store_true")
    parser.add_argument('-y', '--batcheffectlabeltest', action="store_true")
    parser.add_argument('-r', '--randomseed', type=int, default=None)
    parser.add_argument('-o', '--outdir', type=str,
                        default=None)  # --- change to the same output directory as histoqc output so that UI can refind it without looking else where

    parser.add_argument('-n', '--nclusters', type=int, default=-1, help="Number of clusters to attempt to divide data into before splitting into cohorts, default -1 of negative 1 makes best guess")
    parser.add_argument('histoqcdir', help="The directory containing the output of HistoQC", type=str)
    # -- add batch effect test
    args = parser.parse_args()
    print(args)


    # ------------------------- RUN COHORTFINDER ------------------------- #
    output, preds = runCohortFinder(args)

    logging.shutdown()


