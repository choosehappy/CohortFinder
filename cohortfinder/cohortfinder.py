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
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score

def batcheffecttester(data, columnName, datasub, name):
    """
    Batch effect test function between the selected measurements and the user-given label/site information.

    Parameters:
    - data (pandas.DataFrame): The dataframe of the raw result.tsv file.
    - columnName (str): The column name for the user-given label/site information (e.g., siteid/labelid).
    - datasub (pandas.DataFrame): The standardized dataframe with the selected measurements.
    - name (str): Batch effect test based on Site or Label.

    Returns:
    - None: All the results are recorded in the log file, including:
      - Which site(s)/label(s) are significantly related to the batch effects.
      - Which given measurements drive the batch effect.
    """
    col = data[columnName]
    logging.info(f"Starting  Batch Effect {name} test...") #--- write to file 
    df_true=pd.DataFrame(index=set([str(s) for s in col])) #convert sites/labels to strings and then use set to get unique list
    df_rand=pd.DataFrame(index=set([str(s) for s in col])) #convert sites/labels to strings and then use set to get unique list

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

def check_for_column(columname, data,hqc_results_tsv_path):
    """
    Check the if the column exists in the raw tsv file.

    Parameters:
    - column_name (str): The column name.
    - data (pandas.DataFrame): The raw data frame structure.
    - hqc_results_tsv_path (str): The complete path for the histoqc or mrqy result TSV file.

    Returns:
    - Returns the column name if it exists in the raw data.
    - Returns an error
    """

    if columname in data.columns:
        col = columname
        logging.info(f"Column {col} found")
        return col
    else:
        error_message = f"Column {columname} *NOT* found, please check if the column named {columname} is in the {hqc_results_tsv_path} file."
        logging.error(error_message)
        raise ValueError(error_message)

def draw_plot(embedding,pred,cmap,linewidths,plots_outdir,suffix,testind=None):
    """
    Generate and save a scatter plot based on input data.

    Parameters:
    - embedding (numpy.ndarray): 2D array representing the embedding data.
    - pred (numpy.ndarray): Array of predicted labels.
    - cmap (str): Colormap for the scatter plot.
    - linewidths (float or None): Optional linewidth for scatter plot markers.
    - plots_outdir (str): Directory to save the generated plot.
    - suffix (str): Suffix to be used in the saved plot filename.
    - testind (numpy.ndarray or None): Indices for marking specific points-split test/train (optional).

    Returns:
    - None

    Description:
    This function generates a scatter plot based on the provided embedding and prediction data.
    If 'testind' is specified, different markers are used for selected and non-selected points.
    The resulting plot is saved in the specified 'plots_outdir' with the given 'suffix'.
    """

    plt.figure(figsize=(20, 20))
    if testind.empty:
        if linewidths != None:
            plt.scatter(embedding[:, 0], embedding[:, 1], c=pred, cmap=cmap, linewidths=linewidths)
        else:
            plt.scatter(embedding[:, 0], embedding[:, 1], c=pred, cmap=cmap)
        plt.savefig(os.path.join(plots_outdir, f'{suffix}.png'))
    else:
        plt.scatter(embedding[testind, 0], embedding[testind, 1], c=pred[testind], cmap=cmap, marker='+')
        plt.scatter(embedding[~testind, 0], embedding[~testind, 1], c=pred[~testind], cmap=cmap, marker='x')
        plt.savefig(os.path.join(plots_outdir, f'{suffix}.png'))
    return

def batch_effect_score_calculation(output,preds):
    sil_score = silhouette_score(output[['embed_x', 'embed_y']], preds)
    db_score = davies_bouldin_score(output[['embed_x', 'embed_y']], preds)
    ch_score = calinski_harabasz_score(output[['embed_x', 'embed_y']], preds)
    return sil_score,db_score,ch_score

def runCohortFinder(args):
    """
    Using the output tsv file from HistoQC/MRQy, produce a new tsv file containing four new columns: embed_x, embed_y, groupid, testind

    Unless the disable_save flag is set, this function will also produce the following file structure in the directory containing the input results.tsv file:

    outdir/ (histoqc/mrqy output directory)
        results.tsv
        ... (histoqc/mrqy output folders)
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
        args: A namespace (like argparse.Namespace) containing the arguments for CohortFinder.
    Returns:
        output: A pandas dataframe containing the input data and the cohortfinder results.
        preds: A list of integers representing the cluster assignments.
    """

    outdir = os.path.dirname(args.resultsfilepath)


    # --- check if the input histoqc/mrqy directory and/or results.tsv file exist(s)
    if not (os.path.exists(outdir) and os.path.exists(args.resultsfilepath)):
        error_message = f'"The input histoqc/mrqy directory ({outdir}/) or {args.resultsfilepath} does not exist! ' \
                        f'Please make sure there is no typo in the input directory {outdir} and ' \
                        f'make sure the {outdir}/ or {outdir}/result.tsv exists!"'
        raise ValueError(error_message)


    cf_outdir = os.path.join(outdir, "cohortfinder_output_" + time.strftime("%Y%m%d-%H%M%S"))

    os.mkdir(cf_outdir)

    # --- setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    results_outdir = os.path.join(cf_outdir, "results_cohortfinder.tsv")
    log_outdir = os.path.join(cf_outdir, "cohortfinder.log")
    plots_outdir = os.path.join(cf_outdir, "plots")
    os.mkdir(plots_outdir)




    file = logging.FileHandler(filename=log_outdir)
    file.setLevel(logging.INFO)
    file.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(file)


    # --- setup seed for reproducability
    seed = args.randomseed if (args.randomseed) else random.randrange(sys.maxsize) % (2**32)  # get a random seed so that we can reproducibly do the cross validation setup

    random.seed(seed)  # set the seed
    np.random.seed(seed)
    logging.info(f"random seed (note down for reproducibility): {seed}")
    quality_control_tool = args.quality_control_tool

    coluse = args.cols.split(",")
    if quality_control_tool == 'histoqc':
        col_name_filename = "#dataset:filename"
        data = pd.read_csv(args.resultsfilepath, sep='\t', header=5)
    elif quality_control_tool == 'mrqy':
        col_name_filename = "#dataset:Patient"
        data = pd.read_csv(args.resultsfilepath, sep='\t', header=2)
    else:
        error_message = f'the input quality control tool name is in an incorrect version, please make sure use either of the following option: histoqc or mrqy!'
        raise ValueError(error_message)
    logging.info(data)
    logging.info(f'Number of slides:\t{len(data)}')

    labelcol = None
    if args.labelcolumn:
        labelcol = check_for_column(args.labelcolumn, data,args.resultsfilepath)

    sitecol = None
    if args.sitecolumn:
        sitecol = check_for_column(args.sitecolumn, data, args.resultsfilepath)

    pidcol = None
    if args.patientidcolumn and  args.patientidcolumn in data.columns:
            pidcol = args.patientidcolumn
            pids = data[pidcol]
            logging.info(f"Patient column {pidcol} found")
    else:
            logging.warning(f"Patient column {pidcol} *NOT* found, assuming no duplicates")
            pids=np.arange(len(data))

    # --- choose the number of clusters
    # if args.nclusters is set to -1, the number of clusters is implicitly computed such that each batch-effect group has an average of 6 patients.
    # for example, if a cohort has 120 patients and args.nclusters is set to -1, then the number of clusters will be computed as 20.
    if args.nclusters == -1:
        nslides = len(data)
        nclusters = int(nslides // 6)  # nclusters: each batch-effect group has average 6 patients
        logging.info(f"Number of clusters implicitly computed to be:\t{nclusters}")    
    else:
        nclusters = args.nclusters
        logging.info(f"Number of clusters explicitly set on command line to:\t{nclusters}")

    for col in coluse:
        check_for_column(col,data,args.resultsfilepath)

    logging.info(data[coluse].describe())

    datasub = data[coluse]
    datasub = preprocessing.scale(datasub)


    # --- add patch embeddings from quick annotator
    if args.batcheffectsitetest and sitecol:
        batcheffecttester(data, sitecol, datasub, 'Site')
    if args.batcheffectlabeltest and labelcol:
        batcheffecttester(data, labelcol, datasub, 'Label')


    # --- back to cohort finder

    embedding = umap.UMAP(metric="correlation",random_state=seed).fit_transform(datasub)
    clustered = KMeans(n_clusters=nclusters,random_state=seed).fit(embedding)
    #clustered = SpectralClustering(n_clusters=nclusters).fit(embedding)
    preds = clustered.labels_


    # --- save embedding plot
    if not args.disable_save:
        cmap=matplotlib.colors.ListedColormap( matplotlib.cm.get_cmap('Set1').colors+ matplotlib.cm.get_cmap('Set2').colors+  matplotlib.cm.get_cmap('Set3').colors  )
        draw_plot(embedding, pred=preds, cmap=cmap, linewidths=5, plots_outdir=plots_outdir, suffix='embed',testind=pd.Series([]))
        logging.info("The embedding plot was successfully saved!")
    logging.info(Counter(preds))


    # --- setup output dataframe with histoqc/mrqy header
    output = pd.DataFrame(data=datasub, columns=coluse)

    output.insert(0, col_name_filename, data[col_name_filename])




    if labelcol:
        # --- add label information to the dataframe
        output["label"] = data[labelcol]

        # --- save label embedding plot
        if not args.disable_save:
            # --- convert labels to integers starting from 0 for plotting
            labellookup={ v:i for i,v in enumerate(set(data[labelcol]))}
            labelids = [labellookup[s] for s in data[labelcol]]
            nlabels = len(labellookup)
            draw_plot(embedding, pred=labelids, cmap=cmap, linewidths=None, plots_outdir=plots_outdir, suffix='embed_by_label',testind=pd.Series([]))
            logging.info("The label embedding plot was successfully saved!")
        logging.info(Counter(data[labelcol]))
        
        
    if sitecol:
        # --- add site information to the dataframe
        output["site"] = data[sitecol]

        # --- save site embedding plot
        if not args.disable_save:
            # --- convert sites to integers starting from 0 for plotting
            sitelookup={ v:i for i,v in enumerate(set(data[sitecol]))}
            siteids = [sitelookup[s] for s in data[sitecol]]
            nsites = len(sitelookup)

            draw_plot(embedding, pred=siteids, cmap=cmap, linewidths=None, plots_outdir=plots_outdir,
                      suffix='embed_by_site',testind=pd.Series([]))
            logging.info("The site embedding plot was successfully saved!")
        logging.info(Counter(data[sitecol]))
        

    # --- add new data to the dataframe
    output["embed_x"] = embedding[:, 0]
    output["embed_y"] = embedding[:, 1]
    output["groupid"] = preds
    output["testind"] = None

    # --- calculate batch-effect scores
    sil_score, db_score, ch_score = batch_effect_score_calculation(output,preds)

    output["sil_score"] = sil_score
    output["db_score"] = db_score
    output["ch_score"] = ch_score

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

    
    if not args.disable_save:
        # --- plot training testing split. + is test, x is train
        testind = output["testind"] == True
        draw_plot(embedding,pred=preds,cmap=cmap, linewidths=None,plots_outdir=plots_outdir,suffix='embed_split',testind=testind)
        logging.info("The training/testing split embedding plot was successfully saved!")
        # ------------------------- WRITE TSV OUTPUT ------------------------- #
        with open(results_outdir, 'w') as cf_out:
            with open(args.resultsfilepath, 'r') as f:
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
            cf_out.write(f"BE SCORE: silhouette_score={sil_score} | davies_bouldin_score={db_score} | calinski_harabasz_score={ch_score}\n")
            output.to_csv(cf_out, sep="\t", lineterminator='\n', index=False)


        # ------------------------- MAKE GROUP PLOTS ------------------------- #
        basedir = os.path.dirname(args.resultsfilepath)
        # Visualize cluster results with thumbnails in the contact sheet.
        # Set the number of thumbnails in each row to 5 by default.
        # 'ngroupsof5' represents the number of rows for the visualization results.
        ngroupsof5 = 3
        for gid in np.unique(preds):
            fig, axs = plt.subplots(ngroupsof5, 5, figsize=(20, 20))
            axs = list(axs.flatten())

            fnamessub = list(output[col_name_filename][gid == preds])
            fnamessub = random.sample(fnamessub, ngroupsof5 * 5) if len(fnamessub) > ngroupsof5 * 5 else fnamessub

            for fname in fnamessub:
                print(fname)
                if quality_control_tool == 'histoqc':
                    fullfname = glob.glob(f"{basedir}/{fname}/{fname}*thumb*small*")
                elif quality_control_tool=='mrqy':
                    fullfname = glob.glob(f"{basedir}/{fname}/*.png")
                if (len(fullfname) ==0):
                    error_message = f"There is *NO* thumbnail images in the subfolders of {basedir}/{fname}/. Please check if the input histoqc/mrqy folder is complete. "
                    logging.error(error_message)
                    raise ValueError(error_message)
                # print(hqc_results_tsv)
                else:
                    logging.info(f"This is the filename name: {fullfname}")
                    # print(f"This is the filename name: {basedir}")
                    io = cv2.cvtColor(cv2.imread(fullfname[0]), cv2.COLOR_BGR2RGB)
                    axs.pop().imshow(io)

            plt.savefig(os.path.join(plots_outdir, f'group_{gid}.png'))
            plt.close(fig)


        # ------------------------- MAKE OVERVIEW PLOT ------------------------- #

        fig, axs = plt.subplots(int(np.ceil(len(np.unique(preds)) / 5)), 5, figsize=(20, 20))
        axs = list(axs.flatten())
        for gid in np.unique(preds):
            fnamessub = list(output[col_name_filename][gid == preds])
            fname = random.sample(fnamessub, 1)[0]

            if quality_control_tool == 'histoqc':
                fullfname = glob.glob(f"{basedir}/{fname}/{fname}*thumb*small*")
            elif quality_control_tool == 'mrqy':
                fullfname = glob.glob(f"{basedir}/{fname}/*.png")
            io = cv2.cvtColor(cv2.imread(fullfname[0]), cv2.COLOR_BGR2RGB)
            axs.pop().imshow(io)

        plt.savefig(os.path.join(plots_outdir, f'allgroups.png'))
        plt.close(fig)
    logging.info(
        f'*****BE SCORE: silhouette_score={sil_score} | davies_bouldin_score={db_score} | calinski_harabasz_score={ch_score}*****')
    logging.info(f'CohortFinder has run successfully!')

    return output
