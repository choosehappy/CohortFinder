from warnings import simplefilter
from sklearn.exceptions import UndefinedMetricWarning
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UndefinedMetricWarning)

import argparse
import logging
from cohortfinder import runCohortFinder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split histoqc/mrqy tsv into training and testing')
    parser.add_argument('-c', '--cols', help="columns to use for clustering, comma seperated", type=str,
                        default="mpp_x,mpp_y,michelson_contrast,rms_contrast,grayscale_brightness,chan1_brightness,chan2_brightness,"
                                "chan3_brightness,chan1_brightness_YUV,chan2_brightness_YUV,chan3_brightness_YUV")
    #metrics used for MRQy:
    #MEAN,RNG,VAR,CV,CPP,PSNR,SNR1,SNR2,SNR3,SNR4,CNR,CVP,CJV,EFC,FBER
    parser.add_argument('-l', '--labelcolumn', help="column name associated with a 0,1 label", type=str, default=None)
    parser.add_argument('-s', '--sitecolumn', help="column name associated with site variable", type=str, default=None)
    parser.add_argument('-p', '--patientidcolumn', help="column name associated with patient id, ensuring slides are grouped", type=str, default=None)
    parser.add_argument('-t', '--testpercent', type=float, default=.2)
    parser.add_argument('-b', '--batcheffectsitetest', action="store_true")
    parser.add_argument('-y', '--batcheffectlabeltest', action="store_true")
    parser.add_argument('-r', '--randomseed', type=int, default=None)
    parser.add_argument('-q', '--disable_save', action="store_true", help="Run silently, do not save any files.")
    parser.add_argument('-d', '--quality_control_tool', type=str, default='histoqc',help="Which quality tool is used here: HistoQC or MRQy (--histoqc/ --mrqy)")
    parser.add_argument('-n', '--nclusters', type=int, default=-1, help="Number of clusters to attempt to divide data into before splitting into cohorts, default -1 of negative 1 makes best guess")
    parser.add_argument('resultsfilepath', help="The full path to the HistoQC/MRQy output file. This argument is required.", type=str)
    # -- add batch effect test
    args = parser.parse_args()
    print(args)

    # ------------------------- RUN COHORTFINDER ------------------------- #
    output = runCohortFinder(args)

    logging.shutdown()


