"""
Helper functions for multi-node2vec. (duplicate of mltn2v_utils.py used for testing)
Details can be found in the paper: "Fast Embedding of Multilayer Networks: An Algorithm and Application to Group fMRI" 
by JD Wilson, M Baybay, R Sankar, and P Stillman

Preprint here: https://arxiv.org/pdf/1809.06437.pdf

Contributors:
- Melanie Baybay
University of San Francisco, Department of Computer Science
- Rishi Sankar
Henry M. Gunn High School
- James D. Wilson (maintainer)
University of San Francisco, Department of Mathematics and Statistics

Questions or Bugs? Contact James D. Wilson at jdwilson4@usfca.edu
"""

import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
import time


# -------------------------------------------------------------------------------
# PARSING AND CONVERSION FOR MULTILAYER GRAPHS
# -------------------------------------------------------------------------------
def parse_matrix_layers(network_dir, delim=',', binary=False, thresh=None):
    """
    Converts directory of adjacency matrix files into pandas dataframes.
    :param network_dir: Directory of adjacency matrix files
    :param delim: separator for adjacency matrix
    :param binary: boolean of whether or not to convert edge weights to binary
    :param thresh: threshold for edge weights. Will accepts weights <= thresh
    :return: List of adjacency lists. Each adjacency list is one layer and is represented
            as pandas DataFrames with 'source', 'target', 'weight' columns.
    """
    # expand directory path
    network_dir = expand_path(network_dir)

    # iterate files and convert to pandas dataframes
    layers = []
    for network_file in os.listdir(network_dir):
        file_path = os.path.join(network_dir, network_file)
        try:
            # read as pandas DataFrame, index=source, col=target
            layer = pd.read_csv(file_path, index_col=0)
            if layer.shape[0] != layer.shape[1]:
                print('[ERROR] Invalid adjacency matrix. Expecting matrix with index as source and column as target.')
                return
            if thresh is not None:
                layer[layer <= thresh] = 0
            if binary:
                layer[layer != 0] = 1
            # ensure that index (node name) is string, since word2vec will need it as str
            if is_numeric_dtype(layer.index):
                layer.index = layer.index.map(str)
            # replace all 0s with NaN
            layer.replace(to_replace=0, value=pd.np.nan, inplace=True)
            # convert matrix --> adjacency list with cols ["source", "target", "weight"]
            layer = layer.stack(dropna=True).reset_index()
            # rename columns
            layer.columns = ["source", "target", "weight"]
            layers.append(layer)
        except Exception as e:
            print('[ERROR] Could not read file "{}": {} '.format(file_path, e))
    return layers


def expand_path(path):
    """
    Expands a file path to handle user and environmental variables.
    :param path: path to expand
    :return: expanded path
    """
    new_path = os.path.expanduser(path)
    return os.path.expandvars(new_path)


# -------------------------------------------------------------------------------
# OUTPUT
# -------------------------------------------------------------------------------
def feature_matrix_to_csv(ftrs, filename):
    """
    Convert feature matrix to csv.
    :param matrix: pandas DataFrame  of features
    :param filename: absolute path to output file (no extension)
    :param nodes_on: if True, first column indicates node_id
    :return:
    """
    out = filename + ".csv"
    ftrs.to_csv(out, sep=',', header=False)
    return


def timed_invoke(action_desc, method):
    """
    Invokes a method with timing.
    :param action_desc: The string describing the method action
    :param method: The method to invoke
    :return: The return object of the method
    """
    print('Started {}...'.format(action_desc))
    start = time.time()
    try:
        output = method()
        print('Finished {} in {} seconds'.format(action_desc, int(time.time() - start)))
        return output
    except Exception:
        print('Exception while {} after {} seconds'.format(action_desc, int(time.time() - start)))
        raise


def clean_output(directory):
    """
    Checks if output directory exists, otherwise it is created.
    """
    directory = expand_path(directory)
    if os.path.isdir(directory):
        return directory
    else:
        os.makedirs(directory)
        print("[WARNING] Directory not found. Created {}".format(directory))
        return directory
