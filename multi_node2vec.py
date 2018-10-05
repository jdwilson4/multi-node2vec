'''
Wrapper for the multi-node2vec algorithm. 

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
'''
import os
import src as mltn2v
import argparse
import time
###JAMES - todo: add an argument for r and call it in the function args.r set default to [0.25, 0.5, 0.75]###

def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-node2vec on multilayer networks.")

    parser.add_argument('--dir', nargs='?', default='data/brainData/CONTROL_fmt',
                        help='Absolute path to directory of correlation/adjacency matrix files (csv format).')

    parser.add_argument('--output', nargs='?', default='new_results/',
                        help='Absolute path to output directory (no extension).')

    #parser.add_argument('--filename', nargs='?', default='new_results/mltn2v_control',
    #                    help='output filename (no extension).')

    parser.add_argument('--d', type=int, default=100,
                        help='Dimensionality. Default is 100.')

    parser.add_argument('--nbsize', type=int, default=10,
                        help='Neighborhood size. Default is 10.')

    parser.add_argument('--n_samples', type=int, default=1,
                        help='Number of samples per layer. Default is 1.')

    parser.add_argument('--thresh', type=float, default=None,
                        help='Edge weight threshold for neighborhood selection.')

    parser.add_argument('--w2v_iter', default=1, type=int,
                        help='Number of epochs in word2vec')

    parser.add_argument('--w2v_workers', type=int, default=8,
                        help='Number of parallel worker threads. Default is 8.')

    #parser.add_argument('--w', type=float, default=1,
    #                    help='Value of w in neighborhood generation.')

    return parser.parse_args()


def main(args):
    start = time.time()
    # PARSE LAYERS -- THRESHOLD & CONVERT TO BINARY
    # TODO: create option for adj list or csv
    layers = mltn2v.timed_invoke("parsing network layers",
                                 lambda: mltn2v.parse_matrix_layers(args.dir, binary=True, thresh=args.thresh))
    # check if layers were parsed
    if layers:
        wvals = [0.25, 0.5, 0.75]
        # EXTRACT NEIGHBORHOODS
        nbrhd_dict = mltn2v.timed_invoke("extracting neighborhoods",
                                     lambda: mltn2v.extract_neighborhoods_walk(layers, args.nbsize, wvals))
        # GENERATE FEATURES
        out = mltn2v.clean_output(args.output)
        for w in wvals:
            out_path = os.path.join(out, 'w' + str(w) + '/mltn2v_control')
            mltn2v.timed_invoke("generating features",
                                lambda: mltn2v.generate_features(nbrhd_dict[w], args.d, out_path, nbrhd_size=args.nbsize,
                                                                 w2v_iter=args.w2v_iter, workers=args.w2v_workers))

            print("\nCompleted Multilayer Network Embedding for w=" + str(w) + " in {:.2f} secs.\nSee results:".format(time.time() - start))
            print("\t" + out_path + ".csv")
        print("Completed Multilayer Network Embedding for all w values.")
    else:
        print("Whoops!")


if __name__ == '__main__':
    args = parse_args()
    main(args)
