import os
import glob
import pickle
import numpy as np
from ahoproc_tools.io import read_aco_file


TR_GUIA="../data/LibriSpeech/libri_tr.scp"
MFCC_STATS="stats/librispeech_mfcc13s160d1d2.stats"
DATA_ROOT="../data/LibriSpeech/mfcc13_s160_d1d2"
ORDER=39

with open(TR_GUIA) as tr_guia_f:
    tr_files = [l.rstrip() for l in tr_guia_f]

X = []
for ii, tr_file in enumerate(tr_files, start=1):
    print('Reading mfcc {}/{}...'.format(ii, len(tr_files)))
    bname = os.path.splitext(tr_file)[0]
    tr_fpath = os.path.join(DATA_ROOT, bname + '.mfcc')
    cc = read_aco_file(tr_fpath, (-1, ORDER))
    X.append(cc)

X = np.concatenate(X, axis=0)
with open(MFCC_STATS, 'wb') as stats_f:
    x_m = list(np.mean(X, axis=0))
    x_std = list(np.std(X, axis=0))
    pickle.dump({'mean':x_m, 'std':x_std}, stats_f)