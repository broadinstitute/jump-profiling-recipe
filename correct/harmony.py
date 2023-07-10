'''Correction methods'''
import logging

from harmonypy import run_harmony
from preprocessing import io

logger = logging.getLogger(__name__)


def harmony(dframe_path, batch_key, output_path):
    '''Harmony correction'''
    meta, feats, features = io.split_parquet(dframe_path)
    harmony_out = run_harmony(feats,
                              meta,
                              batch_key,
                              max_iter_harmony=20,
                              nclust=300)  # Number of compounds

    feats = harmony_out.Z_corr.T
    features = [f'harmony_{i}' for i in range(feats.shape[1])]
    io.merge_parquet(meta, feats, features, output_path)
