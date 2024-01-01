import pandas as pd
from nigsp import io

E = pd.read_csv('no_tau_0/files/sub-1_ts-innov.tsv.gz', sep='\t') 
print(E.shape)
E= io.load_txt('no_tau_0/files/sub-1_ts-innov.tsv.gz')
print(E.shape)