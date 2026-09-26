"""Read only the final dose-replay tables and frozen lexicon."""
import json
import pandas as pd
from ..paths import ROOT, OUTPUT
ALPHAS=[-4,-2,-1,0,1,2,4]
def fresh(name):return pd.read_csv(OUTPUT/'tables'/f'{name}.csv')
def lexicon():return json.loads((ROOT/'configs/lexicon.json').read_text())
