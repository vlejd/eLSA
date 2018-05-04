"""Collection of small utility functions used in jupyter notebooks for nicer visualuzation and latex table generation """

import datasets
import seaborn as sns

def subtract_baseline(df):
    for d in datasets.ALL_DATASETS + datasets.TREC_DATASETS:
        if d.name() in df.columns:
            df[d.name()] -= d.bias()

def tabular(text, caption='', label=''):
    top = r"""
\begin{table}[H]
\begin{center}

"""
    bot = r"""
\caption["""+caption+r"""]{"""+caption+r"""}
\label{"""+label+"""}
\end{center}
\end{table}

"""
    return top + text + bot


STANDARD_REPLACE = (
    ('  ', ' '),
    ('Dataset',''),
    ('TREC-',''),
)

def multireplace(txt, lis=STANDARD_REPLACE):
    for q,w in lis:
        txt = txt.replace(q,w)
    return txt

cm = sns.light_palette("green", as_cmap=True)
def color_positives(val):
    color = 'red' if val > 0 else 'black'
    return 'color: %s' % color


TREC = ['TRECDataset-ABBR', 'TRECDataset-DESC', 'TRECDataset-ENTY', 'TRECDataset-HUM', 'TRECDataset-LOC', 'TRECDataset-NUM']
NOTREC = ['CRDataset', 'MPQADataset', 'MRDataset', 'SUBJDataset']

def bold_positives(x):
    if x>0:
        return "\\textbf{{{}}}".format(x)
    else:
        return "{}".format(x)
