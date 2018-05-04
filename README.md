# Diploma thesis
Code related to my diploma thesis: **Improving LSA word weights for document classification**

We run experiments on google cloud.

# Instalation guide
These sommands are designed to be run in home directory (you may need to change some paths).

```
yes | sudo apt install htop ;
yes | sudo apt install unzip ;
yes | sudo apt install git ;
git clone https://github.com/facebookresearch/SentEval.git ;
yes | sudo apt install python3-pip  ;
yes | sudo apt-get install python3-tk ;
export LC_ALL=C ;
cd ;
cd SentEval/data/downstream ;
./get_transfer_data.bash ;

git clone https://github.com/vlejd/eLSA.git ;
cd eLSA/ ;
yes | pip3 install -r requirements.txt ;
cd ;
cd eLSA/ ;
mkdir dumps ;
```


Beacause we need to run a lot of experiments, our code is designed to be run on google cloud compute engine.
Experiment reads environsment variables that determine, which part of the experiment should be computed on specific machine.

To run experiments, do 
```
screen -Sdm dip; 
screen -S dip -X stuff '
    cd eLSA; 
    yes | git pull; 
    export SENTEVAL_DATA_BASE=/home/vlejd/SentEval/data/downstream/ ;
    export SHARDING=1; 
    export OFFSET=1; 
    export THREADS=1 ; 
    export INSCRIPT=1 ; 
    python3 experiments.py; 
    sleep 10;"$(echo -ne "\015");"
';
```


# Directories


- `clean_dumps`: cleaned data about experiments

These files are not present in the git, but only after a request. However they should be reproducible
- `dumps`: data about experiments
- `dumps_multiw`: data about experiments with multiple 2 steps
- `word2vec`: word2vec experiments

Implementation:

- `classify.py`: wrappers around classifiers
- `datasets.py`: wrapper around datasets
- `elsa.py`: implementation of eLSA
- `experiments.py`: experiments with eLSA (can be used to run eLSA experiments)
- `testing.py`: lsa baseline testing
- `utils.py`: utilities for jupyter notebooks
- `term_weights.py`: implementation of term weights (not written by us!)

Jupyter notebooks with results and some experiments (name should be descriptive enough):
- `gen_dataset_stats.ipynb`:
- `experiments_baselines.ipynb`:
- `experiments_batch_clean.ipynb`: another way how to run experiments with eLSA
- `experiments_batch_multiw.ipynb`: eLSA with multiple gradient steps
- `experiments_word_vectors_pretrained.ipynb`:
- `experiments_word_vectors_trained.ipynb`:
- `results_baselines.ipynb`:
- `results_batch.ipynb`: eLSA results
- `results_batch_multiw.ipynb`: eLSA with multiple gradient steps
- `wanal_batch.ipynb`: weight analysis
