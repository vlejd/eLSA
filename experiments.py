""" Script that perform experiments for all parameters. 

It  cen be run directly by python3 expeirments.py
But ake sure, you set environment variables: 
    'SHARDING'= number of shard (1), 
    'OFFSET'=which part of hyperparameters should be checked (0), 
    'THREADS'=number ofthreds to use (1)
    'INSCRIPT'= set to one (will disable ploting)

You can also import the function run_test. 
"""
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import os
import pickle
import tqdm


import datasets
import classify
from elsa import ELSA


def get_environ(key, default=None):
    if key in os.environ:
        if os.environ[key] is None:
            return default
        return os.environ[key]
    else: return default


INSCRIPT = bool(int(get_environ('INSCRIPT', "0")))

if INSCRIPT:
    from tqdm import trange as tnrange
else:
    from tqdm import tnrange

def gradient_w(model, dataset, alpha=0.01, epochs=150, w_steps=1):
    train_scores = []
    valid_scores = []
    test_scores = []
    model.fit(dataset.train_samples(), dataset.train_labels())
    if epochs is None:
        t = tnrange(100000)
    else:
        t = tnrange(epochs)
    for e in t:
        for wstep in tnrange(w_steps):
            w = model.get_matrix_w()
            w -= alpha * model.dw(dataset.train_samples(), dataset.train_labels())
            model.save_matrix_w(w)
        model.fit(dataset.train_samples(), dataset.train_labels())

        train_score = model.score(dataset.train_samples(), dataset.train_labels())
        valid_score = model.score(dataset.valid_samples(), dataset.valid_labels())
        test_score = model.score(dataset.test_samples(), dataset.test_labels())
        train_scores.append(train_score)
        valid_scores.append(valid_score)
        test_scores.append(test_score)
        t.set_postfix(train_score=train_score, valid_score=valid_score, test_score=test_score)
        
        if epochs is None and e > 30:
            end_mean = np.mean(valid_scores[-10:])
            previos_mean = np.mean(valid_scores[-20:-10])
            t.set_postfix(train_score=train_score, valid_score=valid_score, test_score=test_score, previos=previos_mean, end=end_mean)
            if end_mean <= previos_mean:
                break
    if not INSCRIPT:
        plt.plot(train_scores)
        plt.plot(valid_scores)
        plt.plot(test_scores)
        plt.legend(['train', 'valid', 'test'])
        plt.show()
    return train_scores, valid_scores, test_scores


def train_elsa(
    model, dataset, gradient_iters=300, dims=300, alpha=0.01, tag=None, results=None, dump=None, with_models=False, folds=1, w_steps=1):
    
    for i in dataset.reshufle(None, folds):
        model.internal_w=None
        train_ps, valid_ps, test_ps = gradient_w(model, dataset, alpha, gradient_iters, w_steps)
        
        train_p = np.mean(train_ps[-10:])
        valid_p = np.mean(valid_ps[-10:])
        test_p = np.mean(test_ps[-10:])
        if results is not None:
            results[dataset.name()][('batch', tag, alpha, dims, 'train', i)] = train_p
            results[dataset.name()][('batch', tag, alpha, dims, 'valid', i)] = valid_p
            results[dataset.name()][('batch', tag, alpha, dims, 'test', i)] = test_p

        if dump is not None:
            dump[dataset.name()][('batch', tag, alpha, i)] = {
                'train': list(train_ps),
                'valid': list(valid_ps),
                'test': list(test_ps),
                'w': model.internal_w,
            }
            if with_models:
                dump[dataset.name()][('batch', tag, alpha, i)]['model']= model


        print(dataset.name())
        print("Train precision", train_p)
        print("Valid precision", valid_p)
        print("Test precision", test_p)



def args2tag(args):
    tag = ('{}'+('_{}'*(len(args)-1))).format(args[0].name(), *args[1:])
    return tag
    

result_file_pattern = 'dumps/elsa_results_{}.pickle'
dump_file_pattern = 'dumps/elsa_dump_{}.pickle'
def run_test(args):
    start = True
    start_on = ' '
    dump_results = True

    results = defaultdict(dict)
    dump = defaultdict(dict)

    dataset, scheme, alpha, dims = args
    tag = args2tag(args)
    results_file = result_file_pattern.format(tag)
    dumps_file = dump_file_pattern.format(tag)
    if os.path.isfile(results_file):
        print('skipping', results_file)
        return
    if not start:
        start = (tag == start_on)
        return
    print(dataset.name(), scheme, alpha, dims, tag)
    model = ELSA(classify.SkClassifier(), use_svd=True, weights=scheme, svd_dim=dims)
    train_elsa(
        model, dataset, alpha=alpha, dims=dims, tag=scheme, 
        gradient_iters=None, results=results, dump=dump, with_models=False, folds=3)
    print(list(model.internal_w.items())[:10])
    if dump_results:
        pickle.dump(results, open(results_file, 'bw'))
        pickle.dump(dump, open(dumps_file, 'bw'))
                

result_multiw_file_pattern = 'dumps_multiw/elsa_results_{}.pickle'
dump_multiw_file_pattern = 'dumps_multiw/elsa_dump_{}.pickle'
def run_test_multiw(args):
    start = True
    start_on = ' '
    dump_results = True

    results = defaultdict(dict)
    dump = defaultdict(dict)

    dataset, scheme, alpha, dims, w_steps = args
    tag = args2tag(args)
    results_file = result_multiw_file_pattern.format(tag)
    dumps_file = dump_multiw_file_pattern.format(tag)
    if os.path.isfile(results_file):
        print('skipping', results_file)
        return
    if not start:
        start = (tag == start_on)
        return
    print(dataset.name(), scheme, alpha, dims, tag)
    model = ELSA(classify.SkClassifier(), use_svd=True, weights=scheme, svd_dim=dims)
    train_elsa(
        model, dataset, alpha=alpha, dims=dims, tag=scheme, 
        gradient_iters=None, results=results, dump=dump, with_models=False, folds=3, w_steps=w_steps)
    print(list(model.internal_w.items())[:10])
    if dump_results:
        pickle.dump(results, open(results_file, 'bw'))
        pickle.dump(dump, open(dumps_file, 'bw'))
                


def main(sharding = 10, offset = 0, threads=3):
    done = None
    try:
        done = json.load(open('done.json'))
    except:
        done = []
    with open('log.log','w') as log:
        print('start',file=log)
    args = []
    for scheme in ELSA.SCHEMES:
        for alpha in [0.1, 0.01, 0.001]:
            for dims in [200, 300, 400]:
                for dataset in datasets.ALL_DATASETS+ datasets.TREC_DATASETS:
                    arg = (dataset, scheme, alpha, dims)
                    if result_file_pattern.format(args2tag(arg)) not in done:
                        args.append(arg)
                        
    todo = args[offset::sharding]
    print(len(todo))
    if threads==1:
        for i,tod in enumerate(todo):
            with open('log.log','w') as log:
                print(i,file=log)
            try:
                run_test(tod)
            except:
                with open('log.log','w') as log:
                    print(i, "error",file=log)
                raise
    else:
        with Pool(threads) as p:
            print(p.map(run_test, todo))
    
    with open('log.log','w') as log:
        print('done',file=log)

    
if __name__=="__main__":
    print(INSCRIPT)
    sharding, offset, threads = int(get_environ('SHARDING',1)), int(get_environ('OFFSET',0)), int(get_environ('THREADS',1))
    print(sharding, offset, threads)
    main(sharding, offset, threads)

