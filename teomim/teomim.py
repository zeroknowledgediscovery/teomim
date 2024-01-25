import pandas as pd
import numpy as np
from quasinet.qnet import load_qnet
from quasinet.qsampling import qsample
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse
from scipy.spatial.distance import cosine
import pkg_resources
import glob
import random

def bhattacharyya_coefficient(pmf1, pmf2):
    return np.sum(np.sqrt((np.array(pmf1) * np.array(pmf2)).astype(float)))


# Global variables
global_model = None
global_steps = None
global_alpha = None

def select_key_by_probability(prob_dict):
    """
    Select a key from a dictionary where the keys are the items to be selected
    and the values are the probabilities of each key.
    """
    # Normalize the probabilities to ensure they sum up to 1
    total = sum(prob_dict.values())
    normalized_probs = {k: v / total for k, v in prob_dict.items()}

    # Randomly select a key based on the probabilities
    return random.choices(list(normalized_probs.keys()), weights=normalized_probs.values(), k=1)[0]




def init_globals(model, steps, alpha):
    global global_model, global_steps, global_alpha
    global_model = model
    global_steps = steps
    global_alpha = alpha

def select_random_row(arr):
    if arr.shape[0] == 1:
        # Only one row
        return arr[0]
    else:
        # Multiple rows, select one randomly
        random_index = np.random.randint(arr.shape[0])
        return arr[random_index]


def parallel_qsample(seed):
    
    return qsample(seed, global_model,
                   steps=global_steps,
                   random_seed=True,
                   alpha=select_key_by_probability(global_alpha))

def generate(modelpath, gz=True, alpha=1.3, outfile=None,
             steps=200000, numworkers=11, num_patients=1000,
             seed=None):
    model = load_qnet(modelpath, gz=gz)
    featurenames = np.array(model.feature_names)
    if seed is None:
        seed = np.array([''] * len(featurenames)).astype('U100')
        seeds = [seed for _ in range(num_patients)]
        seed_used='EMPTY STR'
    else:
        seeds = [select_random_row(seed) for _ in range(num_patients)]
        seed_used = 'DATAFRAME'
        

    # Initialize global variables
    init_globals(model, steps, alpha)

    with ProcessPoolExecutor(max_workers=numworkers,
                             initializer=init_globals,
                             initargs=(model, steps, alpha)) as executor:
        results = list(tqdm(executor.map(parallel_qsample, seeds),
                            total=num_patients))

    Sf = pd.DataFrame(results, columns=featurenames)
    if outfile:
        Sf.to_csv(outfile)
    return Sf,seed_used


def evaluate__(df, code_prefixes, suffix=None, age_prefix=''):
    if not isinstance(code_prefixes, (np.ndarray, list)):
        code_prefixes = [code_prefixes]

    valid_rows = np.array([True] * df.index.size)

    if suffix is not None and not isinstance(suffix, (np.ndarray, list)):
        suffix = [suffix]

    for code_prefix in code_prefixes:
        af = df[[col for col in df.columns if col.startswith(code_prefix+'_'+age_prefix)]]
        af=af.replace('.','').replace('',np.nan)

        if suffix:
            for s in suffix:
                af = af.replace(s, np.nan)
        # Determine if any non-NaN values exist in the row after handling suffixes
        current_valid = af.notna().sum(axis=1).astype(bool)
        # Perform an AND operation between the currently valid rows and the overall valid_rows
        valid_rows &= current_valid

    num_valid_rows = valid_rows.sum()

    return num_valid_rows / df.index.size

class teomim:
    def __init__(self, modelpath=None, gz=True, alpha=1.3,
                 outfile=None, steps=200000,
                 numworkers=11,
                 num_patients=1000,seed=None):
        self.modelpath = modelpath
        self.gz = gz
        self.alpha = alpha
        self.outfile = outfile
        self.steps = steps
        self.numworkers = numworkers
        self.num_patients = num_patients
        self.seed = seed
        self.patients = None
        self.seed_used = None
        self.EVAL_PREFIXES={'I10':.7,'I25':.4,'I50':.25,'E11':.46,
                            'E66':.3,'I63':.4,'G20':.15,'F32':.5,
                            'F41':.4,'M81':.25,'J44':.55,'J84':0.005}

        self.asset_path = pkg_resources.resource_filename('teomim', 'assets/')

    def set_modelpath(self,specifier,path=None,gz=None):
        if gz:
            self.gz = gz
        if not path:
            self.modelpath = glob.glob(self.asset_path+'/*'+specifier+'*')[0]
        else:
            self.modelpath = specifier
        return 
        
    def load(self,patientdata):
        self.patients = pd.read_csv(patientdata)
        
    def generate(self):
        self.patients,self.seed_used\
            = generate(modelpath=self.modelpath,
                       gz=self.gz, alpha=self.alpha,
                       outfile=self.outfile,
                       steps=self.steps,
                       seed=self.seed,
                       numworkers=self.numworkers,
                       num_patients=self.num_patients)

    def set_model(self): 
        self.model = load_qnet(self.modelpath, gz=self.gz)
        self.featurenames = np.array(self.model.feature_names)


    def evaluate(self,EVAL=None):

        if EVAL is None:
            EVAL = self.EVAL_PREFIXES
        elif not isinstance(EVAL, dict) or not all(isinstance(key,
                                                              str)
                                                   and isinstance(value,
                                                                  float)
                                                   for key, value in EVAL.items()):
            raise ValueError("EVAL must be a dictionary\
            with keys as strings and values as floats.")
        
            
        self.evaldf = pd.DataFrame([evaluate__(self.patients,x)
                                    for x in EVAL.keys()],
                                   list(EVAL.keys()),
                                   columns=[
                                       'prevalences']).assign(
                                           prevalence_expected
                                =(np.array(EVAL.values())))

        return self.evaldf.copy()

        
    def quality(self,df=None):

        if not df:
            df=self.evaldf
            
        if df.shape[1] != 2:
            raise ValueError("DataFrame should have exactly\
            two columns representing two PMFs.")

        # Extracting PMFs from DataFrame columns
        pmf1 = df.iloc[:, 0]
        pmf2 = df.iloc[:, 1]

        # Normalize PMFs to ensure they sum to 1
        pmf1 = np.array(pmf1) / np.sum(pmf1)
        pmf2 = np.array(pmf2) / np.sum(pmf2)

        # Calculate Bhattacharyya Coefficient
        b_coeff = bhattacharyya_coefficient(pmf1, pmf2)*100

        return np.round(b_coeff,2)

        
