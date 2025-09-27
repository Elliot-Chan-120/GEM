import yaml
from pathlib import Path
import pickle as pkl
import pandas as pd
import numpy as np
import re

import sklearn

from a02_1_CompositeDNA_Toolkit import CompositeDNA
from a02_2_CompositeProt_Toolkit import CompositeProt
from a02_3_DNAProfiler_Toolkit import *
from a02_4_ProtProfiler_Toolkit import *
from b01_utility import custom_parse


class LookingGlass:
    def __init__(self, gene_file, ml_model_name):
        self.root = Path(__file__).parent.resolve()
        with open(self.root / 'config.yaml', 'r') as outfile:
            self.cfg = yaml.safe_load(outfile)


        self.gene_filename = gene_file

        # navigation
        self.predict_folder = Path(self.cfg['screen_predictions'])
        self.predict_folder.mkdir(exist_ok=True)
        self.input_file = Path(self.cfg['gene_databank']) / self.gene_filename

        # model loading
        model_path = Path(self.cfg['model_folder']) / ml_model_name / f"{ml_model_name}.pkl"
        with open(model_path, 'rb') as model:
            self.model = pkl.load(model)


    def DNA_fingerprint(self):
        dataframe = custom_parse(self.input_file)
        dataframe = pd.DataFrame(dataframe)

        # ==[DNA data]==
        with CompositeDNA() as dna_module:
            dna_df = dna_module.gen_DNAfp_dataframe(dataframe)

        # ==[Protein data]==
        with CompositeProt() as prot_module:
            prot_df = prot_module.gen_AAfp_dataframe(dataframe)

        with PosWeightProfiler() as pwm_module:
            pwm_df = pwm_module.gen_PWM_dataframe(dataframe)

        # [[Save DataFrame]]
        # ensure alignment - in case something goes wrong with one of them
        dna_df.index = dataframe.index
        prot_df.index = dataframe.index
        pwm_df.index = dataframe.index

        variant_df = pd.concat([dna_df, prot_df, pwm_df], axis=1)

        return variant_df


    def run_model(self):
        mutation_fingerprint = self.DNA_fingerprint()

        mutation_fingerprint = mutation_fingerprint.loc[:, ~mutation_fingerprint.columns.duplicated()]
        names = mutation_fingerprint.Name
        mutation_fingerprint = mutation_fingerprint.drop(['ClinicalSignificance', 'Name'], axis=1)

        results = self.model.predict_proba(mutation_fingerprint)
        predictions = (results[:, 1] >= self.cfg['optimal_threshold']).astype(int)
        result_df = pd.DataFrame({
            'Predicted_Class': predictions,
            'Prob_Benign': results[:, 0],
            'Prob_Pathogenic': results[:, 1]
        })

        prediction_df = pd.concat([names, result_df], axis=1)

        return prediction_df


    def predict_file(self, outfile_name= str):
        screen_results = self.run_model()
        outpath = self.predict_folder / f'{outfile_name}.csv'
        screen_results.to_csv(outpath, index=False)

