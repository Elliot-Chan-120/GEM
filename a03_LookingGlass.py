import yaml
from pathlib import Path
import pickle as pkl
import pandas as pd
import numpy as np
import re

import sklearn

from a02_1_CompositeDNA_Toolkit import CompositeDNA
from a02_2_CompositeProt_Toolkit import CompositeProt
from a02_3_DNAMatrix_Toolkit import *
from a02_4_ProtMatrix_Toolkit import *
from b01_utility import custom_parse
from DataSift import *


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
        self.model_name = ml_model_name
        model_path = Path(self.cfg['model_folder']) / ml_model_name / f"{ml_model_name}.pkl"
        with open(model_path, 'rb') as model:
            self.model = pkl.load(model)

        self.Sift = None


    def DNA_fingerprint(self):
        control = SiftControl()
        control.LoadConfig(self.model_name)
        self.Sift = control.LoadSift()

        dataframe = custom_parse(self.input_file)
        dataframe = pd.DataFrame(dataframe)

        # columns: ReferenceAlleleVCF, AlternateAlleleVCF, Flank_1, Flank_2, Name, Chromosome, ClinicalSignificance

        # ==[Protein extraction + data]==
        with CompositeProt() as prot_module:
            composite_df = prot_module.gen_AAseqs(dataframe)
            prot_df = prot_module.gen_AAfp_dataframe(composite_df)

        # ==[DNA data]==
        with CompositeDNA() as dna_module:
            dna_df = dna_module.gen_DNAfp_dataframe(composite_df)
            hmm_df = dna_module.gen_HMM_dataframe(composite_df)

        with DNAMatrix() as dnapwm_module:
            dnapwm_df = dnapwm_module.gen_DNAPWM_dataframe(composite_df)

        with ProtMatrix() as protpwm_module:
            aapwm_df = protpwm_module.gen_AAPWM_dataframe(composite_df)

        # [[Save DataFrame]]
        # ensure alignment - in case something goes wrong with one of them
        dna_df.index = dataframe.index
        prot_df.index = dataframe.index
        dnapwm_df.index = dataframe.index
        aapwm_df.index = dataframe.index
        hmm_df.index = dataframe.index

        variant_df = pd.concat([dna_df, prot_df, dnapwm_df, aapwm_df, hmm_df], axis=1)

        useless_columns = ['ref_protein_list', 'alt_protein_list',
                           'non_ambiguous_ref', 'non_ambiguous_alt',
                           'ref_protein_length', 'alt_protein_length']

        variant_df = variant_df.drop(useless_columns, axis=1)

        return variant_df


    def run_model(self):
        mutation_fingerprint = self.DNA_fingerprint()

        mutation_fingerprint = mutation_fingerprint.loc[:, ~mutation_fingerprint.columns.duplicated()]
        names = mutation_fingerprint.Name
        mutation_fingerprint = mutation_fingerprint.drop(['ClinicalSignificance', 'Name'], axis=1)

        # apply data sift
        mutation_fingerprint = mutation_fingerprint[self.Sift]

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

