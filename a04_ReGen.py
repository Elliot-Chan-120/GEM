import yaml
from pathlib import Path
import pickle as pkl
import pandas as pd
import numpy as np
import re
import random

import sklearn

# file imports - will need more thorough error handling later on
from a02_1_CompositeDNA_Toolkit import CompositeDNA
from a02_2_CompositeProt_Toolkit import CompositeProt
from a03_LookingGlass import LookingGlass
from b00_bio_library import ALL_AA_COMBINATIONS
from b01_utility import custom_parse



class ReGen:
    def __init__(self, pathogenic_gene_file, ml_model, outfile_name):
        self.root = Path(__file__).parent.resolve()
        with open(self.root / 'config.yaml', 'r') as outfile:
            self.cfg = yaml.safe_load(outfile)

        self.target_gene = ""


        self.root = Path(__file__).parent.resolve()
        self.input_folder = Path('ReGen_input')
        self.input_folder.mkdir(exist_ok=True)
        self.input_name = pathogenic_gene_file
        self.input_filepath = self.input_folder / f"{pathogenic_gene_file}.fasta"

        self.output_root = Path('ReGen_output')
        self.output_root.mkdir(exist_ok=True)
        self.output_name = outfile_name
        self.output_folder = self.output_root / f"{self.output_name}_repair_predictions"
        self.output_folder.mkdir(exist_ok=True)


        self.n_iterations = self.cfg['n_iterations']
        self.n_copies = self.cfg['n_copies']
        self.retain_threshold = self.cfg['retain_threshold']
        self.random_choices = self.cfg['random_choices']
        self.benign_threshold = self.cfg['benign_goal']
        self.error_threshold = self.cfg['decrease_threshold']
        self.scale_factor = self.cfg['scale_factor']

        self.model_name = ml_model
        model_path = Path(self.cfg['model_folder']) / ml_model / f"{ml_model}.pkl"
        with open(model_path, 'rb') as model:
            self.model = pkl.load(model)

        # Load analysis modules
        self.DNA_module = CompositeDNA()
        self.Prot_module = CompositeProt()

        # Load possible modifications
        self.nt_database = ALL_AA_COMBINATIONS

    def repair(self):
        """
        Attempts to transform a pathogenic class gene to turn it into a benign one
        :return:
        """
        # ==[[INITIALIZE STARTING GENE DATA]]==
        # Parse input FASTA file
        fasta_dict = custom_parse(self.input_filepath)
        self.target_gene = fasta_dict[0]['Name']

        initial_data = pd.DataFrame(fasta_dict).drop(['Name'], axis=1)
        initial_fp = self.mutation_fp(initial_data)
        initial_score = self.benign_score(initial_fp)

        # initialize starting arrays - tested - ok everything seems good, finally...
        current_genes = pd.concat([initial_data for _ in range(self.n_copies)], axis=0)
        current_scores = pd.Series([initial_score for _ in range(self.n_copies)])
        retain_counter = np.zeros(self.n_copies, dtype=int)

        max_benign_batch_genes = []
        threshold_genes = []

        seen_sequences = set()

        # ====[[ REPAIR LOOP ]]====
        # for n_iterations, mutate all previous genes and put them into new gene
        for _ in range(self.n_iterations):
            new_genes = []
            new_scores = []

            # scan the previous gene's information
            for idx in range(self.n_copies):
                gene = current_genes.iloc[idx].copy()
                score = current_scores.iloc[idx].item() # access raw scalar - previously was a series, pd is so hard

                if retain_counter[idx] >= self.retain_threshold:
                    # if retain threshold is met or passed, return a stochastic mutation
                    # + allow room for some decrease in performance
                    print('random_mutation')
                    new_gene, new_score = self.stoch_mutation(gene)
                    new_score = float(new_score)  # ensure scalar
                    if new_score <= score + self.error_threshold * (self.scale_factor * retain_counter[idx]):
                        retain_counter[idx] += 1   # if there's no improvement, then keep the old data
                        new_genes.append(gene)
                        new_scores.append(score)
                        print(f"index: {idx}")
                        print(f"score: {score}")
                    else:
                        retain_counter[idx] = 0
                        new_genes.append(new_gene)
                        new_scores.append(new_score)
                        print(f"index: {idx}")
                        print(f"score: {new_score}")
                else:
                    # Guided mutation if retain counter < threshold
                    # generate new variants and their scores w/ the best mutations for each idx
                    # check if new scores are an improvement, if there is no possible improvement, increment retain counter
                    new_gene, new_score = self.guided_mutation(gene)
                    new_score = float(new_score)  # ensure scalar
                    if new_score <= score:
                        retain_counter[idx] += 1
                        new_genes.append(gene)
                        new_scores.append(score)
                        print("No improvement, keeping score")
                        print(f"index: {idx}")
                        print(f"score: {new_score}")
                    else:
                        retain_counter[idx] = 0
                        new_genes.append(new_gene)
                        new_scores.append(new_score)
                        print(f"index: {idx}")
                        print(f"score: {new_score}")

                # if any genes have passed the benign threshold, add them to the list
                if new_score > self.benign_threshold:
                    string = new_gene['AlternateAlleleVCF']
                    if string not in seen_sequences:
                        threshold_genes.append((new_gene, new_score))
                        seen_sequences.add(string)


            # move new genes into current for comparison in next iteration
            current_genes = pd.DataFrame(new_genes).reset_index(drop=True)
            current_scores = pd.Series(new_scores).reset_index(drop=True)

            # take most benign gene from this iteration and save it
            max_idx = current_scores.idxmax()
            max_benign_batch_genes.append((current_genes.iloc[max_idx], current_scores[max_idx]))


        last_variants = current_genes.copy()
        last_variants['Scores'] = current_scores

        print(last_variants.to_string())

        # Exit analysis modules
        self.DNA_module.terminate_pool()
        self.Prot_module.terminate_pool()

        print("Saving data to textfile...")
        self.save_data(initial_score.item(), initial_data.iloc[0], last_variants, max_benign_batch_genes, threshold_genes)


    # core mutation functions - utilize guided mutations until a plateau is hit
    # then use random mutations to start a new 'mutation path'
    def guided_mutation(self, gene_var):
        """
        Pass this one gene variant, and it will figure out which of the possible nucleotide additions or section deletions is optimal
        Optimized data handling, it was pretty sloppy before
        :param gene_var:
        :return:
        """
        candidate_genes = []
        variant = gene_var['AlternateAlleleVCF']

        for nt in self.nt_database:    # generate all possible addition mutations
            new_gene = gene_var.copy()
            new_gene['AlternateAlleleVCF'] = variant + nt
            candidate_genes.append(new_gene)

        if len(variant) > 3:  # if length is sufficient, generate all possible deletion mutations of section length 1-3
            for pos in range(4, len(variant)-2):
                for section in range(3):
                    new_gene = gene_var.copy()
                    new_gene['AlternateAlleleVCF'] = variant[:pos] + variant[pos + section:]
                    candidate_genes.append(new_gene)

        candidate_genes = pd.DataFrame(candidate_genes, columns=gene_var.index)
        candidate_fps = self.mutation_fp(candidate_genes)
        scores = self.benign_score(candidate_fps)

        candidate_genes['Scores'] = scores
        best_idx = scores.idxmax()
        return candidate_genes.iloc[best_idx][gene_var.index], float(scores[best_idx])

    def stoch_mutation(self, gene_var):
        # randomly remove 1,2,3 nucleotides at 10 random positions of alternate allele vcf
        # - probably make this configurable or adaptable to the size of the variant
        # get random nucleotide size

        variant = gene_var['AlternateAlleleVCF']

        if len(variant) <= 3:
            candidate_genes = []
            # if it's too small, it has to undergo a random addition mutation
            for idx in range(self.random_choices):
                new_gene = gene_var.copy()
                new_gene['AlternateAlleleVCF'] = variant + random.choice(self.nt_database)
                candidate_genes.append(new_gene)
        else:
            candidate_genes = []
            for idx in range(self.random_choices):
                new_gene = gene_var.copy()
                nt_size = random.randint(1, min(3, len(variant) - 1))
                pos = random.randint(0, len(variant) - nt_size)
                # replace and score
                new_gene['AlternateAlleleVCF'] = variant[:pos] + variant[pos + nt_size:]
                candidate_genes.append(new_gene)

        candidate_genes = pd.DataFrame(candidate_genes, columns=gene_var.index)
        candidate_fp = self.mutation_fp(candidate_genes)

        scores = self.benign_score(candidate_fp)
        candidate_genes['Scores'] = scores
        best_idx = scores.idxmax()

        return candidate_genes.iloc[best_idx][gene_var.index], float(scores[best_idx])



    # helper functions
    def mutation_fp(self, variant_dataframe):
        df = variant_dataframe.copy()
        dna_df = self.DNA_module.gen_DNAfp_dataframe(df)
        prot_df = self.Prot_module.gen_AAfp_dataframe(df)
        return pd.concat([dna_df, prot_df], axis=1)

    def benign_score(self, muta_fingerprint):
        predictions = self.model.predict_proba(muta_fingerprint.drop(['ClinicalSignificance'], axis=1))[:, 0]
        return pd.Series(predictions)


    def save_data(self, start_score, initial_allele, final_df, max_benign_vars, threshold_vars):
        export_data = {
            'starting_score': start_score,
            'original_allele': initial_allele['ReferenceAlleleVCF'],
            'alternate_allele': initial_allele['AlternateAlleleVCF'],
            'Flank_1': initial_allele['Flank_1'],
            'Flank_2': initial_allele['Flank_2'],
            'final_variants': [],
            'max_benign_variants': [],
            'benign_threshold_variants': []
        }

        # Extract final variants + scores
        for idx in range(len(final_df)):
            variant_info = {
                'Allele': final_df.iloc[idx]['AlternateAlleleVCF'],
                'Benign_pct': final_df.iloc[idx]['Scores'] * 100,
            }
            export_data['final_variants'].append(variant_info)

        # Extract max iteration batch variants
        for gene, score in max_benign_vars:
            variant_info = {
                'Allele': gene['AlternateAlleleVCF'],
                'Benign_pct': score * 100,
            }
            export_data['max_benign_variants'].append(variant_info)

        # Extract max iteration batch variants
        for gene, score in threshold_vars:
            variant_info = {
                'Allele': gene['AlternateAlleleVCF'],
                'Benign_pct': score * 100,
            }
            export_data['benign_threshold_variants'].append(variant_info)


        self.export_txt_fasta(export_data)

    def export_txt_fasta(self, export_data):
        """
        outputs export data as human-readable text and fasta format
        :param export_data:
        :return:
        """
        self.readable_text(export_data)

        self.fasta_file(export_data)

    def readable_text(self, export_data):
        """
        Output all information as a txt file
        :param export_data:
        :return:
        """

        starting_pct = export_data['starting_score'] * 100

        content = ""

        content += "=" * 80 + "\n"
        content += f"ReGen Analysis Results: {self.model_name} | {self.input_name} | {self.target_gene}\n"
        content += "=" * 80 + "\n\n"

        # Original Variant and Benign probability
        content += "ORIGINAL VARIANT STATS: \n"
        content += f"Ref Sequence: {export_data['original_allele']}\n"
        content += f"Alt Sequence: {export_data['alternate_allele']}\n"
        content += f"Benign % chance: {starting_pct:.6f}\n\n"

        # Analysis Summary
        content += "ANALYSIS SUMMARY:\n"
        content += f"|- Starting Score: {export_data['starting_score']:.6f}\n"
        content += f"|- Original Length: {len(export_data['original_allele']):,} bp\n"
        content += f"|- Final Variants: {len(export_data['final_variants'])}\n"
        content += f"|- Benign Threshold Variants: {len(export_data['benign_threshold_variants'])}\n"
        content += f"|- ReGen config: {self.n_iterations} iterations, {self.n_copies} copies\n\n"


        # Log down each info of the best variant from each iteration
        if export_data['max_benign_variants']:
            content += "MAX BENIGN VARIANTS PER ITERATION:\n"
            content += "-" * 50 + "\n"
            for variant in export_data['max_benign_variants']:
                content += f"Score: {variant['Benign_pct']} | "
                content += f"Length: {len(variant['Allele']):,} bp\n"
                content += f"Benign % increase: {variant['Benign_pct'] - starting_pct}\n"

                sequence = variant['Allele']
                content += f"   Sequence:\n"
                for i in range(0, len(sequence), 80):
                    content += f"    {sequence[i:i+80]}\n"
                content += "\n"


        # Log down info on threshold-exceeding variants
        if export_data['benign_threshold_variants']:
            content += "BENIGN THRESHOLD VARIANTS:\n"
            content += "-" * 50 + "\n"
            for variant in export_data['benign_threshold_variants']:
                content += f"Score: {variant['Benign_pct']} | "
                content += f"Length: {len(variant['Allele']):,} bp\n"
                content += f"Benign % increase: {variant['Benign_pct'] - starting_pct}\n"

                sequence = variant['Allele']
                content += f"   Sequence:\n"
                for i in range(0, len(sequence), 80):
                    content += f"    {sequence[i:i + 80]}\n"
                content += "\n"

        content += "FINAL VARIANTS:\n"
        content += "-" * 50 + "\n"
        for variant in export_data['final_variants']:
            content += f"Score: {variant['Benign_pct']} | "
            content += f"Length: {len(variant['Allele']):,} bp\n"
            content += f"Benign % increase: {variant['Benign_pct'] - starting_pct}\n"

            sequence = variant['Allele']
            content += f"   Sequence: \n"
            for i in range(0, len(sequence), 80):
                content += f"    {sequence[i:i + 80]}\n"
            content += "\n"

        outfile = self.output_folder / f"{self.input_name}_{self.model_name}_results.txt"
        outfile.write_text(content)

    def fasta_file(self, export_data):
        """
        Will complete this at a later date
        :param export_data:
        :return:
        """
        pass



