# Navigation help
import yaml
from pathlib import Path

# math
import numpy as np
import pandas as pd

# helpers
from collections import Counter
import re
from tqdm import tqdm
import random

# multiprocessing
import multiprocessing
from multiprocessing import Pool

from b00_bio_library import IUPAC_CODES

global_config = {}

def config_init(config, iupac_codes):
    global global_config, IUPAC_CODES
    global_config = config
    IUPAC_CODES = iupac_codes


# to do
# PWM threshold calculator
# - scan PWm and bring up theoretical min max
# - get % threshold (user defined but lets use some default values 0.8-0.9) and calculate its value on that scale
# - return threshold that we then input in the scoring function (threshold parameter unfilled)


class PosWeightProfiler:
    _pool = None

    def __init__(self, core_num=None):
        self.root = Path(__file__).parent.resolve()

        # === load up motif PWMs globally for repeated reference ===
        motif_path = Path('database/pwm_database')
        splice_3_pwm = np.loadtxt(motif_path / '3_splice_pwm.txt')  # human donor splice site (3')
        splice_5_pwm = np.loadtxt(motif_path / '5_splice_pwm.txt')  # human acceptor splice site (5')
        branch_pt_pwm = np.loadtxt(
            motif_path / 'branch_pt_pwm.txt')  # human branch point -> took pwm logo from study and ran it through AI to reconstruct PWM, take this w/ grain of salt

        # simple transcription factors
        ctcf_pwm = np.loadtxt(motif_path / 'CTCF_TF_pwm.txt')  # CTCF transcription factor
        caat_pwm = np.loadtxt(motif_path / 'CAAT_pwm.txt')  # CAAT box TF
        tata_pwm = np.loadtxt(motif_path / 'TATA_pwm.txt')  # TATA box TF


        # create config for downstream multiproc
        self.config = {
            'splice_3_pwm': splice_3_pwm,
            'splice_5_pwm': splice_5_pwm,
            'branch_pt_pwm': branch_pt_pwm,
            'ctcf_pwm': ctcf_pwm,
            'caat_pwm': caat_pwm,
            'tata_pwm': tata_pwm,
        }

        # define number of cores
        self.core_num = core_num if core_num is not None else multiprocessing.cpu_count() - 2

        # initialize pool if not already initialized
        if PosWeightProfiler._pool is None:
            self.initialize_pool()


    def initialize_pool(self):
        """
        Initialize multiprocessing pool w/ provided configuration and data
        """
        core_num = self.core_num

        if PosWeightProfiler._pool is None:
            PosWeightProfiler._pool = multiprocessing.Pool(
                processes = core_num,
                initializer = config_init,
                initargs=(self.config, IUPAC_CODES)
            )

    def terminate_pool(self):
        """
        Terminate multiprocessing pool when no longer needed
        """
        if PosWeightProfiler._pool is not None:
            PosWeightProfiler._pool.close()
            PosWeightProfiler._pool.join()
            PosWeightProfiler._pool = None

    def __enter__(self):
        self.initialize_pool()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate_pool()



    # ====[[PWM MOTIF FINGERPRINT DATAFRAME]]====
    @staticmethod
    def gen_DNApwm_dataframe(dataframe):
        """
        Uses persistent mp pool to generate DNA mutation data fingerprint
        :param dataframe:
        :return:
        """
        fingerprint_rows = [
            (row['Chromosome'], row['ReferenceAlleleVCF'],
             row['AlternateAlleleVCF'], row['Flank_1'], row['Flank_2'],
             row['non_ambiguous_ref'], row['non_ambiguous_alt'])
            for _, row in dataframe.iterrows()
        ]

        fingerprint_rows = list(tqdm(
            # pool.imap method applies function self.PWM_profile_wrapper to each row in fingerprint_rows
            PosWeightProfiler._pool.imap(PosWeightProfiler.PWM_profile_wrapper, fingerprint_rows),
            total=len(fingerprint_rows),
            desc="[Generating DNA PWM motif fingerprints]"
        ))

        fingerprint_df = pd.DataFrame(fingerprint_rows)
        fingerprint_df = pd.concat([dataframe.reset_index(drop=True), fingerprint_df],
                                   axis=1)  # axis = 1 to concatenate column wise (side by side)

        fingerprint_df = fingerprint_df.drop(['Chromosome', 'ClinicalSignificance', 'ReferenceAlleleVCF', 'AlternateAlleleVCF', 'Flank_1', 'Flank_2'],
                                             axis=1)

        return fingerprint_df

    @staticmethod
    def PWM_profile_wrapper(fp_row):
        """
        multiprocessing wrapper, processes a single row
        :param DNA dataframe fp_row:
        :return:
        """
        chromosome, ref_allele, alt_allele, flank_1, flank_2, ref_protein, alt_protein = fp_row

        # Organize sequence dimensions
        # Full-length alleles (context-inclusive)
        ref_full = flank_1 + ref_allele + flank_2
        alt_full = flank_1 + alt_allele + flank_2

        fp = {}
        fp.update(PosWeightProfiler.DNA_pwm_profile(ref_full, alt_full, flank_1, flank_2, ref_allele, alt_allele))

        return fp

    @staticmethod
    def DNA_pwm_profile(full_ref_allele, full_alt_allele, flank1, flank2, ref_vcf, alt_vcf):
        """
        :param full_ref_allele:
        :param full_alt_allele:
        :param flank1:
        :param flank2:
        :param ref_vcf:
        :param alt_vcf:
        :return:
        """
        # ====[PWM MOTIF DISRUPTIONS]====
        # Combines gaussian scoring and pwm navigation to create a motif disruption score for each specific motif
        # For now, let's see how impactful each motif search is going to be for XGBoost

        pwm_dict = {}
        non_ambi_ref = PosWeightProfiler.non_ambi_seq(full_ref_allele)
        non_ambi_alt = PosWeightProfiler.non_ambi_seq(full_alt_allele)

        # splice sites + branch points
        sp3_count, sp3_score = PosWeightProfiler.dna_pwm_stats(global_config['splice_3_pwm'], non_ambi_ref, non_ambi_alt, flank1, flank2, ref_vcf, alt_vcf, 0.95)
        sp5_count, sp5_score = PosWeightProfiler.dna_pwm_stats(global_config['splice_5_pwm'], non_ambi_ref, non_ambi_alt, flank1, flank2, ref_vcf, alt_vcf, 0.95)
        branch_pt_count, branch_pt_score = PosWeightProfiler.dna_pwm_stats(global_config['branch_pt_pwm'], non_ambi_ref, non_ambi_alt, flank1, flank2, ref_vcf, alt_vcf, 0.85)

        # transcription factors
        caat_count, caat_score = PosWeightProfiler.dna_pwm_stats(global_config['caat_pwm'], non_ambi_ref, non_ambi_alt, flank1, flank2, ref_vcf, alt_vcf, 0.90)
        ctcf_count, ctcf_score = PosWeightProfiler.dna_pwm_stats(global_config['ctcf_pwm'], non_ambi_ref, non_ambi_alt, flank1, flank2, ref_vcf, alt_vcf, 0.90)
        tata_count, tata_score = PosWeightProfiler.dna_pwm_stats(global_config['tata_pwm'], non_ambi_ref, non_ambi_alt, flank1, flank2, ref_vcf, alt_vcf, 0.90)

        total_motif_count = sp3_count + sp5_count + branch_pt_count + caat_count + ctcf_count + tata_count
        total_score_shift = sp3_score + sp5_score + branch_pt_score + caat_score + ctcf_score + tata_score

        pwm_dict['sp3_count'] = sp3_count
        pwm_dict['sp3_score'] = sp3_score

        pwm_dict['sp5_count'] = sp5_count
        pwm_dict['sp5_score'] = sp5_score

        pwm_dict['branch_pt_count'] = branch_pt_count
        pwm_dict['branch_pt_score'] = branch_pt_score

        pwm_dict['caat_count'] = caat_count
        pwm_dict['caat_score'] = caat_score

        pwm_dict['ctcf_count'] = ctcf_count
        pwm_dict['ctcf_score'] = ctcf_score

        pwm_dict['tata_count'] = tata_count
        pwm_dict['tata_score'] = tata_score

        # totals
        pwm_dict['total_motif_count'] = total_motif_count
        pwm_dict['total_score_shift'] = total_score_shift

        return pwm_dict


    @staticmethod
    def dna_pwm_stats(pwm, ref_allele, alt_allele, flank1, flank2, ref_vcf, alt_vcf, threshold):
        """
        CALL ON THIS
        motif stat changes due to mutation
        :param pwm:
        :param ref_allele:
        :param alt_allele:
        :param flank1:
        :param flank2:
        :param ref_vcf:
        :param alt_vcf:
        :param threshold: input as percentile
        :return: quantity [0] and score delta [1]
        """
        full_ref = flank1 + ref_allele + flank2
        full_alt = flank1 + alt_allele + flank2

        calc_threshold = PosWeightProfiler.get_threshold(pwm, threshold)

        # get results first
        motif_length = pwm.shape[0]
        ref_motif_idxs, ref_motif_scores = PosWeightProfiler.probability_all_pos(full_ref, motif_length, pwm, calc_threshold)
        alt_motifs_idxs, alt_motif_scores = PosWeightProfiler.probability_all_pos(full_alt, motif_length, pwm, calc_threshold)

        # === quantity delta ===
        motif_quantity_delta = len(alt_motifs_idxs) - len(ref_motif_idxs)

        # === positional-strength composite score
        window_start = len(flank1)
        ref_window_end = len(flank1) + len(ref_vcf)
        alt_window_end = len(flank1) + len(alt_vcf)

        ref_weighted_score = PosWeightProfiler.pos_weight_gaussian(ref_motif_idxs, ref_motif_scores,
                                                 window_start, ref_window_end, motif_length)

        alt_weighted_score = PosWeightProfiler.pos_weight_gaussian(alt_motifs_idxs, alt_motif_scores,
                                                 window_start, alt_window_end, motif_length)

        position_score_delta = alt_weighted_score - ref_weighted_score

        return motif_quantity_delta, position_score_delta

    @staticmethod
    def probability_subseq(subseq, pwm):
        """
        Calculate the probability this sequence will contain the motif model
        :param subseq:
        :param pwm:
        :return: probability float
        """
        alphabet = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        nuc_idx = np.fromiter((alphabet[c] for c in subseq), dtype=np.int8)
        probs = pwm[np.arange(len(subseq)), nuc_idx]
        background_prob = 0.25

        # handle 0s
        probs = np.maximum(probs, 1e-10)

        scores = np.log2(probs / background_prob)

        return scores.sum()

    @staticmethod
    def probability_all_pos(sequence, motif_size, pwm, motif_threshold):
        """
        Performs sliding window and returns list of indices that are likely to contain the motif
        :param full sequence:
        :param motif_size:
        :param pwm:
        :return: list of each probable motif location
        """
        seq_len = len(sequence)
        idxs, scores = [], []
        for i in range(seq_len - motif_size + 1):
            score = PosWeightProfiler.probability_subseq(sequence[i:i + motif_size], pwm)
            if score > motif_threshold:
                idxs.append(i)
                scores.append(score)
        return idxs, scores

    @staticmethod
    # === POSITION_WEIGHTED SCORING MECHANISMS ===
    def pos_weight_gaussian(idxs, scores,
                            vcf_start, vcf_end, motif_length):
        """
        Uses a plateau + gaussian weight decay scoring mechanism
        - full weight 1.0 when motif is within vcf window +/- plateau radius
        - Then weight = exp(-d^2 / (2 * sigma^2)), where d = distance_to_window
        - sigma controls tail decay in base pairs
        :return:
        """
        results = []

        distances = PosWeightProfiler.distance_from_window(idxs, vcf_start, vcf_end)
        weights = PosWeightProfiler.gaussian_eq(distances, motif_length)  # motif length is sigma

        for j in range(len(distances)):
            results.append(scores[j] * weights[j])

        return sum(results)

    @staticmethod
    def distance_from_window(idx_list, window_start, window_end):
        """
        Distance in bp from motif start to nearest idx inside vcf window
        returns 0 if within the window
        :param idx_list:
        :param window_start:
        :param window_end:
        :return:
        """
        idxs = np.array(idx_list)
        dist = np.where(idxs < window_start,
                        window_start - idxs,
                        np.where(idxs > window_end, idxs - window_end, 0))
        return dist.tolist()

    @staticmethod
    def gaussian_eq(distances, sigma):
        distances = np.asarray(distances)
        weights = np.exp(-(distances**2) / (2 * sigma**2))
        weights[distances==0] = 1.0
        return weights

    @staticmethod
    def non_ambi_seq(seq):
        return ''.join(random.choice(IUPAC_CODES[char]) for char in seq)


    @staticmethod
    def get_threshold(pwm, threshold):
        """
        Calculate threshold based on PWM min/max scores
        :param pwm:
        :param threshold: input this as a percentile e.g. 0.75 for 75%
        :return:
        """
        background_prob = 0.25

        theoretical_max = 0
        theoretical_min = 0

        for position in range(pwm.shape[0]):
            position_nums = pwm[position]
            position_num_safe = np.maximum(position_nums, 1e-10)
            log_odds = np.log2(position_num_safe / background_prob)

            theoretical_max += np.max(log_odds)
            theoretical_min += np.min(log_odds)

        motif_spec_threshold = theoretical_min + (threshold *(theoretical_max - theoretical_min))

        return motif_spec_threshold