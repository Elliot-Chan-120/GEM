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

global_config = {}

class DNAMatrixProfile:
    _pool = None

    def __init__(self, core_num=None):
        self.root = Path(__file__).parent.resolve()

        # === load up motif PWMs globally for repeated reference ===
        motif_path = Path('database/pwm_database')
        splice_3_pwm = np.loadtxt(motif_path / '3_splice_pwm.txt')  # human donor splice site (3')
        splice_5_pwm = np.loadtxt(motif_path / '5_splice_pwm.txt')  # human acceptor splice site (5')
        branch_pt_pwm = np.loadtxt(
            motif_path / 'branch_pt_pwm.txt')  # human branch point -> took image from study and ran it through AI to reconstruct PWM, take this w/ grain of salt

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
        if DNAMatrixProfile._pool is None:
            self.initialize_pool()


    def initialize_pool(self):
        """
        Initialize multiprocessing pool w/ provided configuration and data
        """
        core_num = self.core_num

        if CompositeDNA._pool is None:
            CompositeDNA._pool = multiprocessing.Pool(
                processes = core_num,
                initializer=config_init,
                initargs=(self.config, )
            )

    def terminate_pool(self):
        """
        Terminate multiprocessing pool when no longer needed
        """
        if CompositeDNA._pool is not None:
            CompositeDNA._pool.close()
            CompositeDNA._pool.join()
            CompositeDNA._pool = None

    def __enter__(self):
        self.initialize_pool()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate_pool()


    def DNA_profile(self, full_ref_allele, full_alt_allele, flank1, flank2, ref_vcf, alt_vcf):
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
        non_ambi_ref = self.non_ambi_seq(full_ref_allele)
        non_ambi_alt = self.non_ambi_seq(full_alt_allele)

        # splice sites + branch points
        sp3_count, sp3_score = self.pwm_stats(SPLICE_3, non_ambi_ref, non_ambi_alt, flank1, flank2, ref_vcf, alt_vcf)
        sp5_count, sp5_score = self.pwm_stats(SPLICE_5, non_ambi_ref, non_ambi_alt, flank1, flank2, ref_vcf, alt_vcf)
        branch_pt_count, branch_pt_score = self.pwm_stats(BRANCH_PT, sp3_count, sp3_score, flank1, flank2, ref_vcf, alt_vcf)

        # transcription factors
        caat_count, caat_score = self.pwm_stats(CAAT, non_ambi_ref, non_ambi_alt, flank1, flank2, ref_vcf, alt_vcf)
        ctcf_count, ctcf_score = self.pwm_stats(CTCF, non_ambi_ref, non_ambi_alt, flank1, flank2, ref_vcf, alt_vcf)
        tata_count, tata_score = self.pwm_stats(TATA, non_ambi_ref, non_ambi_alt, flank1, flank2, ref_vcf, alt_vcf)

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
    def pwm_stats(pwm, ref_allele, alt_allele, flank1, flank2, ref_vcf, alt_vcf):
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
        :return: quantity [0] and score delta [1]
        """
        # get results first
        motif_length = len(pwm)
        ref_motif_idxs, ref_motif_scores = probability_all_pos(ref_allele, motif_length, pwm)
        alt_motifs_idxs, alt_motif_scores = probability_all_pos(alt_allele, motif_length, pwm)

        # === quantity delta ===
        motif_quantity_delta = len(alt_motifs_idxs) - len(ref_motif_idxs)

        # === positional-strength composite score
        window_start = len(flank1)
        ref_window_end = len(flank1) + len(ref_vcf)
        alt_window_end = len(flank2) + len(alt_vcf)

        ref_weighted_score = pos_weight_gaussian(ref_motif_idxs, ref_motif_scores,
                                                 window_start, ref_window_end, motif_length)

        alt_weighted_score = pos_weight_gaussian(alt_motifs_idxs, alt_motif_scores,
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

        with np.errstate(divide="ignore"):
            scores = np.log2(probs / background_prob)

        scores[np.isneginf(scores)] = -1e9

        return scores.sum()

    @staticmethod
    def probability_all_pos(sequence, motif_size, pwm):
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
            score = probability_subseq(sequence[i:i + motif_size], pwm)
            if score > 0:
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

        distances = distance_from_window(idxs, vcf_start, vcf_end)
        weights = gaussian_eq(distances, motif_length)  # motif length is sigma

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
        return dist.to_list()

    @staticmethod
    def gaussian_eq(distances, sigma):
        distances = np.asarray(distances)
        weights = np.exp(-(distances**2) / (2 * sigma**2))
        weights[distances==0] = 1.0
        return weights

    @staticmethod
    def non_ambi_seq(seq):
        return ''.join(
            random.choice(IUPAC_CODES[char] for char in seq)
        )