# Navigation help
import yaml
from pathlib import Path

# math
import numpy as np
import pandas as pd

# helpers
import re
from tqdm import tqdm

# multiprocessing
import multiprocessing
from multiprocessing import Pool

from b00_bio_library import AMBIGUOUS, IUPAC_CODES, POLY_TRACTS, NLS, NES

global_config = {}

def config_init(config, ambiguous, iupac_codes, nls, nes):
    global global_config, AMBIGUOUS, IUPAC_CODES, NLS, NES
    global_config = config
    AMBIGUOUS = ambiguous
    IUPAC_CODES = iupac_codes
    NLS = nls
    NES = nes



class ProtMatrix:
    """
    Motif scanning class - scans various motifs on both DNA and protein level
    """
    _pool = None

    def __init__(self, core_num=None):
        self.root = Path(__file__).parent.resolve()
        with open(self.root / 'config.yaml', 'r') as outfile:
            self.cfg = yaml.safe_load(outfile)

        self.search_radius = self.cfg['motif_search_radius']

        motif_database = self.root / self.cfg['database_folder'] / self.cfg['pwm_folder']
        aa_motif_path = motif_database / self.cfg['aa_motifs']

        # === load up motif PWMs globally for repeated reference ===

        # [AA PWMs]
        # [post-translational modification domains]
        cdkphos_mod_pwm = np.loadtxt(aa_motif_path / 'MOD_CDK_SPxK_pwm.txt')  # - Phosphorylation
        camppka_pwm = np.loadtxt(aa_motif_path / 'cAMP_pka_pwm.txt')
        ck2_pwm = np.loadtxt(aa_motif_path / 'ck2_pwm.txt')
        tyr_csk_pwm = np.loadtxt(aa_motif_path / 'tyr_csk_pwm.txt')

        ngly_1_pwm = np.loadtxt(aa_motif_path / 'mod_ngly_1_pwm.txt')  # - Glycosylation
        ngly_2_pwm = np.loadtxt(aa_motif_path / 'mod_ngly_2_pwm.txt')

        dbox_pwm = np.loadtxt(aa_motif_path / 'dbox_pwm.txt') # - Ubiquitination
        kenbox_pwm = np.loadtxt(aa_motif_path / 'kenbox_pwm.txt')



        # create config for downstream multiproc - these get added to global_config -> e.g. pwm = global_config['pwm_name_here']
        self.config = {
            'search_radius': self.search_radius,

            # Protein motifs
            # - Phosphorylation motifs
            'cdkphos_pwm': cdkphos_mod_pwm,
            'camppka_pwm': camppka_pwm,
            'ck2_pwm': ck2_pwm,
            'tyrcsk_pwm': tyr_csk_pwm,

            # - Glycosylation motifs -- Had way too much trouble trying to find more
            'ngly_1_pwm': ngly_1_pwm,
            'ngly_2_pwm': ngly_2_pwm,

            # - Ubiquitination motifs
            'dbox_pwm': dbox_pwm,
            'kenbox_pwm': kenbox_pwm,

        }

        # define number of cores
        self.core_num = core_num if core_num is not None else multiprocessing.cpu_count() - 2

        # initialize pool if not already initialized
        if ProtMatrix._pool is None:
            self.initialize_pool()


    def initialize_pool(self):
        """
        Initialize multiprocessing pool w/ provided configuration and data
        """
        core_num = self.core_num

        if ProtMatrix._pool is None:
            ProtMatrix._pool = multiprocessing.Pool(
                processes = core_num,
                initializer = config_init,
                initargs=(self.config, AMBIGUOUS, IUPAC_CODES, NLS, NES)
            )

    def terminate_pool(self):
        """
        Terminate multiprocessing pool when no longer needed
        """
        if ProtMatrix._pool is not None:
            ProtMatrix._pool.close()
            ProtMatrix._pool.join()
            ProtMatrix._pool = None

    def __enter__(self):
        self.initialize_pool()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate_pool()



    # ====[[PWM MOTIF FINGERPRINT DATAFRAME]]====
    @staticmethod
    def gen_AAPWM_dataframe(dataframe):
        """
        Uses persistent mp pool to generate DNA mutation data fingerprint
        :param dataframe:
        :return:
        """
        fingerprint_rows = [
            (row['non_ambiguous_ref'], row['non_ambiguous_alt'])
            for _, row in dataframe.iterrows()
        ]

        fingerprint_rows = list(tqdm(
            # pool.imap method applies function self.PWM_profile_wrapper to each row in fingerprint_rows
            ProtMatrix._pool.imap(ProtMatrix.PWM_profile_wrapper, fingerprint_rows),
            total=len(fingerprint_rows),
            desc="[Generating AA profile fingerprints -- Cluster distance-weighted composite scoring]"
        ))

        fingerprint_df = pd.DataFrame(fingerprint_rows)
        fingerprint_df = pd.concat([dataframe.reset_index(drop=True), fingerprint_df],
                                   axis=1)  # axis = 1 to concatenate column wise (side by side)

        fingerprint_df = fingerprint_df.drop(['Chromosome', 'ReferenceAlleleVCF', 'AlternateAlleleVCF',
                                              'Flank_1', 'Flank_2', 'ClinicalSignificance', 'ref_protein_list',
                                              'alt_protein_list', 'ref_protein_length', 'alt_protein_length',
                                              'non_ambiguous_ref', 'non_ambiguous_alt'], axis=1)

        return fingerprint_df

    @staticmethod
    def PWM_profile_wrapper(fp_row):
        """
        multiprocessing wrapper, processes a single row
        :param DNA dataframe fp_row:
        :return:
        """
        ref_protein, alt_protein = fp_row

        fp = {}
        fp.update(ProtMatrix.AA_pwm_profile(ref_protein, alt_protein))

        return fp


    @staticmethod
    def AA_pwm_profile(nonambi_prot_ref, nonambi_prot_alt):
        """
        :param nonambi_prot_ref:
        :param nonambi_prot_alt:
        :return:
        """
        # ====[PWM MOTIF DISRUPTIONS]====
        # Combines gaussian scoring and pwm navigation to create a motif disruption score for each specific motif
        # For now, let's see how impactful each motif search is going to be for XGBoost
        aa_alphabet = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
                       'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
                       'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}

        pwm_dict = {}

        total_ref_idxs = []
        total_alt_idxs = []
        total_ref_scores = []
        total_alt_scores = []

        # ===[Phosphorylation profile]===
        phos_dict, phos_ref_idxs, phos_alt_idxs, phos_ref_scores, phos_alt_scores = ProtMatrix.phosphorylation_profile(nonambi_prot_ref, nonambi_prot_alt, aa_alphabet)
        total_ref_idxs.extend(phos_ref_idxs)
        total_alt_idxs.extend(phos_alt_idxs)
        total_ref_scores.extend(phos_ref_scores)
        total_alt_scores.extend(phos_alt_scores)

        # ===[Glycosylation profile]===
        glyc_dict, glyc_ref_idxs, glyc_alt_idxs, glyc_ref_scores, glyc_alt_scores = ProtMatrix.glycosylation_profile(nonambi_prot_ref, nonambi_prot_alt, aa_alphabet)
        total_ref_idxs.extend(glyc_ref_idxs)
        total_alt_idxs.extend(glyc_alt_idxs)
        total_ref_scores.extend(glyc_ref_scores)
        total_alt_scores.extend(glyc_alt_scores)

        # ===[Ubiquitination profile]===
        ubiq_dict, ubiq_ref_idxs, ubiq_alt_idxs, ubiq_ref_scores, ubiq_alt_scores = ProtMatrix.ubiquitination_profile(nonambi_prot_ref, nonambi_prot_alt, aa_alphabet)
        total_ref_idxs.extend(ubiq_ref_idxs)
        total_alt_idxs.extend(ubiq_alt_idxs)
        total_ref_scores.extend(ubiq_ref_scores)
        total_alt_scores.extend(ubiq_alt_scores)

        # --- Finalize DataFrame: bulk update + finishing details ---
        pwm_dict.update(phos_dict)
        pwm_dict.update(glyc_dict)
        pwm_dict.update(ubiq_dict)

        # identification of overall cluster disruption
        pwm_dict['Total_cluster_delta'] = ProtMatrix.cluster_delta(total_ref_idxs, total_alt_idxs, total_ref_scores, total_alt_scores)

        pwm_dict['NLS_delta'] = ProtMatrix.regex_motif_delta(nonambi_prot_ref, nonambi_prot_alt, NLS)
        pwm_dict['NES_delta'] = ProtMatrix.regex_motif_delta(nonambi_prot_ref, nonambi_prot_alt, NES)

        return pwm_dict


    @staticmethod
    def phosphorylation_profile(nonambi_prot_ref, nonambi_prot_alt, aa_alphabet):
        phosphorylation_dict = {}

        all_ref_idxs = []
        all_alt_idxs = []
        all_ref_scores = []
        all_alt_scores = []

        # === PHOSPHORYLATION ===
        cdkphos_count, cdkphos_score, cdkphos_ref_idxs, cdkphos_alt_idxs, cdkphos_ref_scores, cdkphos_alt_scores = (
            ProtMatrix.AA_pwm_stats(global_config['cdkphos_pwm'],
                                    nonambi_prot_ref, nonambi_prot_alt,
                                    aa_alphabet, 0.7))
        all_ref_idxs.extend(cdkphos_ref_idxs)
        all_alt_idxs.extend(cdkphos_alt_idxs)
        all_ref_scores.extend(cdkphos_ref_scores)
        all_alt_scores.extend(cdkphos_alt_scores)


        camppka_count, camppka_score, camppka_ref_idxs, camppka_alt_idxs, camppka_ref_scores, camppka_alt_scores = (
            ProtMatrix.AA_pwm_stats(global_config['camppka_pwm'],
                                    nonambi_prot_ref, nonambi_prot_alt,
                                    aa_alphabet, 0.7))
        all_ref_idxs.extend(camppka_ref_idxs)
        all_alt_idxs.extend(camppka_alt_idxs)
        all_ref_scores.extend(camppka_ref_scores)
        all_alt_scores.extend(camppka_alt_scores)

        ck2_count, ck2_score, ck2_ref_idxs, ck2_alt_idxs, ck2_ref_scores, ck2_alt_scores = (
            ProtMatrix.AA_pwm_stats(global_config['ck2_pwm'],
                                    nonambi_prot_ref, nonambi_prot_alt,
                                    aa_alphabet, 0.7))
        all_ref_idxs.extend(ck2_ref_idxs)
        all_alt_idxs.extend(ck2_alt_idxs)
        all_ref_scores.extend(ck2_ref_scores)
        all_alt_scores.extend(ck2_alt_scores)

        tyrcsk_count, tyrcsk_score, tyrcsk_ref_idxs, tyrcsk_alt_idxs, tyrcsk_ref_scores, tyrcsk_alt_scores= (
            ProtMatrix.AA_pwm_stats(global_config['tyrcsk_pwm'],
                                    nonambi_prot_ref, nonambi_prot_alt,
                                    aa_alphabet, 0.7))
        all_ref_idxs.extend(tyrcsk_ref_idxs)
        all_alt_idxs.extend(tyrcsk_alt_idxs)
        all_ref_scores.extend(tyrcsk_ref_scores)
        all_alt_scores.extend(tyrcsk_alt_scores)


        phosphorylation_dict['cdkphos_count'] = cdkphos_count
        phosphorylation_dict['cdkphos_score'] = cdkphos_score

        phosphorylation_dict['camppka_count'] = camppka_count
        phosphorylation_dict['camppka_score'] = camppka_score

        phosphorylation_dict['ck2_count'] = ck2_count
        phosphorylation_dict['ck2_score'] = ck2_score

        phosphorylation_dict['tyrcsk_count'] = tyrcsk_count
        phosphorylation_dict['tyrcsk_score'] = tyrcsk_score

        phosphorylation_dict['phosphorylation_count_delta'] = cdkphos_count + camppka_count + ck2_count + tyrcsk_count
        phosphorylation_dict['phosphorylation_score_delta'] = cdkphos_score + camppka_score + ck2_score + tyrcsk_score

        # cluster scores
        cluster_delta = ProtMatrix.cluster_delta(all_ref_idxs, all_alt_idxs, all_ref_scores, all_alt_scores)
        phosphorylation_dict['phos_cluster_delta'] = cluster_delta

        return phosphorylation_dict, all_ref_idxs, all_alt_idxs, all_ref_scores, all_alt_scores


    @staticmethod
    def glycosylation_profile(nonambi_prot_ref, nonambi_prot_alt, aa_alphabet):
        glycosylation_dict = {}
        all_ref_idxs = []
        all_alt_idxs = []
        all_ref_scores = []
        all_alt_scores = []

        # === GLYCOSYLATION ===
        ngly_1_count, ngly_1_score, ngly_1_ref_idxs, ngly_1_alt_idxs, ngly_1_ref_scores, ngly_1_alt_scores = (
            ProtMatrix.AA_pwm_stats(global_config['ngly_1_pwm'],
                                    nonambi_prot_ref, nonambi_prot_alt,
                                    aa_alphabet, 0.7))
        all_ref_idxs.extend(ngly_1_ref_idxs)
        all_alt_idxs.extend(ngly_1_alt_idxs)
        all_ref_scores.extend(ngly_1_ref_scores)
        all_alt_scores.extend(ngly_1_alt_scores)


        ngly_2_count, ngly_2_score, ngly_2_ref_idxs, ngly_2_alt_idxs, ngly_2_ref_scores, ngly_2_alt_scores = (
            ProtMatrix.AA_pwm_stats(global_config['ngly_2_pwm'],
                                    nonambi_prot_ref, nonambi_prot_alt,
                                    aa_alphabet, 0.7))
        all_ref_idxs.extend(ngly_2_ref_idxs)
        all_alt_idxs.extend(ngly_2_alt_idxs)
        all_ref_scores.extend(ngly_2_ref_scores)
        all_alt_scores.extend(ngly_2_alt_scores)

        glycosylation_dict['ngly_1_count'] = ngly_1_count
        glycosylation_dict['ngly_1_score'] = ngly_1_score

        glycosylation_dict['ngly_2_count'] = ngly_2_count
        glycosylation_dict['ngly_2_score'] = ngly_2_score

        glycosylation_dict['glycosylation_count_delta'] = ngly_1_count + ngly_2_count
        glycosylation_dict['glycosylation_score_delta'] = ngly_1_score + ngly_2_score

        # cluster scores
        cluster_delta = ProtMatrix.cluster_delta(all_ref_idxs, all_alt_idxs, all_ref_scores, all_alt_scores)
        glycosylation_dict['glyc_cluster_delta'] = cluster_delta

        return glycosylation_dict, all_ref_idxs, all_alt_idxs, all_ref_scores, all_alt_scores


    @staticmethod
    def ubiquitination_profile(nonambi_prot_ref, nonambi_prot_alt, aa_alphabet):
        ubiquitination_dict = {}
        all_ref_idxs = []
        all_alt_idxs = []
        all_ref_scores = []
        all_alt_scores = []

        # === UBIQUITINATION ===
        dbox_count, dbox_score, dbox_ref_idxs, dbox_alt_idxs, dbox_ref_scores, dbox_alt_scores = (
            ProtMatrix.AA_pwm_stats(global_config['dbox_pwm'],
                                    nonambi_prot_ref, nonambi_prot_alt,
                                    aa_alphabet, 0.7))
        all_ref_idxs.extend(dbox_ref_idxs)
        all_alt_idxs.extend(dbox_alt_idxs)
        all_ref_scores.extend(dbox_ref_scores)
        all_alt_scores.extend(dbox_alt_scores)


        kenbox_count, kenbox_score, kenbox_ref_idxs, kenbox_alt_idxs, kenbox_ref_scores, kenbox_alt_scores = (
            ProtMatrix.AA_pwm_stats(global_config['kenbox_pwm'],
                                    nonambi_prot_ref, nonambi_prot_alt,
                                    aa_alphabet, 0.7))
        all_ref_idxs.extend(kenbox_ref_idxs)
        all_alt_idxs.extend(kenbox_alt_idxs)
        all_ref_scores.extend(kenbox_ref_scores)
        all_alt_scores.extend(kenbox_alt_scores)


        ubiquitination_dict['dbox_count'] = dbox_count
        ubiquitination_dict['dbox_score'] = dbox_score

        ubiquitination_dict['kenbox_count'] = kenbox_count
        ubiquitination_dict['kenbox_score'] = kenbox_score

        ubiquitination_dict['ubiquitination_count_delta'] = dbox_count + kenbox_count
        ubiquitination_dict['ubiquitination_score_delta'] = dbox_score + kenbox_score

        cluster_delta = ProtMatrix.cluster_delta(all_ref_idxs, all_alt_idxs, all_ref_scores, all_alt_scores)
        ubiquitination_dict['ubiq_cluster_delta'] = cluster_delta

        return ubiquitination_dict, all_ref_idxs, all_alt_idxs, all_ref_scores, all_alt_scores


    @staticmethod
    def AA_pwm_stats(pwm, prot_ref, prot_alt, alphabet, threshold):
        """
        handles motif finding for amino acid sequences - no position weight scoring since we are taking the most likely protein to be produced with no coordinates
        :param pwm:
        :param prot_ref:
        :param prot_alt:
        :param alphabet:
        :param threshold:
        :return: motif_quantity_delta, motif_score_delta, ref_motif_idxs, alt_motif_idxs, ref_motif_scores, alt_motif_scores
        """
        motif_size = pwm.shape[0]

        ref_motif_idxs, ref_motif_scores = ProtMatrix.probability_all_pos(prot_ref, motif_size, pwm, alphabet, threshold)
        alt_motif_idxs, alt_motif_scores = ProtMatrix.probability_all_pos(prot_alt, motif_size, pwm, alphabet, threshold)

        motif_quantity_delta = len(alt_motif_idxs) - len(ref_motif_idxs)
        motif_score_delta = sum(alt_motif_scores) - sum(ref_motif_scores)


        return motif_quantity_delta, motif_score_delta, ref_motif_idxs, alt_motif_idxs, ref_motif_scores, alt_motif_scores

    @staticmethod
    def probability_all_pos(sequence, motif_size, pwm, alphabet, threshold):
        """
        Performs sliding window and returns list of indices that are likely to contain the motif
        :param full sequence:
        :param motif_size:
        :param pwm:
        :param alphabet:
        :param threshold:
        :return: list of each probable motif location
        """
        motif_threshold = ProtMatrix.get_threshold(pwm, threshold)

        seq_len = len(sequence)
        idxs, scores = [], []
        for i in range(seq_len - motif_size + 1):  # proper search space handled
            score = ProtMatrix.probability_subseq(sequence[i:i + motif_size], pwm, alphabet)
            if score > motif_threshold:
                idxs.append(i)  # contains the index the motif was found
                scores.append(score)  # contains the score that index contained
        return idxs, scores


    @staticmethod
    def probability_subseq(subseq, pwm, alphabet):
        """
        Calculate the probability this sequence will contain the motif model
        :param subseq:
        :param pwm:
        :param alphabet
        :return: probability float
        """
        background_prob = 1 / len(alphabet)  # background can tell us if seq is DNA or AA

        nuc_idx = np.fromiter((alphabet[c] for c in subseq), dtype=np.int8)
        probs = pwm[np.arange(len(subseq)), nuc_idx]
        # handle 0s

        probs = np.maximum(probs, 1e-10)
        scores = np.log2(probs / background_prob)
        return scores.sum()


    @staticmethod
    def get_threshold(pwm, threshold):
        """
        Calculate threshold based on PWM min/max scores
        :param pwm:
        :param threshold: input this as a percentile e.g. 0.75 for 75%
        :return:
        """
        background_prob = 1 / pwm.shape[1]   # [0] is the length of the motif, [1] is the number of possible characters

        theoretical_max = 0
        theoretical_min = 0

        for position in range(pwm.shape[0]):
            position_nums = pwm[position]
            position_num_safe = np.maximum(position_nums, 1e-10)
            log_odds = np.log2(position_num_safe / background_prob)

            theoretical_max += np.max(log_odds)
            theoretical_min += np.min(log_odds)

        motif_spec_threshold = theoretical_min + (threshold * (theoretical_max - theoretical_min))

        return motif_spec_threshold


    @staticmethod
    def regex_motif_delta(nonambi_ref, nonambi_alt, regex_motif):
        return ProtMatrix.count_regex(nonambi_alt, regex_motif) - ProtMatrix.count_regex(nonambi_ref, regex_motif)


    @staticmethod
    def count_regex(sequence, regex_list):
        counts = []
        for motif in regex_list:
            counts.append(len(re.findall(motif, sequence)))
        return sum(counts)


    @staticmethod
    def cluster_delta(ref_idxs, alt_idxs, ref_scores, alt_scores):
        """Cluster score determined by composite inverse distance scoring"""
        return ProtMatrix.cluster_finder(alt_idxs, alt_scores) - ProtMatrix.cluster_finder(ref_idxs, ref_scores)


    @staticmethod
    def cluster_finder(idxs, scores, max_distance=30):
        cluster_score = 0
        for pos in range(len(idxs) - 1):  # motifs within 30 are usually considered to be a cluster (maybe change this if proven otherwise)
            distance = idxs[pos + 1] - idxs[pos]
            if distance <= 0:
                continue

            if distance <= max_distance:
                cluster_score += (scores[pos] + scores[pos + 1]) / (distance + 1)

        return cluster_score



