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



class DNAMatrix:
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
        dna_motif_path = motif_database / self.cfg['dna_motifs']

        # === load up motif PWMs globally for repeated reference ===
        # [DNA PWMs]
        splice_3_pwm = np.loadtxt(dna_motif_path / '3_splice_pwm.txt')  # human donor splice site (3')
        splice_5_pwm = np.loadtxt(dna_motif_path / '5_splice_pwm.txt')  # human acceptor splice site (5')
        branch_pt_pwm = np.loadtxt(dna_motif_path / 'branch_pt_pwm.txt')  # human branch point -> took pwm logo from study and ran it through AI to reconstruct PWM, take this w/ grain of salt

        # simple transcription factors
        ctcf_pwm = np.loadtxt(dna_motif_path / 'CTCF_TF_pwm.txt')  # CTCF transcription factor
        caat_pwm = np.loadtxt(dna_motif_path / 'CAAT_pwm.txt')  # CAAT box TF
        tata_pwm = np.loadtxt(dna_motif_path / 'TATA_pwm.txt')  # TATA box TF


        # create config for downstream multiproc - these get added to global_config -> e.g. pwm = global_config['pwm_name_here']
        self.config = {
            'search_radius': self.search_radius,

            #  - DNA motifs
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
        if DNAMatrix._pool is None:
            self.initialize_pool()


    def initialize_pool(self):
        """
        Initialize multiprocessing pool w/ provided configuration and data
        """
        core_num = self.core_num

        if DNAMatrix._pool is None:
            DNAMatrix._pool = multiprocessing.Pool(
                processes = core_num,
                initializer = config_init,
                initargs=(self.config, AMBIGUOUS, IUPAC_CODES, NLS, NES)
            )

    def terminate_pool(self):
        """
        Terminate multiprocessing pool when no longer needed
        """
        if DNAMatrix._pool is not None:
            DNAMatrix._pool.close()
            DNAMatrix._pool.join()
            DNAMatrix._pool = None

    def __enter__(self):
        self.initialize_pool()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate_pool()



    # ====[[PWM MOTIF FINGERPRINT DATAFRAME]]====
    @staticmethod
    def gen_DNAPWM_dataframe(dataframe):
        """
        Uses persistent mp pool to generate DNA mutation data fingerprint
        :param dataframe:
        :return:
        """
        fingerprint_rows = [
            (row['Chromosome'], row['ReferenceAlleleVCF'],
             row['AlternateAlleleVCF'], row['Flank_1'], row['Flank_2'])
            for _, row in dataframe.iterrows()
        ]

        fingerprint_rows = list(tqdm(
            # pool.imap method applies function self.PWM_profile_wrapper to each row in fingerprint_rows
            DNAMatrix._pool.imap(DNAMatrix.PWM_profile_wrapper, fingerprint_rows),
            total=len(fingerprint_rows),
            desc="[Generating DNA motif fingerprints -- Gaussian-weighted composite scoring]"
        ))

        fingerprint_df = pd.DataFrame(fingerprint_rows)
        fingerprint_df = pd.concat([dataframe.reset_index(drop=True), fingerprint_df],
                                   axis=1)  # axis = 1 to concatenate column wise (side by side)

        fingerprint_df = fingerprint_df.drop(['Chromosome', 'ClinicalSignificance', 'ReferenceAlleleVCF',
                                              'AlternateAlleleVCF', 'Flank_1', 'Flank_2'], axis=1)

        return fingerprint_df

    @staticmethod
    def PWM_profile_wrapper(fp_row):
        """
        multiprocessing wrapper, processes a single row
        :param DNA dataframe fp_row:
        :return:
        """
        chromosome, ref_allele, alt_allele, flank_1, flank_2 = fp_row

        fp = {}
        fp.update(DNAMatrix.DNA_pwm_profile(flank_1, flank_2, ref_allele, alt_allele))

        return fp

    @staticmethod
    def DNA_pwm_profile(flank_1, flank_2, ref_vcf, alt_vcf):
        """
        :param flank_1:
        :param flank_2:
        :param ref_vcf:
        :param alt_vcf:
        :return:
        """
        # ====[PWM MOTIF DISRUPTIONS]====
        # Combines gaussian scoring and pwm navigation to create a motif disruption score for each specific motif
        # For now, let's see how impactful each motif search is going to be for XGBoost
        dna_alphabet = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        pwm_dict = {}

        # [1] Isolate sequence sections via search radius
        flank_length = len(flank_1)
        search_radius = global_config['search_radius']
        f1_search = flank_1[flank_length - search_radius:]
        f2_search = flank_2[:search_radius]

        ref_section = f1_search + ref_vcf + f2_search
        alt_section = f1_search + alt_vcf + f2_search


        # [2] get counts and scores of each motif listed
        # splice sites + branch points
        sp3_count, sp3_score = DNAMatrix.DNA_pwm_stats(ref_vcf, alt_vcf, flank_length,
                                                               global_config['splice_3_pwm'],
                                                               ref_section, alt_section, dna_alphabet)

        sp5_count, sp5_score = DNAMatrix.DNA_pwm_stats(ref_vcf, alt_vcf, flank_length,
                                                               global_config['splice_5_pwm'],
                                                               ref_section, alt_section, dna_alphabet)

        branch_pt_count, branch_pt_score = DNAMatrix.DNA_pwm_stats(ref_vcf, alt_vcf, flank_length,
                                                                           global_config['branch_pt_pwm'],
                                                                           ref_section, alt_section, dna_alphabet)

        # transcription factors
        caat_count, caat_score = DNAMatrix.DNA_pwm_stats(ref_vcf, alt_vcf, flank_length,
                                                                 global_config['caat_pwm'],
                                                                 ref_section, alt_section, dna_alphabet)

        ctcf_count, ctcf_score = DNAMatrix.DNA_pwm_stats(ref_vcf, alt_vcf, flank_length,
                                                                 global_config['ctcf_pwm'],
                                                                 ref_section, alt_section, dna_alphabet)

        tata_count, tata_score = DNAMatrix.DNA_pwm_stats(ref_vcf, alt_vcf, flank_length,
                                                                 global_config['tata_pwm'],
                                                                 ref_section, alt_section, dna_alphabet)

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
    def DNA_pwm_stats(ref_vcf, alt_vcf, flank_length, pwm, ref_section, alt_section, alphabet):
        """
        CALL ON THIS for DNA sequences
        motif stat changes due to mutation
        :param ref_vcf:
        :param alt_vcf:
        :param flank_length:
        :param pwm:
        :param ref_section:
        :param alt_section:
        :param alphabet:
        :return: quantity [0] and score delta [1]
        """
        search_start = flank_length - global_config['search_radius']

        motif_length = pwm.shape[0]
        ref_motif_idxs, ref_motif_scores = DNAMatrix.probability_all_pos(ref_section, motif_length, pwm, alphabet)
        alt_motif_idxs, alt_motif_scores = DNAMatrix.probability_all_pos(alt_section, motif_length, pwm, alphabet)

        # readjust indices back to full sequence coordinates before Gaussian weights
        ref_idxs_adjusted = [idx + search_start for idx in ref_motif_idxs]
        alt_idxs_adjusted = [idx + search_start for idx in alt_motif_idxs]

        # === quantity delta ===
        motif_quantity_delta = len(alt_motif_idxs) - len(ref_motif_idxs)

        # === positional-strength composite scoring ===
        window_start = flank_length
        ref_window_end = window_start + len(ref_vcf)
        alt_window_end = window_start + len(alt_vcf)

        ref_weighted_score = DNAMatrix.pos_weight_gaussian(ref_idxs_adjusted, ref_motif_scores,
                                                 window_start, ref_window_end, motif_length)

        alt_weighted_score = DNAMatrix.pos_weight_gaussian(alt_idxs_adjusted, alt_motif_scores,
                                                 window_start, alt_window_end, motif_length)

        position_score_delta = alt_weighted_score - ref_weighted_score


        return motif_quantity_delta, position_score_delta


    @staticmethod
    def probability_all_pos(sequence, motif_size, pwm, alphabet):
        """
        Performs sliding window and returns list of indices that are likely to contain the motif
        :param full sequence:
        :param motif_size:
        :param pwm:
        :param alphabet:
        :return: list of each probable motif location
        """
        seq_len = len(sequence)
        idxs, scores = [], []
        for i in range(seq_len - motif_size + 1):  # proper search space handled
            score = DNAMatrix.probability_subseq(sequence[i:i + motif_size], pwm, alphabet)
            if score > 0.001:
                idxs.append(i)
                scores.append(score)
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
        background_prob = 1 / len(alphabet)

        total_score = 0

        for i, base in enumerate(subseq):
            if base in AMBIGUOUS:
                possible_bases = IUPAC_CODES[base]
                prob = sum(pwm[i][alphabet[b]] for b in possible_bases) / len(possible_bases)
            else:
                prob = pwm[i][alphabet[base]]

            prob = max(prob, 1e-10)
            total_score += np.log2(prob / background_prob)

        return total_score


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

        distances = DNAMatrix.distance_from_window(idxs, vcf_start, vcf_end)
        weights = DNAMatrix.gaussian_eq(distances, motif_length)  # motif length is sigma

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
        return DNAMatrix.count_regex(nonambi_alt, regex_motif) - DNAMatrix.count_regex(nonambi_ref, regex_motif)


    @staticmethod
    def count_regex(sequence, regex_list):
        counts = []
        for motif in regex_list:
            counts.append(len(re.findall(motif, sequence)))
        return sum(counts)
