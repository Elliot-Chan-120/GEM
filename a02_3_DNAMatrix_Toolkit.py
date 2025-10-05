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
        self.cluster_threshold = self.cfg['dna_cluster_distance']

        motif_database = self.root / self.cfg['database_folder'] / self.cfg['pwm_folder']
        dna_motif_path = motif_database / self.cfg['dna_motifs']

        # === load up motif PWMs in config "global_config['pwm_name'] ===
        # [Initiation]
        inr_pwm = np.loadtxt(dna_motif_path / 'inr_pwm.txt')
        tata_pwm = np.loadtxt(dna_motif_path / 'TATA_pwm.txt')
        kozak_pwm = np.loadtxt(dna_motif_path / 'kozak_pwm.txt')

        # [Transcription Factors]
        ctcf_pwm = np.loadtxt(dna_motif_path / 'CTCF_TF_pwm.txt')  # CTCF transcription factor
        caat_pwm = np.loadtxt(dna_motif_path / 'CAAT_pwm.txt')  # CAAT box TF
        sp1_pwm = np.loadtxt(dna_motif_path / 'sp1_pwm.txt')
        nfkb_pwm = np.loadtxt(dna_motif_path / 'nfkb_pwm.txt')
        ap1_pwm = np.loadtxt(dna_motif_path / 'ap1_pwm.txt')
        creb_pwm = np.loadtxt(dna_motif_path / 'creb_pwm.txt')

        # [Post-translational Regulation]
        splice_3_pwm = np.loadtxt(dna_motif_path / '3_splice_pwm.txt')  # human donor splice site (3')
        splice_5_pwm = np.loadtxt(dna_motif_path / '5_splice_pwm.txt')  # human acceptor splice site (5')
        branch_pt_pwm = np.loadtxt(dna_motif_path / 'branch_pt_pwm.txt')  # human branch point -> took pwm logo from study and ran it through AI to reconstruct PWM, take this w/ grain of salt
        polyadenylation_pwm = np.loadtxt(dna_motif_path / 'polyad_pwm.txt')


        # create config for downstream multiproc - these get added to global_config -> e.g. pwm = global_config['pwm_name_here']
        self.config = {
            'search_radius': self.search_radius,
            'cluster_distance': self.cluster_threshold,

            # [Initiation]
            'inr_pwm': inr_pwm,
            'tata_pwm': tata_pwm,
            'kozak_pwm': kozak_pwm,

            # [Transcription Factors]
            'ctcf_pwm': ctcf_pwm,
            'caat_pwm': caat_pwm,
            'sp1_pwm': sp1_pwm,
            'nfkb_pwm': nfkb_pwm,
            'ap1_pwm': ap1_pwm,
            'creb_pwm': creb_pwm,

            # [Post-Translational Regulation]
            'splice_3_pwm': splice_3_pwm,
            'splice_5_pwm': splice_5_pwm,
            'branch_pt_pwm': branch_pt_pwm,
            'polyadenylation_pwm': polyadenylation_pwm,
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
            desc="[Generating DNA motif fingerprints -- Position Weight Matrix Signals * Gaussian-weighted composite scoring + Cluster Composite Scoring]"
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

        dna_alphabet = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        # [1] Isolate sequence sections via search radius
        flank_length = len(flank_1)
        search_radius = global_config['search_radius']
        f1_search = flank_1[flank_length - search_radius:]
        f2_search = flank_2[:search_radius]

        ref_section = f1_search + ref_allele + f2_search
        alt_section = f1_search + alt_allele + f2_search

        fp = {}

        (init_dict, init_ref_idxs, init_alt_idxs,
         init_ref_scores, init_alt_scores) = DNAMatrix.INIT_profile(ref_section, alt_section,
                                                                    ref_allele, alt_allele, flank_length,
                                                                    dna_alphabet)
        init_count = init_dict['init_motif_count']
        init_score = init_dict['init_score_shift']
        init_cluster = init_dict['init_cluster_shift']
        init_domain_cluster = init_dict['init_domain_cluster_shift']

        (tf_dict, tf_ref_idxs, tf_alt_idxs,
         tf_ref_scores, tf_alt_scores) = DNAMatrix.TF_profile(ref_section, alt_section,
                                                              ref_allele, alt_allele, flank_length,
                                                              dna_alphabet)
        tf_count = tf_dict['TF_motif_count']
        tf_score = tf_dict['TF_score_shift']
        tf_cluster = tf_dict['TF_cluster_shift']
        tf_domain_cluster = tf_dict['TF_domain_cluster_shift']

        (ptr_dict, ptr_ref_idxs, ptr_alt_idxs,
         ptr_ref_scores, ptr_alt_scores) = DNAMatrix.PTR_profile(ref_section, alt_section,
                                                                 ref_allele, alt_allele, flank_length,
                                                                 dna_alphabet)
        ptr_count = ptr_dict['PTR_motif_count']
        ptr_score = ptr_dict['PTR_score_shift']
        ptr_cluster = ptr_dict['PTR_cluster_shift']
        ptr_domain_cluster = ptr_dict['PTR_domain_cluster_shift']


        # calculate totals - will do this for proteins later, but it's definitely more important here
        # proteins are computed via 'most probable protein' algorithm so it just works differently and certain attributes may not mean the same thing

        TOTAL_COUNT = init_count + tf_count + ptr_count
        TOTAL_SCORE = init_score + tf_score + ptr_score
        TOTAL_CLUSTER = init_cluster + tf_cluster + ptr_cluster
        TOTAL_DOMAIN_CLUSTER = init_domain_cluster + tf_domain_cluster + ptr_domain_cluster
        # to do -> try to figure out a way to calculate DNA domain shift

        # dictionary assembly
        fp.update(init_dict)
        fp.update(tf_dict)
        fp.update(ptr_dict)

        fp['TOTAL_DNA_MOTIF_COUNT'] = TOTAL_COUNT
        fp['TOTAL_DNA_MOTIF_SCORE'] = TOTAL_SCORE
        fp['TOTAL_DNA_MOTIF_CLUSTER'] = TOTAL_CLUSTER
        fp['TOTAL_DNA_MOTIF_DOMAIN_CLUSTER'] = TOTAL_DOMAIN_CLUSTER

        return fp

    @staticmethod
    def INIT_profile(ref_section, alt_section, ref_vcf, alt_vcf, flank_length, dna_alphabet):
        """
        :param ref_section:
        :param alt_section:
        :param ref_vcf:
        :param alt_vcf:
        :param flank_length:
        :param dna_alphabet:
        :return:
        """
        all_ref_idxs = []
        all_alt_idxs = []
        all_ref_scores = []
        all_alt_scores = []

        (inr_count, inr_score, inr_cluster,
         inr_refidxs, inr_altidxs,
         inr_refscores, inr_altscores) = DNAMatrix.DNA_pwm_stats(ref_vcf, alt_vcf, flank_length,
                                                                 global_config['inr_pwm'],
                                                                 ref_section, alt_section, dna_alphabet,
                                                                 cluster_threshold=0.9)
        all_ref_idxs.extend(inr_refidxs)
        all_alt_idxs.extend(inr_altidxs)
        all_ref_scores.extend(inr_refscores)
        all_alt_scores.extend(inr_altscores)

        (tata_count, tata_score, tata_cluster,
         tata_refidxs, tata_altidxs,
         tata_refscores, tata_altscores) = DNAMatrix.DNA_pwm_stats(ref_vcf, alt_vcf, flank_length,
                                                                             global_config['tata_pwm'],
                                                                             ref_section, alt_section, dna_alphabet,
                                                                             cluster_threshold=0.9)
        all_ref_idxs.extend(tata_refidxs)
        all_alt_idxs.extend(tata_altidxs)
        all_ref_scores.extend(tata_refscores)
        all_alt_scores.extend(tata_altscores)

        (kozak_count, kozak_score, kozak_cluster,
         kozak_refidxs, kozak_altidxs,
         kozak_refscores, kozak_altscores) = DNAMatrix.DNA_pwm_stats(ref_vcf, alt_vcf, flank_length,
                                                                       global_config['kozak_pwm'],
                                                                       ref_section, alt_section, dna_alphabet,
                                                                       cluster_threshold=0.9)
        all_ref_idxs.extend(kozak_refidxs)
        all_alt_idxs.extend(kozak_altidxs)
        all_ref_scores.extend(kozak_refscores)
        all_alt_scores.extend(kozak_altscores)

        # totals
        total_motif_count = inr_count + tata_count + kozak_count
        total_score_shift = inr_score + tata_score + kozak_score
        cluster_additive = inr_cluster + tata_cluster + kozak_cluster

        # now calculate domain cluster  shifts
        domain_cluster = DNAMatrix.cluster_composite_delta(all_ref_idxs, all_alt_idxs, all_ref_scores, all_alt_scores)

        # dictionary assembly
        init_dict = {}

        init_dict['inr_count'] = inr_count
        init_dict['inr_score'] = inr_score
        init_dict['inr_cluster'] = inr_cluster

        init_dict['tata_count'] = tata_count
        init_dict['tata_score'] = tata_score
        init_dict['tata_cluster'] = tata_cluster

        init_dict['kozak_count'] = kozak_count
        init_dict['kozak_score'] = kozak_score
        init_dict['kozak_cluster'] = kozak_cluster

        # totals
        init_dict['init_motif_count'] = total_motif_count
        init_dict['init_score_shift'] = total_score_shift
        init_dict['init_cluster_shift'] = cluster_additive
        init_dict['init_domain_cluster_shift'] = domain_cluster

        return init_dict, all_ref_idxs, all_alt_idxs, all_ref_scores, all_alt_scores


    @staticmethod
    def TF_profile(ref_section, alt_section, ref_vcf, alt_vcf, flank_length, dna_alphabet):
        """
        Need to figure out which motif family to include here
        :param ref_section:
        :param alt_section:
        :param ref_vcf:
        :param alt_vcf:
        :param flank_length:
        :param dna_alphabet:
        :return:
        """
        all_ref_idxs = []
        all_alt_idxs = []
        all_ref_scores = []
        all_alt_scores = []

        (ctcf_count, ctcf_score, ctcf_cluster,
         ctcf_refidxs, ctcf_altidxs,
         ctcf_refscores, ctcf_altscores) = DNAMatrix.DNA_pwm_stats(ref_vcf, alt_vcf, flank_length,
                                                                 global_config['ctcf_pwm'],
                                                                 ref_section, alt_section, dna_alphabet,
                                                                 cluster_threshold=0.9)
        all_ref_idxs.extend(ctcf_refidxs)
        all_alt_idxs.extend(ctcf_altidxs)
        all_ref_scores.extend(ctcf_refscores)
        all_alt_scores.extend(ctcf_altscores)

        (caat_count, caat_score, caat_cluster,
         caat_refidxs, caat_altidxs,
         caat_refscores, caat_altscores) = DNAMatrix.DNA_pwm_stats(ref_vcf, alt_vcf, flank_length,
                                                                 global_config['caat_pwm'],
                                                                 ref_section, alt_section, dna_alphabet,
                                                                 cluster_threshold=0.9)
        all_ref_idxs.extend(caat_refidxs)
        all_alt_idxs.extend(caat_altidxs)
        all_ref_scores.extend(caat_refscores)
        all_alt_scores.extend(caat_altscores)

        (sp1_count, sp1_score, sp1_cluster,
         sp1_refidxs, sp1_altidxs,
         sp1_refscores, sp1_altscores) = DNAMatrix.DNA_pwm_stats(ref_vcf, alt_vcf, flank_length,
                                                                             global_config['sp1_pwm'],
                                                                             ref_section, alt_section, dna_alphabet,
                                                                             cluster_threshold=0.9)
        all_ref_idxs.extend(sp1_refidxs)
        all_alt_idxs.extend(sp1_altidxs)
        all_ref_scores.extend(sp1_refscores)
        all_alt_scores.extend(sp1_altscores)

        (nfkb_count, nfkb_score, nfkb_cluster,
         nfkb_refidxs, nfkb_altidxs,
         nfkb_refscores, nfkb_altscores) = DNAMatrix.DNA_pwm_stats(ref_vcf, alt_vcf, flank_length,
                                                                       global_config['nfkb_pwm'],
                                                                       ref_section, alt_section, dna_alphabet,
                                                                       cluster_threshold=0.9)
        all_ref_idxs.extend(nfkb_refidxs)
        all_alt_idxs.extend(nfkb_altidxs)
        all_ref_scores.extend(nfkb_refscores)
        all_alt_scores.extend(nfkb_altscores)

        (ap1_count, ap1_score, ap1_cluster,
         ap1_refidxs, ap1_altidxs,
         ap1_refscores, ap1_altscores) = DNAMatrix.DNA_pwm_stats(ref_vcf, alt_vcf, flank_length,
                                                                 global_config['ap1_pwm'],
                                                                 ref_section, alt_section, dna_alphabet,
                                                                 cluster_threshold=0.9)
        all_ref_idxs.extend(ap1_refidxs)
        all_alt_idxs.extend(ap1_altidxs)
        all_ref_scores.extend(ap1_refscores)
        all_alt_scores.extend(ap1_altscores)

        (creb_count, creb_score, creb_cluster,
         creb_refidxs, creb_altidxs,
         creb_refscores, creb_altscores) = DNAMatrix.DNA_pwm_stats(ref_vcf, alt_vcf, flank_length,
                                                                 global_config['creb_pwm'],
                                                                 ref_section, alt_section, dna_alphabet,
                                                                 cluster_threshold=0.9)
        all_ref_idxs.extend(creb_refidxs)
        all_alt_idxs.extend(creb_altidxs)
        all_ref_scores.extend(creb_refscores)
        all_alt_scores.extend(creb_altscores)



        total_motif_count = ctcf_count + caat_count + sp1_count + nfkb_count + ap1_count + creb_count
        total_score_shift = ctcf_score + caat_score + sp1_score + nfkb_score + ap1_score + creb_score
        cluster_additive = ctcf_cluster + caat_cluster + sp1_cluster + nfkb_cluster + ap1_cluster + creb_cluster
        domain_cluster = DNAMatrix.cluster_composite_delta(all_ref_idxs, all_alt_idxs, all_ref_scores, all_alt_scores)

        tf_dict = {}

        tf_dict['ctcf_count'] = ctcf_count
        tf_dict['ctcf_score'] = ctcf_score
        tf_dict['ctcf_cluster'] = ctcf_cluster

        tf_dict['caat_count'] = caat_count
        tf_dict['caat_score'] = caat_score
        tf_dict['caat_cluster'] = caat_cluster

        tf_dict['sp1_count'] = sp1_count
        tf_dict['sp1_score'] = sp1_score
        tf_dict['sp1_cluster'] = sp1_cluster

        tf_dict['nfkb_count'] = nfkb_count
        tf_dict['nfkb_score'] = nfkb_score
        tf_dict['nfkb_cluster'] = nfkb_cluster

        tf_dict['ap1_count'] = ap1_count
        tf_dict['ap1_score'] = ap1_score
        tf_dict['ap1_cluster'] = ap1_cluster

        tf_dict['creb_count'] = creb_count
        tf_dict['creb_score'] = creb_score
        tf_dict['creb_cluster'] = creb_cluster


        # totals
        tf_dict['TF_motif_count'] = total_motif_count
        tf_dict['TF_score_shift'] = total_score_shift
        tf_dict['TF_cluster_shift'] = cluster_additive
        tf_dict['TF_domain_cluster_shift'] = domain_cluster

        return tf_dict, all_ref_idxs, all_alt_idxs, all_ref_scores, all_alt_scores

    @staticmethod
    def PTR_profile(ref_section, alt_section, ref_vcf, alt_vcf, flank_length, dna_alphabet):
        """
        Post translational regulation motif profile
        :param ref_section:
        :param alt_section:
        :param ref_vcf:
        :param alt_vcf:
        :param flank_length:
        :param dna_alphabet:
        :return: pwm_dict, all_ref_idxs, all_alt_idxs, all_ref_scores, all_alt_scores
        """
        # ====[PWM MOTIF DISRUPTIONS]====
        # Combines gaussian scoring and pwm navigation to create a motif disruption score for each specific motif
        all_ref_idxs = []
        all_alt_idxs = []
        all_ref_scores = []
        all_alt_scores = []

        (sp3_count, sp3_score, sp3_cluster,
         sp3_refidxs, sp3_altidxs,
         sp3_refscores, sp3_altscores) = DNAMatrix.DNA_pwm_stats(ref_vcf, alt_vcf, flank_length,
                                                                 global_config['splice_3_pwm'],
                                                                 ref_section, alt_section, dna_alphabet,
                                                                 cluster_threshold=0.9)
        all_ref_idxs.extend(sp3_refidxs)
        all_alt_idxs.extend(sp3_altidxs)
        all_ref_scores.extend(sp3_refscores)
        all_alt_scores.extend(sp3_altscores)

        (sp5_count, sp5_score, sp5_cluster,
         sp5_refidxs, sp5_altidxs,
         sp5_refscores, sp5_altscores) = DNAMatrix.DNA_pwm_stats(ref_vcf, alt_vcf, flank_length,
                                                                 global_config['splice_5_pwm'],
                                                                 ref_section, alt_section, dna_alphabet,
                                                                 cluster_threshold=0.9)
        all_ref_idxs.extend(sp5_refidxs)
        all_alt_idxs.extend(sp5_altidxs)
        all_ref_scores.extend(sp5_refscores)
        all_alt_scores.extend(sp5_altscores)

        (branch_pt_count, branch_pt_score, branch_pt_cluster,
         branch_pt_refidxs, branch_pt_altidxs,
         branch_pt_refscores, branch_pt_altscores) = DNAMatrix.DNA_pwm_stats(ref_vcf, alt_vcf, flank_length,
                                                                             global_config['branch_pt_pwm'],
                                                                             ref_section, alt_section, dna_alphabet,
                                                                             cluster_threshold=0.9)
        all_ref_idxs.extend(branch_pt_refidxs)
        all_alt_idxs.extend(branch_pt_altidxs)
        all_ref_scores.extend(branch_pt_refscores)
        all_alt_scores.extend(branch_pt_altscores)

        (polyad_count, polyad_score, polyad_cluster,
         polyad_refidxs, polyad_altidxs,
         polyad_refscores, polyad_altscores) = DNAMatrix.DNA_pwm_stats(ref_vcf, alt_vcf, flank_length,
                                                                       global_config['polyadenylation_pwm'],
                                                                       ref_section, alt_section, dna_alphabet,
                                                                       cluster_threshold=0.9)
        all_ref_idxs.extend(polyad_refidxs)
        all_alt_idxs.extend(polyad_altidxs)
        all_ref_scores.extend(polyad_refscores)
        all_alt_scores.extend(polyad_altscores)

        # totals
        total_motif_count = sp3_count + sp5_count + branch_pt_count + polyad_count
        total_score_shift = sp3_score + sp5_score + branch_pt_score + polyad_score
        cluster_additive = sp3_cluster + sp5_cluster + branch_pt_cluster + polyad_cluster

        # now calculate domain cluster  shifts
        domain_cluster = DNAMatrix.cluster_composite_delta(all_ref_idxs, all_alt_idxs, all_ref_scores, all_alt_scores)

        # dictionary assembly
        ptr_dict = {}

        ptr_dict['sp3_count'] = sp3_count
        ptr_dict['sp3_score'] = sp3_score
        ptr_dict['sp3_cluster'] = sp3_cluster

        ptr_dict['sp5_count'] = sp5_count
        ptr_dict['sp5_score'] = sp5_score
        ptr_dict['sp5_cluster'] = sp5_cluster

        ptr_dict['branch_pt_count'] = branch_pt_count
        ptr_dict['branch_pt_score'] = branch_pt_score
        ptr_dict['branch_pt_cluster'] = branch_pt_cluster

        ptr_dict['polyad_count'] = polyad_count
        ptr_dict['polyad_score'] = polyad_score
        ptr_dict['polyad_cluster'] = polyad_cluster

        # totals
        ptr_dict['PTR_motif_count'] = total_motif_count
        ptr_dict['PTR_score_shift'] = total_score_shift
        ptr_dict['PTR_cluster_shift'] = cluster_additive
        ptr_dict['PTR_domain_cluster_shift'] = domain_cluster

        return ptr_dict, all_ref_idxs, all_alt_idxs, all_ref_scores, all_alt_scores

    @staticmethod
    def DNA_pwm_stats(ref_vcf, alt_vcf, flank_length, pwm, ref_section, alt_section, alphabet, cluster_threshold):
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
        :param cluster_threshold:
        :return: motif_quantity_delta, position_score_delta, cluster_score, filtref_idxs, filtalt_idxs, filtref_scores, filtalt_scores
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


        # === cluster composite scoring ===
        calc_threshold = DNAMatrix.get_threshold(pwm, cluster_threshold)

        filtref_idxs, filtref_scores = DNAMatrix.index_filter(ref_idxs_adjusted, ref_motif_scores, calc_threshold)
        filtalt_idxs, filtalt_scores = DNAMatrix.index_filter(alt_idxs_adjusted, alt_motif_scores, calc_threshold)

        cluster_score = DNAMatrix.cluster_composite_delta(filtref_idxs, filtalt_idxs,
                                                          filtref_scores, filtalt_scores)


        return (motif_quantity_delta, position_score_delta, cluster_score,
                filtref_idxs, filtalt_idxs, filtref_scores, filtalt_scores)


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
            idxs.append(i)
            scores.append(DNAMatrix.probability_subseq(sequence[i:i + motif_size], pwm, alphabet))
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

            if prob <= 1e-9:
                return 0  # if a subseq contains the character with 0 probability -> not possible it's the motif
                          # therefore return a score of 0

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


    @staticmethod
    def cluster_composite_delta(filtref_idxs, filtalt_idxs, filtref_scores, filtalt_scores):
        """
        Determines cluster-composite score
        :param filtref_idxs:
        :param filtalt_idxs:
        :param filtref_scores:
        :param filtalt_scores:
        :return:
        """
        return (DNAMatrix.cluster_composite_scorer(filtalt_idxs, filtalt_scores) -
                DNAMatrix.cluster_composite_scorer(filtref_idxs, filtref_scores))


    @staticmethod
    def index_filter(idxs, scores, threshold):
        filtered_idxs = []
        filtered_scores = []

        for pair in range(len(idxs)):
            if scores[pair] >= threshold:
                filtered_idxs.append(idxs[pair])
                filtered_scores.append(scores[pair])

        return filtered_idxs, filtered_scores



    @staticmethod
    def cluster_composite_scorer(idxs, scores):
        cluster_score = 0
        for pos in range(len(idxs) - 1):
            distance = idxs[pos+1] - idxs[pos]
            if distance <= 0:
                continue
            if distance <= global_config['cluster_distance']:
                cluster_score += (scores[pos] + scores[pos + 1]) / (distance + 1)

        return cluster_score
