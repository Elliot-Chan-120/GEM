# Navigation help
import yaml
from src.gem.paths import *

# math
import numpy as np
import pandas as pd

# helpers
from collections import Counter
import re
from tqdm import tqdm

# multiprocessing
import multiprocessing

# initialize biological structure libraries
from src.gem.b00_bio_library import *

# initialize biological structure libraries

global_config = {}

def config_init(config, complements, nn_free_energy,
                gene_densities, stops,
                poly_tracts, intron,
                iupac_codes, ambiguous):
    """
    Create global variables for each worker process
    :param config:
    :param complements:
    :param nn_free_energy:
    :param gene_densities:
    :param stops:
    :param poly_tracts:
    :param intron:
    :param iupac_codes:
    :param ambiguous:
    :return:
    """
    # These become GLOBAL variables in each worker process - check multiprocessing pool functions in class
    global global_config, COMPLEMENTS, NN_FREE_ENERGY, GENE_DENSITIES, STOPS, POLY_TRACTS, INTRON, IUPAC_CODES, AMBIGUOUS


    global_config = config
    COMPLEMENTS = complements
    NN_FREE_ENERGY = nn_free_energy
    GENE_DENSITIES = gene_densities
    STOPS = stops
    POLY_TRACTS = poly_tracts
    INTRON = intron
    IUPAC_CODES = iupac_codes
    AMBIGUOUS = ambiguous


class CompositeDNA:
    _pool = None

    def __init__(self, core_num=None):
        config_path = CONFIG

        with open(config_path) as outfile:
            self.cfg = yaml.safe_load(outfile)
            # config parameters
            self.k_min = self.cfg['k_min']
            self.k_max = self.cfg['k_max']
            self.match, self.mismatch = self.cfg['match'], self.cfg['mismatch']
            self.splice_site_window = self.cfg['splice_site_window']

            self.min_repeat_length, self.max_repeat_length = self.cfg['min_repeat_length'], self.cfg[
                'max_repeat_length']
            self.min_repeats = self.cfg['min_repeats']

            # DNA submat
            self.sub_matrix = None
            self.submatrix()

            # buffer region size
            self.structural_buffer = self.cfg['DNA_buffer_region']

            # hmm search window
            self.hmm_window = self.cfg['hmm_window']

            self.initial_probs = {
                "3UTR": 0.075,
                "5UTR": 0.075,
                "CDS": 0.015,
                "Exon": 0.03,
                "Intergenic": 0.67,
                "Intron": 0.27,
            }

            self.emission_probs = {
                "3UTR": {
                    "A": 0.250000,
                    "C": 0.250000,
                    "G": 0.250000,
                    "T": 0.250000,
                },
                "5UTR": {
                    "A": 0.250000,
                    "C": 0.250000,
                    "G": 0.250000,
                    "T": 0.250000,
                },
                "CDS": {
                    "A": 0.268820,
                    "C": 0.253455,
                    "G": 0.258194,
                    "T": 0.219532,
                },
                "Exon": {
                    "A": 0.270445,
                    "C": 0.233545,
                    "G": 0.237919,
                    "T": 0.258092,
                },
                "Intergenic": {
                    "A": 0.298139,
                    "C": 0.200534,
                    "G": 0.201782,
                    "T": 0.299545,
                },
                "Intron": {
                    "A": 0.280442,
                    "C": 0.200074,
                    "G": 0.209608,
                    "T": 0.309875,
                },
            }

            self.transition_probs = {
                "3UTR": {
                    "3UTR": 0.166667,
                    "5UTR": 0.166667,
                    "CDS": 0.166667,
                    "Exon": 0.166667,
                    "Intergenic": 0.166667,
                    "Intron": 0.166667,
                },
                "5UTR": {
                    "3UTR": 0.166667,
                    "5UTR": 0.166667,
                    "CDS": 0.166667,
                    "Exon": 0.166667,
                    "Intergenic": 0.166667,
                    "Intron": 0.166667,
                },
                "CDS": {
                    "3UTR": 0.000000,
                    "5UTR": 0.000000,
                    "CDS": 0.142494,
                    "Exon": 0.000000,
                    "Intergenic": 0.000000,
                    "Intron": 0.857506,
                },
                "Exon": {
                    "3UTR": 0.000000,
                    "5UTR": 0.000000,
                    "CDS": 0.836724,
                    "Exon": 0.000000,
                    "Intergenic": 0.000000,
                    "Intron": 0.163276,
                },
                "Intergenic": {
                    "3UTR": 0.166667,
                    "5UTR": 0.166667,
                    "CDS": 0.166667,
                    "Exon": 0.166667,
                    "Intergenic": 0.166667,
                    "Intron": 0.166667,
                },
                "Intron": {
                    "3UTR": 0.000000,
                    "5UTR": 0.000000,
                    "CDS": 0.000000,
                    "Exon": 1.000000,
                    "Intergenic": 0.000000,
                    "Intron": 0.000000,
                },
            }

            self.states = ['3UTR', '5UTR', 'CDS', 'Exon', 'Intergenic', 'Intron']

            # initialize starting transition state dictionary for future tracking
            self.transstate_dict = {}
            for state in self.states:
                for next_state in self.states:
                    if state != next_state:
                        self.transstate_dict.update({f"{state}_to_{next_state}": 0})



        # create config for downstream multiproc
        self.config = {
            'k_min': self.k_min,
            'k_max': self.k_max,
            'match': self.match,
            'mismatch': self.mismatch,
            'min_repeat_length': self.min_repeat_length,
            'max_repeat_length': self.max_repeat_length,
            'min_repeats': self.min_repeats,
            'structural_buffer': self.structural_buffer,
            'sub_matrix': self.sub_matrix,

            # HMM parameters
            'hmm_window': self.hmm_window,
            'init_probs': self.initial_probs,
            'emit_probs': self.emission_probs,
            'trans_probs': self.transition_probs,
            'states': self.states,
            'transstate_dict': self.transstate_dict,
        }

        # define number of cores
        self.core_num = core_num if core_num is not None else multiprocessing.cpu_count() - 2

        # initialize pool if not already initialized
        if CompositeDNA._pool is None:
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
                initargs=(self.config, COMPLEMENTS,
                          NN_FREE_ENERGY, GENE_DENSITIES, STOPS,
                          POLY_TRACTS, INTRON,
                          IUPAC_CODES, AMBIGUOUS)
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


    # ====[[MUTATION FINGERPRINT DATAFRAME]]====
    def gen_DNAfp_dataframe(self, dataframe):
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
            # pool.imap method applies function self.DNA_fp_wrapper to each row in fingerprint_rows
            CompositeDNA._pool.imap(self.DNA_fingerprinter_wrapper, fingerprint_rows),
            total=len(fingerprint_rows),
            desc="[Generating DNA mutation fingerprints]"
        ))

        fingerprint_df = pd.DataFrame(fingerprint_rows)
        fingerprint_df = pd.concat([dataframe.reset_index(drop=True), fingerprint_df],
                                   axis=1)  # axis = 1 to concatenate column wise (side by side)

        fingerprint_df = fingerprint_df.drop(['ReferenceAlleleVCF', 'AlternateAlleleVCF', 'Flank_1', 'Flank_2'],
                                             axis=1)

        return fingerprint_df

    def DNA_fingerprinter_wrapper(self, fp_row):
        """
        multiprocessing wrapper, processes a single row
        :param DNA dataframe fp_row:
        :return:
        """
        chromosome, ref_allele, alt_allele, flank_1, flank_2 = fp_row
        config = global_config

        # Organize sequence dimensions
        # Full-length alleles (context-inclusive)
        ref_full = flank_1 + ref_allele + flank_2
        alt_full = flank_1 + alt_allele + flank_2
        # Buffered alleles (for specific mutations)
        ref_buffered = flank_1[len(flank_1) - config['structural_buffer']:] + ref_allele + flank_2[:config[
            'structural_buffer']]
        alt_buffered = flank_1[len(flank_1) - config['structural_buffer']:] + alt_allele + flank_2[:config[
            'structural_buffer']]

        fp = {}
        fp.update(self.base_profile(chromosome, ref_allele, alt_allele, config))
        fp.update(self.structural_profile(ref_full, alt_full, ref_buffered, alt_buffered, config))

        # general stability
        fp['FI_General_Stability'] = (fp['FI_Relative_stability'] * fp['FI_Mutation_propensity']) / fp[
            'Chromosome_density']

        # mutation composite
        fp["FI_Mutation_composite"] = (
            fp["Tandem_Duplications"]
            + fp["Hairpins"]
            + fp["tr_base_composite"]
            + fp["Poly_tracts"]
            + fp["SNVs"]
            + fp["Insertion_length"]
        )

        return fp


    def submatrix(self):
        """
        :return: substitution matrix for DNA
        """
        sub_matrix = {}
        alphabet = "ATCGRYSWKMBDHVN"
        match = self.cfg['match']
        mismatch = self.cfg['mismatch']

        for char_1 in alphabet:
            for char_2 in alphabet:
                if char_1 == char_2:  # if it matches - score set to match value and vice versa
                    sub_matrix[char_1 + char_2] = match
                else:
                    sub_matrix[char_1 + char_2] = mismatch

        self.sub_matrix = sub_matrix
        return

    # =====[[[DATA GATHERING TOOLS]]]=====
    def base_profile(self, chromosome, ref_allele, alt_allele, config):
        """
        Generates a numerical fingerprint-like data structure encapsulating local DNA variant characteristics
        - alignment, structural, and mutation data are gathered and organized into a baseline profile
        :param chromosome:
        :param ref_allele:
        :param alt_allele:
        :param config:
        """
        # First get data for downstream work - alignment data and kmer_frequencies
        global_score, global_pct, global1, global2 = self.needleman_data(ref_allele, alt_allele, config)
        local_score, local_pct = self.smith_waterman_data(ref_allele, alt_allele, config)
        ref_kmer_frequencies = self.kmer_freqs(ref_allele)
        alt_kmer_frequencies = self.kmer_freqs(alt_allele)

        base_fp = {
            # 1 - Alignment profile
            'Global_score': global_score,
            'Local_score': local_score,
            'Global_pct': global_pct,
            'Local_pct': local_pct,

            # 2 - Structural profile
            # Chromosome density
            # Length_change context - accounts for frameshifts by modulo 3
            # Length_change_pct / chromosome density
            # GC_context_delta
            # Thermodynamic_delta
            # Shannon Entropy
            'Chromosome_density': self.chromosome_density(chromosome),
            'Length_change_context': self.len_change_context_score(ref_allele, alt_allele),
            'GC_context_delta': self.GC_context(ref_allele, alt_allele),
            'Thermodynamic_stability': self.thermodynamic_delta(ref_allele, alt_allele),
            'Shannon_entropy': self.shannon_entropy_delta(ref_kmer_frequencies, alt_kmer_frequencies),

            # 3 - Simple Mutation Profile detection of mutations + characteristics detectable through alignment data
            # SNVs
            # Insertion Lengths - deletions did not provide much importance hence why i left it out
            'SNVs': self.snv(global1, global2),
            'Insertion_length': self.insertion_len(global1, global2)
        }

        # 4 - FEATURE ENGINEERING
        base_fp['FI_Relative_length_context'] = base_fp['Length_change_context'] / \
                                                            base_fp['Chromosome_density']
        base_fp['FI_Relative_stability'] = base_fp['Thermodynamic_stability'] * \
                                                       base_fp['Chromosome_density']
        # maybe I'll add more feature interactions from previous iterations but these were the most important ones, the others had less than 1%

        return base_fp

    def structural_profile(self, full_ref_allele, full_alt_allele, ref_buffered, alt_buffered, config):
        """
        Generates a numerical fingerprint-like data structure quantifying structural changes within a large sequence window, with the local variants at the center
        - structural data changes, more complex mutations, and the loss / gain of specific motifs are recorded
        :param full_ref_allele:
        :param full_alt_allele:
        :param ref_buffered:
        :param alt_buffered:
        :param config:
        """
        ref_kmer_freqs = self.kmer_freqs(full_ref_allele)
        alt_kmer_freqs = self.kmer_freqs(full_alt_allele)

        # GC content total delta
        # Total thermodynamic delta
        # Regex motif disruptions

        # Duplications
        # Tandem Duplications
        # Microsatellites

        flanked_fp = {
            # 1 - Mutations / structure changes
            'Tandem_Duplications': self.duplication_delta(ref_buffered, alt_buffered, config),
            'Hairpins': self.hairpin_count_delta(ref_buffered, alt_buffered),

            # 2 - start and stop codons gain / loss
            'Start_codons': self.start_delta(ref_buffered, alt_buffered),
            'Stop_codons': self.stop_delta(ref_buffered, alt_buffered),

            # 3 - intron changes
            'Introns': self.intron_delta(ref_buffered, alt_buffered),

            # 4 - transcription factor and instability motifs
            'Poly_tracts': self.poly_tract_delta(ref_buffered, alt_buffered)
        }

        flanked_fp.update(self.repeat_instab(ref_buffered, alt_buffered, ref_kmer_freqs, alt_kmer_freqs, config))

        # 2 Feature Interactions
        flanked_fp["FI_Mutation_propensity"] = (
            flanked_fp["Tandem_Duplications"]
            + flanked_fp["Hairpins"]
            + flanked_fp["tr_base_composite"]
            + flanked_fp["Poly_tracts"]
        )
        return flanked_fp

    # =====[[ALIGNMENT DATA]]=====
    # pairwise sequence alignments comparing the original and mutated -> sequence identities
    # prepare alignment variables and substitution matrix
    # [ALIGNMENT MATRIX GENERATION]
    def global_alignment(self, original, mutation, config):
        """
        performs the needleman_wunsch algorithm to produce a global scoring and traceback matrix
        :returns: score_matrix - [0], traceback_matrix - [1]
        """
        glb_score_matrix = [[0]]  # scoring matrix - contains best alignment scores for each pos
        glb_traceback_matrix = [[0]]  # traceback matrix - records moves that gave the best scores

        # initialize gap row (all gaps in seq 1)
        for col in range(1, len(mutation) + 1):
            glb_score_matrix[0].append(config['mismatch'] * col)
            glb_traceback_matrix[0].append(3)

        # initialize gap column (all gaps in seq 2)
        for row in range(1, len(original) + 1):
            glb_score_matrix.append([config['mismatch'] * row])
            glb_traceback_matrix.append([2])

        # apply recurrence relation to fill remaining of matrix
        for row in range(len(original)):
            for col in range(len(mutation)):
                # calculate scores for 3 possible moves:
                diagonal_score = glb_score_matrix[row][col] + self.score_pos(original[row],
                                                                                  mutation[col], config)  # Diagonal
                up_score = glb_score_matrix[row][col + 1] + config['mismatch']  # Up
                left_score = glb_score_matrix[row + 1][col] + config['mismatch']  # Left
                # choose best score and move
                glb_score_matrix[row + 1].append(max(diagonal_score, up_score, left_score))
                glb_traceback_matrix[row + 1].append(
                    self.argmax_of_three(diagonal_score, up_score,
                                         left_score))  # record which move it was using max3t function
        return glb_score_matrix, glb_traceback_matrix

    def local_alignment(self, original, mutation, config):
        """
        performs the smith-waterman algorithm to produce a local scoring and traceback matrix
        :returns: score_matrix - [0], traceback_matrix - [1], max_score - [2]
        """
        lcl_score_matrix = [[0]]
        lcl_traceback_matrix = [[0]]
        maxscore = 0

        for col in range(1, len(mutation) + 1):
            lcl_score_matrix[0].append(0)
            lcl_traceback_matrix[0].append(0)

        for row in range(1, len(original) + 1):
            lcl_score_matrix.append([0])
            lcl_traceback_matrix.append([0])

        for row in range(len(original)):
            for col in range(len(mutation)):
                # calculate the scores for 3 possible moves
                diagonal_score = lcl_score_matrix[row][col] + self.score_pos(original[row], mutation[col], config)
                up_score = lcl_score_matrix[row][col + 1] + config['mismatch']
                left_score = lcl_score_matrix[row + 1][col] + config['mismatch']
                best_score = max(diagonal_score, up_score, left_score)
                if best_score <= 0:  # calculate and see if best score is greater than 0
                    lcl_score_matrix[row + 1].append(0)
                    lcl_traceback_matrix[row + 1].append(
                        0)  # If everything is smaller than 0, then 0 is used as the score instead
                else:
                    lcl_score_matrix[row + 1].append(best_score)  # Otherwise, best score will be used in matrices
                    lcl_traceback_matrix[row + 1].append(
                        self.argmax_of_three(diagonal_score, up_score, left_score))
                    if best_score > maxscore:
                        maxscore = best_score
        return lcl_score_matrix, lcl_traceback_matrix, maxscore

    # [ALIGNMENT HELPER FUNCTIONS]
    @staticmethod
    def score_pos(char_1, char_2, config):
        """
        Scores a single position in alignment, if any character is a gap -> gap penalty
        otherwise -> look up the score for this pair of nucleotides
        """
        if char_1 == '-' or char_2 == '-':
            return config['mismatch']
        else:
            return config['sub_matrix'][char_1 + char_2]  # look up the score in submat unless a gap is there

    @staticmethod
    def argmax_of_three(diagonal, up, left):
        """
        Returns 1,2,3 depending on which one of the scores passed to it were the largest
        Will help us reconstruct the alignment and understand where the optimal path alignment is
        """
        if diagonal > up:
            if diagonal > left:
                return 1
            else:
                return 3
        else:
            if up > left:
                return 2
            else:
                return 3

    @staticmethod
    def max_mat(mat):
        """
        Obtain maximum value from matrix
        """
        maxval = mat[0][0]
        maxrow = 0
        maxcol = 0
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                if mat[i][j] > maxval:
                    maxval = mat[i][j]
                    maxrow = i
                    maxcol = j
        return maxrow, maxcol

    def score_align(self, seq1, seq2, config):
        """
        Calculates total score for entire alignment by summing them for each position
        """
        alignment_score = 0
        for i in range(len(seq1)):
            alignment_score += self.score_pos(seq1[i], seq2[i], config)
        return alignment_score

    @staticmethod
    def percent_identity(seq1, seq2):
        """
        Calculates the percentage to which these two sequences align
        :param seq1:
        :param seq2:
        :return: % identity (float) between the two sequences
        """
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != '-')
        return (matches / len(seq1)) * 100

    @staticmethod
    def coverage(sub_region, original):
        """
        Determines how much of the best-aligned subregion is in the original
        :param sub_region:
        :param original:
        :return: % coverage (float) between the original and local alignment
        """
        return (len(sub_region) / len(original)) * 100

    # [ALIGNMENT EXTRACTION]
    @staticmethod
    def global_align(original, mutation, glb_traceback_matrix):
        """
        Exracts the aligned sequences from the needleman-wunsch traceback matrix
        :param original:
        :param mutation:
        :param glb_traceback_matrix:
        :return: [sequence1, sequence2] in global alignment
        """
        aligned_seqs = ["", ""]  # Will hold two aligned sequences
        row = len(original)  # Start at bottom right
        col = len(mutation)

        while row > 0 or col > 0:
            if glb_traceback_matrix[row][col] == 1:  # Diagonal move
                aligned_seqs[0] = original[row - 1] + aligned_seqs[0]  # Match/Mismatch two amino acids
                aligned_seqs[1] = mutation[col - 1] + aligned_seqs[1]
                row -= 1
                col -= 1
            elif glb_traceback_matrix[row][col] == 3:  # Left move
                aligned_seqs[0] = "-" + aligned_seqs[0]  # Insert gap in first sequence
                aligned_seqs[1] = mutation[col - 1] + aligned_seqs[1]
                col -= 1
            else:  # Up move
                aligned_seqs[0] = original[row - 1] + aligned_seqs[0]  # Insert gap in second sequence
                aligned_seqs[1] = "-" + aligned_seqs[1]
                row -= 1
        return aligned_seqs

    def local_align(self, scoring_mat, original, mutation, lcl_traceback_matrix):
        """
        Utilizes the scoring and traceback matrix from smith-waterman to extract the aligned local region
        :param scoring_mat:
        :param original:
        :param mutation:
        :param lcl_traceback_matrix
        :return: [sequence1, sequence2] in local alignment
        """
        aligned_seqs = ["", ""]
        current_row, current_col = self.max_mat(scoring_mat)  # start at the highest score

        while lcl_traceback_matrix[current_row][current_col] > 0:  # stop when 0 is reached
            move = lcl_traceback_matrix[current_row][current_col]
            if move == 1:  # diagonal move
                aligned_seqs[0] = original[current_row - 1] + aligned_seqs[0]
                aligned_seqs[1] = mutation[current_col - 1] + aligned_seqs[1]
                current_row -= 1
                current_col -= 1
            elif move == 3:  # left move -> insert gap in the first seq
                aligned_seqs[0] = "-" + aligned_seqs[0]
                aligned_seqs[1] = mutation[current_col - 1] + aligned_seqs[1]
                current_col -= 1
            elif move == 2:  # up move -> insert gap in the second seq
                aligned_seqs[0] = original[current_row - 1] + aligned_seqs[0]
                aligned_seqs[1] = "-" + aligned_seqs[1]
                current_row -= 1
        return aligned_seqs

    # ====[[MASTER ALIGNMENT DATA ACQUISITION: CALL ON THIS]]====
    # --> will output: global alignment score, local alignment score, global % identity, local % identity
    # --> global1 and global2 alignments
    def needleman_data(self, loc_original, loc_mutation, config):
        """
        Local spec. - Carries out needleman-wunsch alignments on the two strings
        :param loc_original: local region - ref_allele
        :param loc_mutation: local region - alt_allele
        :param config
        :return: global alignment score [0], global % identity [1], globally aligned original and mutation strings [2, 3]
        """
        # get aligned DNA strings
        glb_score_matrix, glb_traceback_matrix = self.global_alignment(loc_original, loc_mutation, config)
        global1, global2 = self.global_align(loc_original, loc_mutation, glb_traceback_matrix)

        if len(global1) != len(global2):
            raise ValueError("Sequence Alignment malfunction, aligned sequences are not of equal length.")

        # get raw alignment scores
        global_score = self.score_align(global1, global2, config)
        # get % identity
        global_per = self.percent_identity(global1, global2)

        return global_score, global_per, global1, global2

    def smith_waterman_data(self, loc_original, loc_mutation, config):
        """
        Local spec. - Carries out smith-waterman alignments on the two strings
        :param loc_original: local region - ref_allele
        :param loc_mutation: local region - alt_allele
        :param config:
        :return: local alignment score [0], local % identity [1]
        """
        # get aligned DNA strings
        lcl_score_matrix, lcl_traceback_matrix, max_score = self.local_alignment(loc_original, loc_mutation, config)
        local1, local2 = self.local_align(lcl_score_matrix, loc_original, loc_mutation, lcl_traceback_matrix)

        if len(local1) != len(local2):
            raise ValueError("Sequence Alignment Malfunction, aligned sequences are not of equal length")

        # raw alignment scores and % identity
        local_score = self.score_align(local1, local2, config)
        local_per = self.coverage(local1, loc_original)

        return local_score, local_per

    # ====[STRUCTURAL DATA]====
    # information relating to the overall structure of the DNA sequences
    # can tell us about their stability, complexity .etc

    @staticmethod
    def len_change_context_score(loc_original, loc_mutation):
        """
        Local spec. - Combines percentage change in length with frameshift changes
        :param loc_original: local region - ref_allele
        :param loc_mutation: local region - alt_allele
        :return: length change impact on context - may disrupt protein sequences
        """
        # percentage difference
        len_ref = len(loc_original)
        len_alt = len(loc_mutation)
        pct_change = (len_alt - len_ref) / len_ref * 100

        # frameshift penalty

        if len_ref != len_alt:
            frameshift_penalty = 2.0 if abs(len_ref - len_alt) % 3 != 0 else 1.0
        else:
            frameshift_penalty = 1.0

        return pct_change * frameshift_penalty

    @staticmethod
    def chromosome_density(chromosome):
        """
        Universal - Give the chromosome's density
        :param chromosome: chromosome number, X is 23, Y is 24
        :return: chromosome density in genes/Mb
        """
        # these are in genes/Mb
        return GENE_DENSITIES[chromosome]

    def GC_context(self, univ_original, univ_mutation):
        """
        Universal - Determines the change in GC content and CpG island disruption across sequences
        :param univ_original: can be either the ref_allele fully flanked or variant region
        :param univ_mutation: can be either the alt_allele fully flanked or variant region
        :return: gc delta + cpg island delta
        """
        muta_gc, muta_cpg = self.gc_content(univ_mutation), self.cpg_content(univ_mutation)
        orig_gc, orig_cpg = self.cpg_content(univ_original), self.cpg_content(univ_original)

        return (muta_gc - orig_gc) + (muta_cpg - orig_cpg)

    @staticmethod
    def gc_content(sequence):
        return (sequence.count('G') + sequence.count('C')) / len(sequence)

    @staticmethod
    def cpg_content(sequence):
        return sequence.count('CG') / max(len(sequence) - 1, 1)

    def thermodynamic_delta(self, univ_original, univ_mutation):
        """
        Universal - Utilizing precomputed thermodynamic stability parameters, calculates the change in free energy from original to mutation DNA strings
        Precomputed values involving ambiguous cases were calculated from averages across all possible nucleotide combinations
        :param univ_original: can be either the ref_allele fully flanked or variant region
        :param univ_mutation: can be either the alt_allele fully flanked or variant region
        :return: the change in thermodynamic stability

        """
        # Calculate direct stability changes between the original-reverse, and mutation-reverse
        # scoring in b00_bio_library.py

        if len(univ_original) <= 1:  # set thermodynamic values for single nucleotides to 0
            original_thermo = 0
        else:
            original_thermo = self.thermodynamic_score(univ_original)

        if len(univ_mutation) <= 1:
            mutation_thermo = 0
        else:
            mutation_thermo = self.thermodynamic_score(univ_mutation)

        return mutation_thermo - original_thermo

    @staticmethod
    def thermodynamic_score(sequence):
        """
        Calls on thermodynamic parameters, ambiguous character combinations are precomputed averages of all possible nucleotide combinations
        - check NN FREE ENERGY in b00_bio_library.py
        :param sequence:
        :return: total energy the sequence releases upon duplex formation
        """
        total_energy = 0.0

        # finds the score for each 2 nucleotide subsequence and finds the average for sequences
        for i in range(len(sequence) - 1):
            dinuc_extract = sequence[i:i + 2]
            total_energy += NN_FREE_ENERGY[dinuc_extract]

        return total_energy

    def shannon_entropy_delta(self, univ_ori_kmer_freqs, univ_muta_kmer_freqs):
        """
        Universal - Return the change in shannon entropy between the two sequences
        # this is going to be from complex_stats [3]
        Shannon entropy is a measure of information, basically characterizing sequence complexity
        :param univ_ori_kmer_freqs: kmer_frequencies of either the full or local variant string
        :param univ_muta_kmer_freqs: kmer_frequencies of either the full or local variant string
        :return: change in shannon entropy due to mutation
        """
        return self.shannon_entropy(univ_muta_kmer_freqs) - self.shannon_entropy(univ_ori_kmer_freqs)

    @staticmethod
    def shannon_entropy(seq_kmer_frequencies):
        """
        Use the shannon entropy formula on the kmer frequency dict
        Must input kmer frequencies from kmer_freqs(sequence)
        :param seq_kmer_frequencies to calculate entropy for
        :return: shannon entropy
        """
        if not seq_kmer_frequencies:
            return 0.0
        else:
            total = np.sum(list(seq_kmer_frequencies.values()))
            entropy = 0
            for freq in seq_kmer_frequencies.values():
                p = freq / total
                entropy -= p * np.log2(p)

        return float(entropy)

    @staticmethod
    def kmer_freqs(sequence):
        """
        kmer frequencies -> dict of the kmers (keys) and their respective frequencies (values)
        shannon_entropy_delta and microsatellite delta need these for their respective sequences
        sequence complexity via shannon entropy calculated downstream using data from this
        :param sequence:

        :return: kmer_frequencies
        """
        kmer_frequencies = Counter()
        seq_len = len(sequence)

        # single window pass, calculate kmer frequencies
        for i in range(seq_len):
            for k in range(global_config['k_min'], min(global_config['k_max'] + 1, seq_len - i + 1)):
                kmer = sequence[i:i + k]
                kmer_frequencies[kmer] += 1

        if not kmer_frequencies:
            return None

        return kmer_frequencies

    # ====[MUTATIONS]====
    # Detects secondary structures and specific mutations
    @staticmethod
    def snv(global1, global2):
        """NW Alignments - Returns 1 if it detects an SNV mutation"""
        if len(global1) != len(global2):
            return None

        mismatches = sum(
            1 for i, j in zip(global1, global2) if i != '-' and j != '-' and i != j
        )

        gaps = sum(
            1 for i, j in zip(global1, global2) if i == '-' or j == '-'
        )

        return int(mismatches == 1 and gaps == 0)

    @staticmethod
    def insertion_len(global1, global2):
        """NW Alignments - Returns 1 if it detects insertion mutations"""
        insertions = 0
        for a, b in zip(global1, global2):
            if a == '-' and b != '-':
                insertions += 1

        return insertions

    # duplications
    @staticmethod
    def duplication_delta(br_original, br_mutation, config):
        """
        Global - Returns the number of duplicated regions detected
        Detects a repeated and adjacent sequence - single pass
        :param br_original: buffer region of ref_allele
        :param br_mutation: buffer region of alt_allele
        :param config:
        :return:
        """
        duplication_count = 0

        # moving boundary , detect if substring lies on other side of the boundary
        min_length = config['min_repeat_length']  # 2 default
        max_length = min(config['max_repeat_length'], len(br_mutation) // 2)  # arbitrary cap

        seen_duplications = set()

        original_kmers = {}

        for i in range(len(br_mutation) - 2 * min_length + 1):
            # this is the range of possible duplication lengths
            # the largest length is the minimum between the max_length and the remaining DNA string length
            # as position i, we need room for two copies of a k-mer
            remaining_length = len(br_mutation) - i
            real_max = min(max_length, remaining_length // 2)

            for k in range(min_length, real_max + 1):
                subseq = br_mutation[i:i + k]
                subseq2 = br_mutation[i + k: i + 2 * k]

                if subseq == subseq2 and subseq not in seen_duplications:
                    if k not in original_kmers:
                        original_kmers[k] = set(br_original[j:j + k] for j in range(len(br_original) - k + 1))

                    if subseq not in original_kmers[k]:
                        duplication_count += 1
                        seen_duplications.add(subseq)

        return duplication_count

    # hairpin propensities - 1 nucleotide of error
    def hairpin_count_delta(self, br_original, br_mutation):
        """
        Buffered region spec. - Detects the difference in hairpin structure formation. 1 mismatch is allowed to retain some biological relevancy
        :param br_original: buffer region of ref_allele
        :param br_mutation: buffer region of alt_allele
        :return: the change in hairpin structure count
        """
        return self.hairpin_count(br_mutation) - self.hairpin_count(br_original)

    @staticmethod
    def hairpin_count(sequence):
        """
        potential hairpin structures typically look like [stem1] [loop] [stem2]
        where stem1 and 2 are reverse complements, and the loop is an unpaired section generally of 3-8 bases
        :param sequence:
        :return: number of times a hairpin was detected
        """
        # minimum hairpin dimensions: 4bp stem, 3bp loop, 4bp stem == 11bps
        if len(sequence) < 11:
            return 0

        min_stem_l = 4  # minimum stem length is 4
        min_loop_l = 3  # hairpin loops are generally between 3-6
        hairpin_count = 0

        # precompute reverse complement for to check against easier
        complement = COMPLEMENTS

        # use sliding window approach
        # this will iterate starting from 0 to the remainder of the sequence...
        # not involving the two potential stems and loop at minimum lengths
        # i.e. if len(seq) == 11 then it only goes once because...
        # that's the only window we can possibly scan accounting for the minimum loop and stem sizes
        for i in range(len(sequence) - 2 * min_stem_l - min_loop_l):
            stem_1 = sequence[i:i + min_stem_l]

            # check if stem 1 has complement downstream
            for j in range(i + min_stem_l + min_loop_l, len(sequence) - min_stem_l + 1):
                stem_2 = sequence[j:j + min_stem_l]
                stem_1_rc = stem_1[::-1].translate(complement)
                # generate reverse complement of stem 1 so we can compare how well it aligns with stem 2

                # compare reverse complement of stem 1 with stem 2 - allow for some mismatch
                # retains some biological relevance - hairpins contain errors
                mismatch_count = sum(1 for a, b in zip(stem_1_rc, stem_2[::-1]) if a != b)

                if mismatch_count <= 2:
                    hairpin_count += 1
                    break

        return hairpin_count

    # Repeat Instability
    def repeat_instab(self, br_original, br_mutation, glb_ori_kmer_freqs, glb_muta_kmer_freqs, config):
        """
        Global sequences - Needs kmer frequencies from kmer_stats
        :param br_original:
        :param br_mutation:
        :param glb_ori_kmer_freqs:
        :param glb_muta_kmer_freqs:
        :param config:
        :return: repeat instability profile -> add this as a dictionary
        """
        if not glb_ori_kmer_freqs or not glb_muta_kmer_freqs:
            return 0, 0, 0, 0, 0

        k_min, k_max = config['k_min'], config['k_max']
        orig_candidates = [kmer for kmer, freq in glb_ori_kmer_freqs.items()
                           if k_min <= len(kmer) < k_max and freq >= config['min_repeats']]

        muta_candidates = [kmer for kmer, freq in glb_muta_kmer_freqs.items()
                           if k_min <= len(kmer) < k_max and freq >= config['min_repeats']]

        # call on helper functions to determine if any of the kmers belong to a microsatellite
        orig_sats = self.tandem_repeat(orig_candidates, br_original, config)
        muta_sats = self.tandem_repeat(muta_candidates, br_mutation, config)

        # process both dictionaries
        orig_trs = set(orig_sats.keys())
        muta_trs = set(muta_sats.keys())
        repeats_lost = orig_trs - muta_trs
        repeats_gain = muta_trs - orig_trs
        microsat_delta = len(muta_trs) - len(orig_trs)

        composite = 0

        for tr in repeats_lost:
            composite += sum(orig_sats[tr])

        for tr in repeats_gain:
            composite += sum(muta_sats[tr])

        shared_trs = orig_trs & muta_trs
        for tr in shared_trs:
            composite += sum(muta_sats[tr]) - sum(orig_sats[tr])

        # in case something goes wrong, uncomment this
        # print(f"len_repeats_lost: {len(repeats_lost)}")
        # print(f"len_repeats_gain: {len(repeats_gain)}")
        # print(f"microsat_delta: {microsat_delta}")
        # print(f"composite: {composite}")
        # print(f"repeat_instab: {repeat_instab}")

        microsat_dict = {
            "tr_base_composite": composite,
            "Microsat_delta": microsat_delta,
        }

        microsat_dict.update(self.repeat_instab_data(repeats_lost, repeats_gain, shared_trs, orig_sats, muta_sats))

        return microsat_dict

    # helper function for microsatellite detection
    @staticmethod
    def tandem_repeat(candidates, sequence, config):
        """Figures out microsatellite regions"""

        # this is saying for kmer 'x' occuring 'min_repeats'+ times, add it to patterns
        # chose to use re.escape because of user-defined strings later on - may accidentally insert a character

        patterns = {
            # repeats previous group / token min number of times
            kmer: re.compile(f"({re.escape(kmer)}){{{config['min_repeats']},}}")
            for kmer in candidates
        }

        satellites = {}
        for kmer, pattern in patterns.items():
            occurrences = list(re.finditer(pattern, sequence)) # makes the next line easier to read
            if occurrences:
                satellites.update({kmer: [len(item.group()) for item in occurrences]})

        # so after all of this satellites should look like:
        # {"CGGGA": [20, 21, 39, 80]}

        return satellites

    @staticmethod
    def repeat_instab_data(repeats_lost, repeats_gain, shared_trs, orig_sats, muta_sats):
        """
        Repeat instability should reflect how much the repeat landscape has changed in a biologically meaningful way
        \ncomplete loss / gain of repeat loci are weighted highest -> multiplied by their length
        \ntotal bp changes for lost/gained loci
        \nmeasure expansion or contraction within shared repeats
        \nweigh by repeat unit size
        :return:
        """
        # loci changes
        loci_lost = len(repeats_lost)
        loci_gain = len(repeats_gain)

        # total bp changes
        bp_lost = sum(sum(orig_sats[tr]) for tr in repeats_lost)
        bp_gain = sum(sum(muta_sats[tr]) for tr in repeats_gain)

        # weighted data by repeat unit type
        wbp_lost = sum(sum(orig_sats[tr]) * len(tr) for tr in repeats_lost)
        wbp_gain = sum(sum(muta_sats[tr]) * len(tr) for tr in repeats_gain)

        # expansion and contraction scores
        expansion_score = 0
        contraction_score = 0

        w_expansion_score = 0
        w_contraction_score = 0

        for tr in shared_trs:
            orig_total_bp = sum(orig_sats[tr])
            muta_total_bp = sum(muta_sats[tr])
            bp_delta = muta_total_bp - orig_total_bp

            if bp_delta > 0:
                expansion_score += bp_delta
                w_expansion_score += bp_delta * len(tr)
            else:
                contraction_score += abs(bp_delta)
                w_contraction_score += abs(bp_delta) * len(tr)

        # get other scores ready
        simple_score = bp_lost + bp_gain + expansion_score + contraction_score
        w_simple_score = wbp_lost + wbp_gain + w_expansion_score + w_contraction_score

        composite_score = (loci_lost + loci_gain) + w_simple_score

        return {
            'tr_loci_lost': loci_lost,
            'tr_loci_gain': loci_gain,
            'tr_bp_gain': bp_gain,
            'tr_bp_loss': bp_lost,
            'tr_expansion_score': expansion_score,
            'tr_contraction_score': contraction_score,

            # weighted scores
            'tr_weighted_bploss': wbp_lost,
            'tr_weighted_bpgain': wbp_gain,
            'tr_weighted_expansion_score': w_expansion_score,
            'tr_weighted_contraction_score': w_contraction_score,

            # scores
            # 'tr_simple_score': simple_score,  # test run without these scores
            # 'tr_weighted_score': w_simple_score,
            'tr_composite_score': composite_score,
        }





    def start_delta(self, br_original, br_mutation):
        """
        Global - detect # of changes in start codon ATG
        :param br_original: buffer region of ref_allele
        :param br_mutation: buffer region of alt_allele
        :return: net change in ATG codons
        """
        return self.start_codons(br_mutation) - self.start_codons(br_original)

    @staticmethod
    def start_codons(sequence):
        """Counts number of ATG codons found in one given"""
        return len(re.findall(r'ATG', sequence))

    def stop_delta(self, br_original, br_mutation):
        """
        Global - detect # changes in stop codons TAA, TAG, TGA
        :param br_original: buffer region of ref_allele
        :param br_mutation: buffer region of alt_allele
        :return:
        """
        return self.stop_codons(br_mutation) - self.stop_codons(br_original)

    @staticmethod
    def stop_codons(sequence):
        """Counts number of ATG codons found in one given"""
        stop_codons = '|'.join(STOPS)
        return len(re.findall(stop_codons, sequence))

    # regex-search poly-tracts
    def poly_tract_delta(self, br_original, br_mutation):
        return self.poly_tract_count(br_mutation) - self.poly_tract_count(br_original)

    @staticmethod
    def poly_tract_count(sequence):
        tracts = '|'.join(POLY_TRACTS)
        return len(re.findall(tracts, sequence))

    # intron finder
    def intron_delta(self, br_original, br_mutation):
        return self.find_introns(br_mutation) - self.find_introns(br_original)

    @staticmethod
    def find_introns(sequence):
        intron_count = 0
        result = len(re.findall(INTRON, sequence))
        if result is not None:
            intron_count = result
        return intron_count


# [[[WORK IN PROGRESS]]]
    def gen_HMM_dataframe(self, dataframe):
        fingerprint_rows = [
            (row['ReferenceAlleleVCF'], row['AlternateAlleleVCF'],
             row['Flank_1'], row['Flank_2'])
            for _, row in dataframe.iterrows()
        ]

        fingerprint_rows = list(tqdm(
            CompositeDNA._pool.imap(self.hmm_wrapper, fingerprint_rows),
            total=len(fingerprint_rows),
            desc="[Extracting broad domain changes]"
        ))

        hmm_df = pd.DataFrame(fingerprint_rows)
        hmm_df = pd.concat([dataframe.reset_index(drop=True), hmm_df], axis=1)

        hmm_df = hmm_df.drop(['ReferenceAlleleVCF', 'AlternateAlleleVCF', 'Flank_1', 'Flank_2'], axis=1)

        return hmm_df

    def hmm_wrapper(self, fp_row):
        ref_allele, alt_allele, flank_1, flank_2 = fp_row
        hmm_window_range = global_config['hmm_window']
        result_dict = {}

        # multi-scaled results
        for results in self.multi_scale_viterbi(ref_allele, alt_allele, flank_1, flank_2, hmm_window_range):
            # ok remember that ref is sequence 1, alt is sequence 2
            result_dict.update(results)

        return result_dict

    def multi_scale_viterbi(self, ref_vcf, alt_vcf, flank1, flank2, hmm_window_range):
        result_list = []

        scales = [1, 2, 0.5]  # standard, long, close range analysis of path composition changes
        for scale in scales:
            scaled_window = int(hmm_window_range * scale)

            scaled_flank1 = flank1[-scaled_window:]

            scaled_flank2 = flank2[:scaled_window]
            scaled_ref = scaled_flank1 + ref_vcf + scaled_flank2
            scaled_alt = scaled_flank1 + alt_vcf + scaled_flank2

            results = self.viterbi_delta(scaled_ref, scaled_alt, len(scaled_flank1))

            scaled_results = {f"{scale}xScale_{key}": value for key, value in results.items()}
            result_list.append(scaled_results)

        return result_list  # remember to unpack this list in wrapper


    def viterbi_delta(self, scaled_ref, scaled_alt, flank_length):
        base_dict = {state_orig: 0 for state_orig in global_config['states']}
        vtbi_prob1, vtbi_seq1 = self.opt_viterbi(scaled_ref)
        vtbi_prob2, vtbi_seq2 = self.opt_viterbi(scaled_alt)

        # establish base profile
        for char in vtbi_seq1:
            base_dict[char] += float(1 / len(vtbi_seq1))

        # track changes
        for char in vtbi_seq2:
            base_dict[char] -= float(1 / len(vtbi_seq2))


        # transition state tracking - okay debug this
        base_dict.update(self.transstate_delta(vtbi_seq1, vtbi_seq2))

        # v-prob tracking
        base_dict.update({'V_prob_delta': float(vtbi_prob2 - vtbi_prob1)})

        # state determination + boundary composite scores
        base_dict.update(self.state_stats(vtbi_seq1, vtbi_seq2, flank_length))

        return base_dict

    def opt_viterbi(self, sequence):
        # to-do: log normalize the scores and fix possible division by 0
        # also: input setting to establish domain search window, maybe 1000 bps minimum is a little excessive
        seq_len = len(sequence)
        if seq_len == 0:
            return -np.inf, []

        viterbi = {}
        backptr = [{} for _ in range(seq_len)]

        # initialize - position 0
        # normalize with log probabilities
        for state in global_config['states']:
            init_prob = self.get_init_prob(state)
            emit_prob = self.get_emission_prob(state, sequence[0])

            viterbi[state] = init_prob + emit_prob

        for t in range(1, seq_len):
            viterbi_previous = viterbi
            viterbi = {}

            for state_dest in global_config['states']:
                max_prob = -np.inf
                max_state = None

                for state_orig in global_config['states']:
                    prob = (viterbi_previous[state_orig] +
                            self.get_transition_prob(state_dest, state_orig))

                    if prob > max_prob:
                        max_prob = prob
                        max_state = state_orig

                # stores the probability score and the original state
                viterbi[state_dest] = (max_prob +
                                       self.get_emission_prob(state_dest, sequence[t]))
                backptr[t][state_dest] = max_state

        # traceback
        max_state = max(viterbi.items(), key=lambda x: x[1])[0]
        max_prob = viterbi[max_state]

        # reconstruct path
        path = [max_state]
        for t in range(seq_len - 1, 0, -1):  # iterate backwards and figure out which state you came from
            max_state = backptr[t][max_state]
            path.insert(0, max_state)

        return max_prob, path

    @staticmethod
    def get_init_prob(state):
        if state in global_config['states']:
            prob = global_config['init_probs'][state]
            return np.log(prob) if prob > 0 else -np.inf
        else:
            return -np.inf

    @staticmethod
    def get_emission_prob(state, symbol):
        if state in global_config['states']:
            if symbol in AMBIGUOUS:
                prob = np.mean([global_config['emit_probs'][state][char] for char in IUPAC_CODES[symbol]])
                return np.log(prob) if prob > 0 else -np.inf
            else:
                prob = global_config['emit_probs'][state][symbol]
                return np.log(prob) if prob > 0 else -np.inf
        else:
            return -np.inf

    @staticmethod
    def get_transition_prob(state_orig, state_dest):
        if state_orig in global_config['states'] and state_dest in global_config['states']:
            prob = global_config['trans_probs'][state_orig][state_dest]
            return np.log(prob) if prob > 0 else -np.inf
        else:
            return -np.inf

    def transstate_delta(self, v_seq1, v_seq2):
        ts_delta = {}
        ts_statecopy = global_config['transstate_dict']

        ts_dict1 = self.transstate_composition(v_seq1, ts_statecopy)
        ts_dict2 = self.transstate_composition(v_seq2, ts_statecopy)

        for key in ts_statecopy.keys():
            ts_delta.update({key: int(ts_dict1[key] - ts_dict2[key])})

        return ts_delta

    @staticmethod
    def transstate_composition(seq, state_dict):
        """
        transitional state change composition tracker
        :param seq:
        :param state_dict: initialized at class instantiation
        :return:
        """
        copy = state_dict.copy()

        # build dictionary based off of all possible state changes and track differences between them
        for idx in range(len(seq)-1):
            current_state = seq[idx]
            next_state = seq[idx+1]
            if next_state != current_state:
                copy[f"{current_state}_to_{next_state}"] += 1
        return copy

    # future work -> variant level analysis
    # variant level barely changes the genomic context -> so maybe focus effort into close range
    # track state prob. changes at the variant pos.
    # track what domain the mutation occurred, composition of its neighbours

    def state_stats(self, state_seq1, state_seq2, flank_length):
        stat_dict = {}

        seq1_states, rnb1, rnb2 = self.state_identifier(state_seq1, flank_length)
        seq2_states, anb1, anb2  = self.state_identifier(state_seq2, flank_length)

        # get statistics for core variant sequences - no flag because we'll fill this in during comparison
        rv_stat_analysis = self.state_analysis(seq1_states, flag='')
        rv_subg_analysis = self.subgroup_analysis(seq1_states, flag= '')
        av_stat_analysis = self.state_analysis(seq2_states, flag='')
        av_subg_analysis = self.subgroup_analysis(seq2_states, flag='')

        # mutation-resulted stat shifts
        stat_dict.update(self.compare_dicts(rv_stat_analysis, av_stat_analysis, flag='varState_delta'))
        stat_dict.update(self.compare_dicts(rv_subg_analysis, av_subg_analysis, flag='varSubgr_delta'))

        # statistics about neighbouring sequences - maybe this will be a little difficult to navigate
        stat_dict.update(self.state_analysis(rnb1, flag='RefNB1'))
        stat_dict.update(self.subgroup_analysis(rnb1, flag='RefNB1'))

        stat_dict.update(self.state_analysis(rnb2, flag='RefNB2'))
        stat_dict.update(self.subgroup_analysis(rnb2, flag='RefNB2'))

        # boundary composite delta
        stat_dict.update(self.boundary_composite_delta(seq1_states, seq2_states, rnb1, rnb2))

        return stat_dict


    @staticmethod
    def state_identifier(state_seq, len_flank):
        variant_length = len(state_seq) - (2 * len_flank)
        variant_start = len_flank
        variant_end = len_flank + variant_length

        variant_states = state_seq[variant_start: variant_end]

        neighbour_window = len_flank // 2

        neighbour1 = state_seq[variant_start - neighbour_window: variant_start]
        neighbour2 = state_seq[variant_end:variant_end + neighbour_window]

        return variant_states, neighbour1, neighbour2

    @staticmethod
    def state_analysis(state_seq, flag, pseudocount = 1):
        possible_states = {state: 0 for state in global_config['states']}

        # record all states in mutation zone
        for state in state_seq:
            possible_states[state] += 1

        # name items according to the flag given - variant, neighbour1, neighbour2
        result_dict = {}
        result_dict.update({f"{flag}_{state}_rawc": result for state, result in possible_states.items()})
        seqlen = max(len(state_seq), 1)
        result_dict.update({f"{flag}_{state}_log2": float(np.log2((result+pseudocount)/seqlen)) for state, result in possible_states.items()})
        result_dict.update({f"{flag}_{state}_pctg": (result/seqlen) for state, result in possible_states.items()})

        return result_dict


    @staticmethod
    def subgroup_analysis(state_seq, flag, pseudocount = 1):
        subgroup_results = {
            'coding': 0,
            'noncoding': 0,
            'regulatory': 0,
            'transcribable': 0,
            'processing': 0
        }

        domain_dict = {
            'Exon': ['coding', 'transcribable', 'processing'],
            'Intron': ['noncoding', 'regulatory'],
            'Intergenic': ['noncoding'],
            '3UTR': ['noncoding', 'regulatory', 'transcribable', 'processing'],
            '5UTR': ['noncoding', 'regulatory', 'transcribable', 'processing'],
            'CDS': ['coding', 'transcribable']
        }

        for group in state_seq:
            for subgroup in domain_dict[group]:
                subgroup_results[subgroup] += 1

        result_dict = {}
        result_dict.update({f"{flag}_{group}_rawc": result for group, result in subgroup_results.items()})  # raw count
        seqlen = max(len(state_seq), 1)
        result_dict.update({f"{flag}_{group}_log2": float(np.log2((result+pseudocount)/seqlen)) for group, result in subgroup_results.items()})
        result_dict.update({f"{flag}_{group}_pctg": (result / seqlen) for group, result in subgroup_results.items()})

        return result_dict

    @staticmethod
    def compare_dicts(dict1, dict2, flag):
        return {f"{flag}_{key}": dict1[key] - dict2[key] for key in dict1 if key in dict2}

    def boundary_composite_delta(self, ref_vcf_vtbi, alt_vcf_vtbi, f1_vtbi, f2_vtbi):
        """
        Calculate composite score based on how close domain boundaries are to the mutation site
        """
        if len(f2_vtbi) != len(f1_vtbi):
            raise ValueError("Flank Length Inconsistent")

        flank_length = len(f1_vtbi)

        # get coordinates
        variant_start = flank_length
        refvar_end = flank_length + len(ref_vcf_vtbi)
        altvar_end = flank_length + len(alt_vcf_vtbi)

        # reassemble full state sequences
        ref_full = f1_vtbi + ref_vcf_vtbi + f2_vtbi
        alt_full = f1_vtbi + alt_vcf_vtbi + f2_vtbi

        # get boundaries
        ref_boundaries = [idx for idx in range(len(ref_full) - 1) if ref_full[idx] != ref_full[idx + 1]]
        alt_boundaries = [idx for idx in range(len(alt_full) - 1) if alt_full[idx] != alt_full[idx + 1]]

        ref_composite = self.boundary_gaussian(ref_boundaries, variant_start, refvar_end, len(ref_vcf_vtbi))
        alt_composite = self.boundary_gaussian(alt_boundaries, variant_start, altvar_end, len(alt_vcf_vtbi))

        # get coordinates of both mutation zones - start and stop
        # get coordinates of all domain boundaries
        # gaussian decay for domain boundaries the closer it gets to mutation zone

        return {"Boundary_Composite_delta": alt_composite - ref_composite}

    def boundary_gaussian(self, idxs, var_start, var_end, sigma):
        distances = self.distance_from_window(idxs, var_start, var_end)
        weights = self.gaussian_eq(distances, sigma)

        return sum(weights)

    @staticmethod
    def distance_from_window(idx_list, window_start, window_end):
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
