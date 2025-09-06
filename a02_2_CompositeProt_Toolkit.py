import yaml
from pathlib import Path

# math
import numpy as np
import pandas as pd

# helpers
from collections import Counter
from re import finditer
from tqdm import tqdm
from itertools import product
from Bio.SeqUtils import ProtParam

# multiprocessing
import multiprocessing
from multiprocessing import Pool

# initialize biological structure libraries
from b00_bio_library import *


global_config = {}

def config_init(config, start, stops, ambiguous, iupac_codes):
    """
    Create global variables for each process to access external data and config
    :param config:
    :param start:
    :param stops:
    :param ambiguous:
    :param iupac_codes:
    :return:
    """
    global global_config, START, STOPS, AMBIGUOUS, IUPAC_CODES
    global_config = config
    START = start
    STOPS = stops
    AMBIGUOUS = ambiguous
    IUPAC_CODES = iupac_codes

class CompositeProt:
    _pool = None

    def __init__(self, core_num=None):
        self.root = Path(__file__).parent.resolve()
        with open(self.root / 'config.yaml', 'r') as outfile:
            self.cfg = yaml.safe_load(outfile)

            # Navigation
            self.folder = Path('database')
            self.blosum62_file = self.folder / self.cfg['protein_substitution_matrix']

            # buffer region
            self.aa_buffer = self.cfg['AA_buffer_region']

            # nucleotide maps
            self.complements = COMPLEMENTS
            self.start_codons = '|'.join(START)
            self.stop_codons = '|'.join(STOPS)

            # codon scores - atg is best, then degrade as we go down
            self.start_scores = {
                "ATG": 4,
                "CTC": 3,
                "CTG": 2,
                "TTG": 1,
            }

            # codon to amino acid map
            self.codon_map = DNA_CODON_TO_AA

            # molecular weight dict
            self.molecular_weights = MOLECULAR_WEIGHTS

            # charge dict
            self.net_charges = NET_CHARGES

            # isoelectric point dict
            self.isoelectric_pts = ISOELECTRIC_PTS

            # hydrophobicity dict
            self.hydrophob_idxs = HYDROPHOBICITY_IDXS

            # half life dict
            self.half_life = HALF_LIFE

            # minimum protein length
            self.min_len_prot = self.cfg['minimum_protein_length']

            # protein substitution matrix - will be populated with blosum62 matrix later on
            self.prot_submat = self.read_submat_file()

            # gap penalty for alignments
            self.gap_penalty = self.cfg['protein_gap_penalty']


        self.config = {
            'aa_buffer': self.aa_buffer,
            'complements': self.complements,
            'start_codons': self.start_codons,
            'stop_codons': self.stop_codons,
            'start_scores': self.start_scores,
            'codon_map': self.codon_map,
            'molecular_weights': self.molecular_weights,
            'net_charges': self.net_charges,
            'isoelectric_pts': self.isoelectric_pts,
            'hydrophob_idxs': self.hydrophob_idxs,
            'half_life': self.half_life,
            'min_len_prot': self.min_len_prot,
            'prot_submat': self.prot_submat,
            'gap_penalty': self.gap_penalty
        }

        # define number of cores
        self.core_num = core_num if core_num is not None else multiprocessing.cpu_count() - 2

        # initialize pool if not already initialized
        if CompositeProt._pool is None:
            self.initialize_pool()


    def initialize_pool(self):
        """
        Initiate multiprocessing pool w/ provided configuration and data
        """
        core_num = self.core_num

        if CompositeProt._pool is None:
            CompositeProt._pool = multiprocessing.Pool(
                processes=core_num,
                initializer=config_init,
                initargs=(self.config, START, STOPS, AMBIGUOUS, IUPAC_CODES)
            )


    def terminate_pool(self):
        """
        Terminate multiprocessing pool when no longer needed
        """
        if CompositeProt._pool is not None:
            CompositeProt._pool.close()
            CompositeProt._pool.join()
            CompositeProt._pool = None

    def __enter__(self):
        self.initialize_pool()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate_pool()

    # ====[[MUTATION FINGERPRINT]]====
    def gen_AAfp_dataframe(self, dataframe):
        """
        RUN THIS TO GET RETURN A FULL FINGERPRINT DATAFRAME
        :param dataframe:
        :return:
        """
        fingerprint_rows = [
            (row['Chromosome'], row['ReferenceAlleleVCF'],
             row['AlternateAlleleVCF'], row['Flank_1'], row['Flank_2'])
            for _, row in dataframe.iterrows()
        ]

        fingerprint_rows = list(tqdm(
            CompositeProt._pool.imap(self.AA_fingerprinter_wrapper, fingerprint_rows),
            total=len(fingerprint_rows),
            desc="[Generating AA chain mutation fingerprints]"
        ))

        fingerprint_df = pd.DataFrame(fingerprint_rows)
        fingerprint_df = pd.concat([dataframe.reset_index(drop=True), fingerprint_df],
                                   axis=1)  # axis = 1 to concatenate column wise (side by side)

        # drop everything here - DNA keeps significance and chromosome already
        fingerprint_df = fingerprint_df.drop(
            ['Chromosome', 'ClinicalSignificance', 'ReferenceAlleleVCF', 'AlternateAlleleVCF', 'Flank_1',
             'Flank_2'], axis=1)

        return fingerprint_df

    #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    def AA_fingerprinter_wrapper(self, fp_row):
        chromosome, ref_allele, alt_allele, flank_1, flank_2 = fp_row
        config = global_config

        buffered_ref = flank_1[config['aa_buffer']:] + ref_allele + flank_2[:config['aa_buffer']]
        buffered_alt = flank_1[config['aa_buffer']:] + alt_allele + flank_2[:config['aa_buffer']]

        optimal_ref, ref_orf_coordinates = self.gen_orf(buffered_ref)
        optimal_alt, alt_orf_coordinates = self.gen_orf(buffered_alt)

        # get DNA sequence most likely to be translated via intelligent ORF search
        ref_sequence = (optimal_ref[ref_orf_coordinates[0]: ref_orf_coordinates[1]]).upper()
        alt_sequence = (optimal_alt[alt_orf_coordinates[0]: alt_orf_coordinates[1]]).upper()

        # get protein sequence as list, includes ambiguous cases as list of possible amino acids
        ref_protein_list = self.protein_from_DNA(ref_sequence)
        alt_protein_list = self.protein_from_DNA(alt_sequence)

        # make it all into a string, replace ambiguous cases with the most likely / first amino acid
        # - too computationally expensive to take into account ambiguous cases with these long sequences
        non_ambiguous_ref = self.nonambi_prot(ref_sequence)
        non_ambiguous_alt = self.nonambi_prot(alt_sequence)

        # will need this for a future feature interaction
        ref_protein_length = len(non_ambiguous_ref)

        profile = self.structural_profile(ref_protein_list, alt_protein_list, non_ambiguous_ref, non_ambiguous_alt,
                                          ref_protein_length, config)

        return profile


    # substitution matrix generation for alignment calculations
    def read_submat_file(self):
        sm = {}
        with open(self.blosum62_file, 'r') as f:
            header = f.readline().strip().split()
            alphabet = header

            for line in f:
                tokens = line.strip().split()
                row_aa = tokens[0]
                scores = tokens[1:]

                for col_aa, score in zip(alphabet, scores):
                    pair = row_aa + col_aa
                    sm[pair] = int(score)

        return sm


    # ====[GENERATING MOST LIKELY ORF SEQUENCE]====
    def gen_orf(self, ref_full):
        """
        Returns the DNA sequence most likely to be translated into an amino acid chain / protein
        Look at self.optimal_orf_search for algorithm logic
        :param ref_full:
        :return: DNA sequence most likely to be translated
        """
        reverse_complement = ref_full.translate(global_config['complements'])[::-1]

        seq, ref_orf, ref_orf_score = self.optimal_orf_search(ref_full)
        rc_seq, rc_orf, rc_orf_score = self.optimal_orf_search(reverse_complement)

        # since the reverse complement may have the best orf, we should return the sequence, and not the orf coordinates alone
        # otherwise we would be translating the default sequence with the reverse complement's orf coordinates
        if ref_orf_score > rc_orf_score:
            return seq, ref_orf
        elif ref_orf_score < rc_orf_score:
            return rc_seq, rc_orf
        else:
            return seq, ref_orf  # default to ref seq in the extremely rare event the scores end up being the same

    # OPTIMIZED ORF CALCULATOR
    def optimal_orf_search(self, sequence):
        """
        Scans the given reference sequence for any ORFs.
        Utilizes math-based pre-processing to intelligently scout out the suitable ORFs
        Criteria for ORF:
        - divisible by 3 (%3)
        - length = minimum protein length * 3
        Selects the potential ORF with the highest score and returns its sequenced form
        Score criteria:
        - % length of the sequence
        - kozak motif alignment - most important positions are -3 and +4 relative to the first character of start codon
        - start codon quality - ATG > CTC > CTG > TTG - anything after ATG slowly drops in score
        :param sequence
        :return: ORF of the sequence
        """
        # start translation from the SAME spot on the mutation sequence to compare

        start_idxs = self.start_positions(sequence)
        stop_idxs = self.stop_positions(sequence)
        potential_frames = []
        # find all numbers that are at least self.min_prot_length * 3 (codons) apart and have a modulo 3 of 0
        # see if they have a kozak sequence 10bp behind the start, get ORF and calculate ORF score
        # length / 100 + start_codon score, + kozak score

        # find all potential reading frames - multiples of 3 and are at least 'x' amino acids (10 * 3 nucleotides) long
        for start_idx in start_idxs:
            frame = start_idx % 3
            valid_stops = [stop_idx for stop_idx in stop_idxs if stop_idx > start_idx and stop_idx % 3 == frame]
            for stop in valid_stops:
                orf_length = stop - start_idx
                if orf_length < global_config['min_len_prot'] * 3:
                    continue
                # check for in-frame stop codons
                has_inframe_stop = False
                for pos in range(start_idx + 3, stop, 3):
                    codon = sequence[pos:pos + 3]
                    if codon in STOPS:
                        has_inframe_stop = True
                        break
                if not has_inframe_stop:
                    potential_frames.append((start_idx, stop + 3))

        ORF_data = {}
        for potential_frame in potential_frames:
            start_pos = potential_frame[0]
            stop_pos = potential_frame[1]

            # extract features for scoring - length, start codon type, and kozak motif
            orf_length = stop_pos - start_pos
            start_codon = sequence[start_pos: start_pos + 3]
            score = (self.start_scores.get(start_codon, 0) +
                     self.kozak_score(start_pos, sequence) + (
                        (orf_length / global_config['min_len_prot']) * 100))
            ORF_data[potential_frame] = score

        if not ORF_data:
            # break statement in protein translation function helps handle empty dictionaries
            return sequence, (0, len(sequence)), 0

        sorted_ORFS = sorted(ORF_data.items(), key=lambda item: item[1], reverse=True)

        coordinates = sorted_ORFS[0][0]
        orf_score = sorted_ORFS[0][1]  # we will need to compare this to the reverse complement later

        return sequence, coordinates, orf_score
        # keep track of valid orfs

    @staticmethod
    def start_positions(sequence):
        """
        Find all start codons - no need for overlap, biologically redundant to search for it
        :param sequence:
        :return:
        """
        return [match.start() for match in finditer(global_config['start_codons'], sequence)]

    @staticmethod
    def stop_positions(sequence):
        """
        Finds all stop codons - overlap detections
        :param sequence:
        :return:
        """
        return [match.start() for match in finditer(global_config['stop_codons'], sequence)]

    @staticmethod
    def kozak_score(start_pos, ref_seq):
        if start_pos < 3 or start_pos + 6 >= len(ref_seq):
            return 0  # doesn't fit dimensions for kozak motif

        subseq = ref_seq[start_pos - 3: start_pos + 6]
        score = 0
        # the most important positions in a Kozak sequence are A or G at -3, and G at +4 relative to ATG
        if len(subseq) >= 6:
            if subseq[0] in 'AG': score += 3
            if subseq[7] == 'G': score += 2

        return score


    # STRUCTURAL FEATURE DETECTION
    def structural_profile(self, ref_list, alt_list, ref_prot, alt_prot, ref_length, config):
        """
        Generates molecular fingerprint for structural features:
        prolines, secondary structure propensities, global scores and pct scores
        :param ref_list:
        :param alt_list:
        :param ref_prot:
        :param alt_prot:
        :param ref_length:
        :param config:
        :return:
        """
        super_dict = self.base_fingerprint(ref_list, alt_list, ref_prot, alt_prot, ref_length, config)

        # add structural variant specific data entries - worth more going deeper for structural genes
        super_dict['Proline_delta'] = self.proline_delta(ref_list, alt_list)
        super_dict['Secondary_struct_propensity'] = self.secondary_propensity_delta(ref_prot, alt_prot)

        # add global alignment data
        global_score, global_pct, global1, global2 = self.needleman_data(ref_prot, alt_prot, config)
        alignment_dict = {
            'AA_Global_score': global_score,
            'AA_Global_pct': global_pct,
        }

        super_dict.update(alignment_dict)


        # STRUCTURAL FEATURE INTERACTIONS
        # relative global alignment
        super_dict['FI_relative_global'] = super_dict['AA_Global_score'] / ref_length

        super_dict['FI_Secondary_Global'] = super_dict['Secondary_struct_propensity']  * super_dict['AA_Global_pct']

        super_dict['FI_Hydro_Structs'] = super_dict['Secondary_struct_propensity'] * super_dict['Hydrophobicity_delta']

        super_dict['FI_Proline_Structs'] = super_dict['Secondary_struct_propensity'] * super_dict['Proline_delta']

        return super_dict

    # BASE PROFILE GENERATION
    def base_fingerprint(self, ref_prot_list, alt_prot_list, ref_prot_seq, alt_prot_seq, ref_length, config):
        """
        Generates a structural profile with the protein sequences fed into it
        :param ref_prot_list:
        :param alt_prot_list:
        :param ref_prot_seq:
        :param alt_prot_seq:
        :param ref_length:
        :param config:
        :return:
        """
        base_dict = {
            'Length_change': self.length_change(ref_prot_list, alt_prot_list),
            'Molecular_weight_delta': self.molecular_weight_delta(ref_prot_list, alt_prot_list),
            'Net_charge_delta': self.net_charge_delta(ref_prot_list, alt_prot_list),
            'Isoelectric_delta': self.isoelectric_delta(ref_prot_list, alt_prot_list),
            'Hydrophobicity_delta': self.hydrophobicity_delta(ref_prot_list, alt_prot_list),
            'Aliphatic_delta': self.aliphatic_delta(ref_prot_seq, alt_prot_seq),
            'Half_life_delta': self.half_life_delta(ref_prot_list, alt_prot_list),
            'AA_Local_score': self.smith_waterman_data(ref_prot_seq, alt_prot_seq, config)
        }

        # FEATURE Engineering
        base_dict['FI_charge_composite'] = base_dict['Net_charge_delta'] * base_dict['Isoelectric_delta']

        base_dict['FI_fold_composite'] = base_dict['Aliphatic_delta'] * base_dict['Hydrophobicity_delta']

        base_dict['FI_Relative_weight'] = base_dict['Molecular_weight_delta'] / ref_length

        base_dict['FI_Stability_composite'] = base_dict['Net_charge_delta'] * base_dict['Hydrophobicity_delta']

        base_dict['FI_Charge_Hydro'] = base_dict['Net_charge_delta'] * base_dict['Hydrophobicity_delta']

        base_dict['FI_Weight_life'] = base_dict['Half_life_delta'] * base_dict['Molecular_weight_delta']

        base_dict['FI_Iso_Hydro'] = base_dict['Isoelectric_delta'] * base_dict['Hydrophobicity_delta']

        # relative local alignment
        base_dict['relative_local'] = base_dict['AA_Local_score'] / ref_length
        return base_dict


    # ====[TRANSLATING DNA SEQUENCES]====
    @staticmethod
    def protein_from_DNA(sequence):
        """
        Translates a given DNA sequence into its amino acid chain form as a list.
        The list format is to help handle ambiguous cases, where we can nest another list of all possible amino acids
        from that ambiguous codon -> an average values in downstream calculations
        :param sequence:
        :return:
        """
        protein_sequence = []
        for pos in range(0, len(sequence) - 2, 3):
            codon = sequence[pos:pos + 3]
            if codon in STOPS and len(protein_sequence) == 0:
                continue
            elif codon in STOPS:
                break
            # detect ambiguous characters -> generate list of possible amino acids instead for precomputed values
            if any(char in AMBIGUOUS for char in codon):
                possible_nucleotides = [IUPAC_CODES[base] for base in codon]
                possible_codons = [''.join(nt) for nt in product(*possible_nucleotides)]
                protein_sequence.append([global_config['codon_map'][cod] for cod in possible_codons if cod not in STOPS])
            else:
                protein_sequence.append(global_config['codon_map'][codon])
        return protein_sequence

    @staticmethod
    def nonambi_prot(sequence):
        motif_ready_seq = []
        for char in sequence:
            if isinstance(char, list):
                frequencies = dict(Counter(char))
                freq = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)
                # add the most frequent / first amino acid in the list of possibilities
                motif_ready_seq.append(freq[0][0])
            else:
                motif_ready_seq.append(char)
        return ''.join(motif_ready_seq)

    # ====[[PROTEIN ANALYSIS]]=====
    # make overarching function calling on a single pass function for both ref and alt seqs
    # have one function perform a single pass on a given sequence and gather data from downstream functions
    # these downstream functions calculate physicochemical and structural properties

    @staticmethod
    def length_change(ref_protein, alt_protein):
        return len(alt_protein) - len(ref_protein)

    # molecular weight (kDa)
    def molecular_weight_delta(self, ref_protein_list, alt_protein_list):
        return self.mol_weight(alt_protein_list) - self.mol_weight(ref_protein_list)
    @staticmethod
    def mol_weight(sequence):
        molecular_weight = 0
        for idx in sequence:
            if isinstance(idx, list) and idx:  # skip empty lists in the event all possible AAs are stops
                molecular_weight += np.mean([global_config['molecular_weights'][aa] for aa in idx])
            elif isinstance(idx, str) and idx in global_config['molecular_weights']:
                molecular_weight += global_config['molecular_weights'][idx]
            else:
                molecular_weight += 0
        return molecular_weight


    # net charge at physiological pH ~7.4
    def net_charge_delta(self, ref_protein_list, alt_protein_list):
        return self.net_charge(alt_protein_list) - self.net_charge(ref_protein_list)
    @staticmethod
    def net_charge(sequence):
        net_charge = 0
        for idx in sequence:
            if isinstance(idx, list) and idx:
                net_charge += np.mean([global_config['net_charges'][aa] for aa in idx])
            elif isinstance(idx, str) and idx in global_config['net_charges']:
                net_charge += global_config['net_charges'][idx]
            else:
                net_charge += 0
        return net_charge


    # isoelectric point - pI
    def isoelectric_delta(self, ref_protein_list, alt_protein_list):
        return self.isoelectric_all(alt_protein_list) - self.isoelectric_all(ref_protein_list)
    @staticmethod
    def isoelectric_all(sequence):
        isoelectric = 0
        for idx in sequence:
            if isinstance(idx, list) and idx:
                isoelectric += np.mean([global_config['isoelectric_pts'][aa] for aa in idx])
            elif isinstance(idx, str) and idx in global_config['isoelectric_pts']:
                isoelectric += global_config['isoelectric_pts'][idx]
            else:
                isoelectric += 0
        return isoelectric


    # hydrophobicity idx
    def hydrophobicity_delta(self, ref_protein_list, alt_protein_list):
        return self.hydrophobicity_idx(alt_protein_list) - self.hydrophobicity_idx(ref_protein_list)
    @staticmethod
    def hydrophobicity_idx(sequence):
        hydro_idx = 0
        for idx in sequence:
            if isinstance(idx, list) and idx:
                hydro_idx += np.mean([global_config['hydrophob_idxs'][aa] for aa in idx])
            elif isinstance(idx, str) and idx in global_config['hydrophob_idxs']:
                hydro_idx += global_config['hydrophob_idxs'][idx]
            else:
                hydro_idx += 0
        return hydro_idx


    # aliphatic index - thermal stability
    def aliphatic_delta(self, ref_protein_seq, alt_protein_seq):
        return self.aliphatic_idx(alt_protein_seq) - self.aliphatic_idx(ref_protein_seq)
    @staticmethod
    def aliphatic_idx(sequence):
        ala_pct = (sequence.count('A') / len(sequence)) * 100
        val_pct = (sequence.count('V') / len(sequence)) * 100
        iso_pct = (sequence.count('I') / len(sequence)) * 100
        leu_pct = (sequence.count('L') / len(sequence)) * 100

        return ala_pct + (2.9 * val_pct) + (3.9 * (iso_pct + leu_pct))


    # instability index - protein half life
    def half_life_delta(self, ref_protein_list, alt_protein_list):
        return self.half_life_count(alt_protein_list) - self.half_life_count(ref_protein_list)
    @staticmethod
    def half_life_count(sequence):
        half_life = 0
        for idx in sequence:
            if isinstance(idx, list) and idx:
                half_life += np.mean([global_config['half_life'][aa] for aa in idx])
            elif isinstance(idx, str) and idx in global_config['half_life']:
                half_life += global_config['half_life'][idx]
            else:
                half_life += 0
        return half_life


    # reserve these for structural variants
    def proline_delta(self, ref_protein_list, alt_protein_list):
        return self.prolines(alt_protein_list) - self.prolines(ref_protein_list)
    @staticmethod
    def prolines(sequence):
        proline_count = 0
        for idx in sequence:
            if isinstance(idx, list) and idx:
                proline_count += sum(1 for aa in idx if aa == 'P') / len(idx)
            elif isinstance(idx, str) and idx == 'P':
                proline_count += 1
            else:
                proline_count += 0
        return proline_count


    # secondary structure propensity delta
    def secondary_propensity_delta(self, ref_protein_seq, alt_protein_seq):
        ref_helix, ref_turn, ref_sheet = self.secondary_prop(ref_protein_seq)
        alt_helix, alt_turn, alt_sheet = self.secondary_prop(alt_protein_seq)

        return (alt_helix - ref_helix) + (alt_turn - ref_turn) + (alt_sheet - ref_sheet)

    @staticmethod
    def secondary_prop(sequence):
        analyzer = ProtParam.ProteinAnalysis(sequence)
        helix_frac, turn_frac, sheet_frac = analyzer.secondary_structure_fraction()
        return helix_frac, turn_frac, sheet_frac



    # ====[Future update: MOTIF SCANNING]====
    # scanning for changes in motifs for the most commonly searched post-translational modifications
    # phosphorylation, ubiquitination, glycosylation
    # coming soon i need to do more research on this honestly before I pick the wrong motifs to gather

    # phosphorylation domains
    # ubituination domains
    # glycosylation domains

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
            glb_score_matrix[0].append(config['gap_penalty'] * col)
            glb_traceback_matrix[0].append(3)

        # initialize gap column (all gaps in seq 2)
        for row in range(1, len(original) + 1):
            glb_score_matrix.append([config['gap_penalty'] * row])
            glb_traceback_matrix.append([2])

        # apply recurrence relation to fill remaining of matrix
        for row in range(len(original)):
            for col in range(len(mutation)):
                # calculate scores for 3 possible moves:
                diagonal_score = glb_score_matrix[row][col] + self.score_pos(original[row],
                                                                             mutation[col], config)  # Diagonal
                up_score = glb_score_matrix[row][col + 1] + config['gap_penalty']  # Up
                left_score = glb_score_matrix[row + 1][col] + config['gap_penalty']  # Left
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
                up_score = lcl_score_matrix[row][col + 1] + config['gap_penalty']
                left_score = lcl_score_matrix[row + 1][col] + config['gap_penalty']
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
            return config['gap_penalty']
        else:
            return config['prot_submat'][char_1 + char_2]  # look up the score in submat unless a gap is there

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
    # future improvement - scoring function to return average of scores when ambiguous proteins brought up
    def needleman_data(self, loc_original, loc_mutation, config):
        """
        Local spec. - Carries out needleman-wunsch alignments on the two strings
        :param loc_original: local region - ref_allele
        :param loc_mutation: local region - alt_allele
        :param config:
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

        return local_score

