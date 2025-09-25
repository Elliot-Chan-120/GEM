# DNA + PROTEIN ANALYSIS LIBRARY

# 1 ====[DNA DATA]====
PURINES = ['AG']
PYRIMIDINES= ['CT']

# THERMODYNAMIC PARAM - Free Energy from Nearest Neighbours
# used this because the physiologically relevant temperature of 37 is the constant
# would only need enthalpy and entropy values if we're looking at temp-sensitive variants or temp-changing experiments
# each key, when bound to their reverse complement, will release 'value' kcal/mol upon binding
NN_FREE_ENERGY = {
    # Default
    'AA': -1.02, 'AT': -0.73, 'TA': -0.60,
    'CA': -1.38, 'GT': -1.43, 'CT': -1.16,
    'GA': -1.46, 'CG': -2.09, 'GC': -2.28,
    'GG': -1.77,

    # Reverse Complements
    'TT': -1.02, 'TG': -1.38, 'AC': -1.43,
    'AG': -1.16, 'TC': -1.46, 'CC': -1.77,

    # AMBIGUOUS CHARACTER AVERAGES - PRECOMPUTED

    'AR' : -1.090, 'AY' : -1.080, 'AS' : -1.295,
    'AW' : -0.875, 'AK' : -0.945, 'AM' : -1.225,
    'AB' : -1.107, 'AD' : -0.970, 'AH' : -1.06,
    'AV' : -1.203, 'AN' : -1.085,

    'TR' : -0.99, 'TY' : -1.24, 'TS' : -1.42,
    'TW' : -0.81, 'TK' : -1.20, 'TM' : -1.03,
    'TB' : -1.287, 'TD' : -1.0, 'TH' : -1.0267,
    'TV' : -1.147, 'TN' : -1.115,

    'CR' : -1.735, 'CY' : -1.465, 'CS' : -1.93,
    'CW' : -1.27, 'CK' : -1.625, 'CM' : -1.575,
    'CB' : -1.673, 'CD' : -1.543, 'CH' : -1.437,
    'CV' : -1.747, 'CN' : -1.6,

    'GR' : -1.615, 'GY' : -1.855, 'GS' : -2.025,
    'GW' : -1.445, 'GK' : -1.6, 'GM' : -1.87,
    'GB' : -1.83, 'GD' : -1.553, 'GH' : -1.723,
    'GV' : -1.837, 'GN' : -1.735,

    # AMBIGUOUS CHARACTER AVERAGES - PRECOMPUTED REVERSED
    'RA' : -1.240, 'YA' : -0.990, 'SA' : -1.420,
    'WA' : -0.810,'KA' : -1.030,'MA' : -1.200,
    'BA' : -1.147,'DA' : -1.027,'HA' : -1.000,
    'VA' : -1.287,'NA' : -1.115,

    'RT' : -1.080,'YT' : -1.090,'ST' : -1.295,
    'WT' : -0.875,'KT' : -1.225,'MT' : -0.945,
    'BT' : -1.203,'DT' : -1.060,'HT' : -0.970,
    'VT' : -1.107,'NT' : -1.085,

    'RC' : -1.855,'YC' : -1.615,'SC' : -2.025,
    'WC' : -1.445,'KC' : -1.870,'MC' : -1.600,
    'BC' : -1.837,'DC' : -1.723,'HC' : -1.553,
    'VC' : -1.827,'NC' : -1.735,

    'RG' : -1.465,'YG' : -1.735,'SG' : -1.930,
    'WG' : -1.270,'KG' : -1.575,'MG' : -1.625,
    'BG' : -1.747, 'DG' : -1.437, 'HG' : -1.543,
    'VG' : -1.673,'NG' : -1.600,

    # AMBIGUOUS CHARACTER AVERAGES - PRECOMPUTED ADJACENT AMBIGUOUS
    'RR': -1.353, 'RY': -1.468,'RS': -1.660,
    'RW': -1.160,'RK': -1.273,'RM': -1.548,
    'RB': -1.467,'RD': -1.262,'RH': -1.392,
    'RV': -1.520,'RN': -1.410,

    'YR': -1.365,'YY': -1.353,'YS': -1.675,
    'YW': -1.040,'YK': -1.413,'YM': -1.303,
    'YB': -1.480,'YD': -1.277,'YH': -1.237,
    'YV': -1.447,'YN': -1.358,

    'SR': -1.675,'SY': -1.660,'SS': -1.978,
    'SW': -1.358,'SK': -1.613,'SM': -1.723,
    'SB': -1.750,'SD': -1.548,'SH': -1.580,
    'SV': -1.792,'SN': -1.668,

    'WR': -1.040,'WY': -1.160,'WS': -1.358,
    'WW': -0.843,'WK': -1.073,'WM': -1.123,
    'WB': -1.197,'WD': -0.985,'WH': -1.043,
    'WV': -1.175,'WN': -1.100,

    'KR': -1.303,'KY': -1.548,'KS': -1.723,
    'KW': -1.128,'KK': -1.400,'KM': -1.450,
    'KB': -1.557,'KD': -1.277,'KH': -1.375,
    'KV': -1.492,'KN': -1.425,

    'MR': -1.413,'MY': -1.273,'MS': -1.613,
    'MW': -1.073, 'MK': -1.285,'MM': -1.400,
    'MB': -1.390, 'MD': -1.257,'MH': -1.248,
    'MV': -1.475,'MN': -1.343,

    'BR': -1.447,'BY': -1.520,'BS': -1.792,
    'BW': -1.175, 'BK': -1.475,'BM': -1.492,
    'BB': -1.596, 'BD': -1.366,'BH': -1.396,
    'BV': -1.577, 'BN': -1.483,

    'DR': -1.232,'DY': -1.392,'DS': -1.580,
    'DW': -1.043,'DK': -1.248, 'DM': -1.375,
    'DB': -1.407,'DD': -1.174,'DH': -1.270,
    'DV': -1.396,'DN': -1.312,

    'HR': -1.272,'HY': -1.262,'HS': -1.548,
    'HW': -0.985,'HK': -1.257,'HM': -1.277,
    'HB': -1.356, 'HD': -1.171,'HH': -1.174,
    'HV': -1.366,'HN': -1.267,

    'VR': -1.480, 'VY': -1.467,'VS': -1.750,
    'VW': -1.197,'VK': -1.390,'VM': -1.557,
    'VB': -1.536, 'VD': -1.356,'VH': -1.407,
    'VV': -1.600,'VN': -1.473,

    'NR': -1.358,'NY': -1.410,'NS': -1.668,
    'NW': -1.100,'NK': -1.343,'NM': -1.425,
    'NB': -1.473,'ND': -1.267,'NH': -1.312,
    'NV': -1.483,'NN': -1.384,
}

# left out initiations at GC, AT, symmetry corrections and 5'terminal TA bp^e
# I'm unsure where the sequence's original gene starts and stops from the clinvar data

# IUPAC nucleotide codes mapping for ambiguous stuff
IUPAC_CODES = {
    'A': ['A'], 'T': ['T'], 'C': ['C'], 'G': ['G'],
    'R': ['A', 'G'],  # puRine
    'Y': ['C', 'T'],  # pYrimidine
    'S': ['G', 'C'],  # Strong (3 H-bonds)
    'W': ['A', 'T'],  # Weak (2 H-bonds)
    'K': ['G', 'T'],  # Keto
    'M': ['A', 'C'],  # aMino
    'B': ['C', 'G', 'T'],  # not A
    'D': ['A', 'G', 'T'],  # not C
    'H': ['A', 'C', 'T'],  # not G
    'V': ['A', 'C', 'G'],  # not T
    'N': ['A', 'T', 'C', 'G']  # aNy
}


COMPLEMENTS = str.maketrans("ATCG", "TAGC")


AMBIGUOUS = "RYSWKMBDHVN"

# per chromosome ID, X: 23, Y: 24
GENE_DENSITIES = {
        1: 8.85, 2: 5.47, 3: 5.57,
        4: 4.17, 5: 4.99, 6: 6.62,
        7: 6.44, 8: 5.11, 9: 6.22,
        10: 6.12, 11: 10.3, 12: 8.16,
        13: 3.16, 14: 6.29, 15: 6.39,
        16: 10.41, 17: 15.69, 18: 3.88,
        19: 22.61, 20: 9.88, 21: 6.05,
        22: 10.44, 23: 5.75, 24: 1.38
    }


SPLICE_SIGNALS = ['GT', 'AG', 'GC', 'AT']

START = ["ATG", "CTC", "CTG", "TTG"]

STOPS = ["TAA", "TAG", "TGA"]

# keep this just in case
DNA_RIGID_TFs = ['TATAAA',                        # TATA box - promoter stability - pwm
                'GGGCGG',                         # GC-box (SP1 binding) - pwm
                '[GC]{6,}',                       # not really a TF but promotes stability
                'GGCCAATCT',                      # CAAT box - pwm
                'TGACTCA',                        # AP-1 - not yet
                'GGG(A|G)(A|G)(C|T)(C|T)CC',      # NF-κB consensus: GGGRNNYYCC - not yet
                'TGACGTCA',                       # CREB / CRE - not yet
                'ATGCAAAT',                       # Octamer (Oct-1/2) - not yet
                'CACGTG',                         # E-box (Myc/CLOCK) - not yet
                '[AG]{3}C[AT]{2}[CT]{3}',         # p53 half-site - not yet
                'GATA',                           # GATA family - not yet
                'TTGC..AA']                       # C/EBP] - not yet


# 2 ====[PROTEIN DATA]====

DNA_CODON_TO_AA = {
    # 'M' - START, '_' - STOP
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TGT": "C", "TGC": "C",
    "GAT": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "TTT": "F", "TTC": "F",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    "CAT": "H", "CAC": "H",
    "ATA": "I", "ATT": "I", "ATC": "I",
    "AAA": "K", "AAG": "K",
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATG": "M",
    "AAT": "N", "AAC": "N",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TGG": "W",
    "TAT": "Y", "TAC": "Y",
    "TAA": "_", "TAG": "_", "TGA": "_"
}

# molecular weights are in kilodaltons
MOLECULAR_WEIGHTS = {
    'A': 0.09,
    'C': 0.12,
    'D': 0.13,
    'E': 0.15,
    'F': 0.17,
    'G': 0.08,
    'H': 0.16,
    'I': 0.13,
    'K': 0.15,
    'L': 0.13,
    'M': 0.15,
    'N': 0.13,
    'P': 0.12,
    'Q': 0.15,
    'R': 0.17,
    'S': 0.11,
    'T': 0.12,
    'V': 0.12,
    'W': 0.20,
    'Y': 0.18,
}

# net charges at physiological pH ~7.4
NET_CHARGES = {
    'A': 0,
    'C': 0,
    'D': -1,
    'E': -1,
    'F': 0,
    'G': 0,
    'H': 0,
    'I': 0,
    'K': +1,
    'L': 0,
    'M': 0,
    'N': 0,
    'P': 0,
    'Q': 0,
    'R': +1,
    'S': 0,
    'T': 0,
    'V': 0,
    'W': 0,
    'Y': 0
}

# isoelectric point pHs from peptide web
# NOTE: will retrain model since I was using old textbook values with one outdated value -> shouldn't affect model accuracy much
ISOELECTRIC_PTS = {
    'A': 6.02,
    'C': 5.02,
    'D': 2.98,
    'E': 3.22,
    'F': 5.48,
    'G': 5.97,
    'H': 7.59,
    'I': 5.98,
    'K': 9.47,
    'L': 5.98,
    'M': 5.75,
    'N': 5.41,
    'P': 6.30,
    'Q': 5.65,
    'R': 10.76,
    'S': 5.68,
    'T': 5.60,
    'V': 5.97,
    'W': 5.94,
    'Y': 5.66
}

# hydrophobicity levels - pH 7
# +ve values and up = hydrophobic
# around 0 = neutral
# -ve values = hydrophilic
HYDROPHOBICITY_IDXS = {
    'A': 41,
    'C': 49,
    'D': -55,
    'E': -31,
    'F': 100,
    'G': 0,
    'H': 8,
    'I': 100,
    'K': -23,
    'L': 97,
    'M': 74,
    'N': -28,
    'P': -46,
    'Q': -10,
    'R': -14,
    'S': -5,
    'T': 13,
    'V': 76,
    'W': 97,
    'Y': 63
}


# individual half lives
HALF_LIFE = {
    'A': 4.4,
    'C': 1.2,
    'D': 1.1,
    'E': 1,
    'F': 1,
    'G': 30,
    'H': 3.5,
    'I': 20,
    'K': 1.3,
    'L': 5.5,
    'M': 30,
    'N': 1.4,
    'P': 21,
    'Q': 0.8,
    'R': 1,
    'S': 1.9,
    'T': 7.2,
    'V': 100,
    'W': 2.8,
    'Y': 2.8
}

ALL_AA_COMBINATIONS = [
    # single nts
    'A', 'T', 'C', 'G'
    
    # dinucleotides
    'AA', 'AT', 'AC',
    'AG', 'TA', 'TT',
    'TC', 'TG', 'CA',
    'CT', 'CC', 'CG',
    'GA', 'GT', 'GC', 'GG'
    
    # trinucleotides
    'AAA', 'AAT', 'AAC', 'AAG',
    'ATA', 'ATT', 'ATC', 'ATG',
    'ACA', 'ACT', 'ACC', 'ACG',
    'AGA', 'AGT', 'AGC', 'AGG',
    'TAA', 'TAT', 'TAC', 'TAG',
    'TTA', 'TTT', 'TTC', 'TTG',
    'TCA', 'TCT', 'TCC', 'TCG',
    'TGA', 'TGT', 'TGC', 'TGG',
    'CAA', 'CAT', 'CAC', 'CAG',
    'CTA', 'CTT', 'CTC', 'CTG',
    'CCA', 'CCT', 'CCC', 'CCG',
    'CGA', 'CGT', 'CGC', 'CGG',
    'GAA', 'GAT', 'GAC', 'GAG',
    'GTA', 'GTT', 'GTC', 'GTG',
    'GCA', 'GCT', 'GCC', 'GCG',
    'GGA', 'GGT', 'GGC', 'GGG',
]


# MOTIFS for PosWeightProfiler - these are all going to have scores of 1
POLY_TRACTS = ['A{6,}', 'C{6,}', 'G{6,}', 'T{6,}', '(?:AT){4,}']
# poly tracts
# AT dinucleotide repeat - prone to slippage

# Nuclear Localization Signal
NLS = ['[KR]{4,}', 'K[KR].[KR]',   # Monopartite class 1 and 2 consensus
       '[KR]{2}.{10,12}(?:[KR]{3}[A-Z]{2}|[KR]{4}[A-Z]|[KR]{5})',  # Bipartite
       'KR.[WFY]..AF', '[PR]..KR[KR]',  # Noncanonical importin a, class 3 and 4
       'R{3,}', '[KRH].{2,5}PY']  # Noncanonical importin b

# Nuclear Export Signal
NES = ['L.{2,3}[LIVFM].{2,3}L.[LI]']

INTRON = "GT[ATCG]{1,10}TACTAAC[ATCG]{1,10}AC"


