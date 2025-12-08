from pathlib import Path

def get_project_root() -> Path:
    """
    Returns absolute path to GEM project root
    :return:
    """
    pack_root = Path(__file__).resolve().parent
    return pack_root.parent.parent

PROJECT_ROOT = get_project_root()
GEM_ROOT = PROJECT_ROOT / "src" / "gem"

# =====================
# 1st level directories
# =====================
DATABASE = PROJECT_ROOT / "database"
GENE_DATABANK = PROJECT_ROOT / "gene_databank"
REGEN_CANDIDATES = PROJECT_ROOT / "ReGen_candidates"
SCREEN_RESULTS = PROJECT_ROOT / "Screen_results"
DATASIFT_CONFIGS = PROJECT_ROOT / "DataSift_configs"
CONFIG = PROJECT_ROOT / "config.yaml"
MODEL_FOLDER = PROJECT_ROOT / "model_folder"

# =====================
# embedded directories
# =====================
DATACORE = DATABASE / "datacore"
PWM_DATABASE = DATABASE / "pwm_database"
DNA_MOTIFS = PWM_DATABASE / "DNA_motifs"
AA_MOTIFS = PWM_DATABASE / "AA_motifs"

# database filepaths
CLINVAR_DATA = DATACORE / "variant_summary.txt.gz"
GRCH38_GZ = DATACORE / "GCF_000001405.40_GRCh38.p14_genomic.fna.gz"
GRCH38_TXT = DATACORE / "GCF_000001405.40_GRCh38.p14_assembly_report.txt"
GRCH38_FNA = DATACORE / "GCF_000001405.29_GRCh38.p14_genomic.fna"

# pipeline data files
REF_ALT_DF = DATABASE / "REF_ALT_df.csv"
CONTEXT_DF = DATABASE / "CONTEXT_ref_alt_df.pkl"

COMPOSITE_DF = DATABASE / "CompositeDF.pkl"

DNA_PROFILE = DATABASE / "DNA_profile_df.pkl"
PROT_PROFILE = DATABASE / "PROT_profile_df.pkl"

DNA_PWM_PROFILE = DATABASE / "DNAPWM_profile_df.pkl"
AA_PWM_PROFILE = DATABASE / "AAPWM_profile_df.pkl"

HMM_PROFILE = DATABASE / "HMM_profile_df.pkl"

FULL_VARIANT_DF = DATABASE / "VARIANT_df.pkl"

# blosum matrix
BLOSUM62 = DATABASE / "blosum62.mat"





