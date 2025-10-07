from a01_KeyStone import KeyStone
from a03_LookingGlass import LookingGlass
from a04_ReGen import ReGen

model_name = "ReGen_v4"

# all functions with a name guard need to be called alone

# [1] sourced data extraction and processing
# - ensure the required files mentioned at the start are downloaded and in the database//datacore folder
def keystone_dataframe_processing(ml_modelname=model_name):
    test = KeyStone(ml_modelname)
    test.naive_dataframe()
    test.decompress_genome()
    test.context_dataframe()


# [2] feature extraction / engineering
# note: call each of these sequentially, they each use a multiproc. if name == "__main__" guard like so:
def keystone_extract_proteins(model = model_name):
    test = KeyStone(model)
    try:
        if __name__ == "__main__":
            success = test.protein_extraction()
            if success:
                print("[Protein Sequence Extraction Completed]")
    except Exception as e:
        print(f"Mutation Fingerprint Generation Failed: {e}")
        raise

def ks_dna_profile(model = model_name):
    # 12-13 minutes
    test = KeyStone(model)
    try:
        if __name__ == "__main__":
            success = test.generate_dna_profile()
            if success:
                print("[DNA Profile Completed]")
    except Exception as e:
        print(f"DNA Fingerprint Generation Failed: {e}")
        raise

def ks_prot_profile(model = model_name):
    test = KeyStone(model)
    try:
        if __name__ == "__main__":
            success = test.generate_prot_profile()
            if success:
                print("[Protein Profile Completed]")
    except Exception as e:
        print(f"Protein Fingerprint Generation Failed: {e}")
        raise

def ks_dnamotif_profile(model = model_name):
    # 1.5 hours
    test = KeyStone(model)
    try:
        if __name__ == "__main__":
            success  = test.generate_dnapwm_profile()
            if success:
                print("[PWM Profile Completed]")
    except Exception as e:
        print(f"PWM Fingerprint Generation Failed: {e}")
        raise

def ks_aamotif_profile(model = model_name):
    # 30 minutes
    test = KeyStone(model)
    try:
        if __name__ == "__main__":
            success  = test.generate_aapwm_profile()
            if success:
                print("[PWM Profile Completed]")
    except Exception as e:
        print(f"PWM Fingerprint Generation Failed: {e}")
        raise

# [3] Feature dataframe merging
def keystone_merge(ml_modelname=model_name):
    test = KeyStone(ml_modelname)

    try:
        if __name__ == "__main__":
            success = test.get_final_dataframe()
            if success:
                print("[Fingerprint DataFrame Generation Completed]")
    except Exception as e:
        print(f"Mutation Fingerprint Generation Failed: {e}")
        raise

# [4] Model training and optimization
def keystone_model_training(ml_modelname=model_name):
    # full run will take around 6 hours - latest optimized hyperparameters will be saved in model_name_stats.txt files
    test = KeyStone(ml_modelname)
    test.train_models()

def LookingGlass_Demo(fasta_filename='test.fasta', ml_model=model_name, output_filename='Screen_test_1'):
    test_module = LookingGlass(fasta_filename, ml_model)
    if __name__ == "__main__":
        test_module.predict_file(output_filename)

def Repair_Gene(pathogenic_gene_file='benchmark_fasta', ml_model=model_name, outfile_name='benchmark_repair_test'):
    if __name__ == "__main__":
        module = ReGen(pathogenic_gene_file, ml_model, outfile_name)
        module.repair()


# [ Command ]
Repair_Gene()

