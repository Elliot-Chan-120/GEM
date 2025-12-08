from src.gem.a01_KeyStone import KeyStone
from src.gem.a03_LookingGlass import LookingGlass
from src.gem.a04_ReGen import ReGen

model_name = "ClinicalModel"
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
    # 14 minutes
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


def ks_domain_profile(model = model_name):
    # 1:50 hours
    test = KeyStone(model)
    try:
        if __name__ == "__main__":
            success  = test.generate_hmm_profile()
            if success:
                print("[HMM-domain Profile Completed]")
    except Exception as e:
        print(f"HMM-domain Fingerprint Generation Failed: {e}")
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
                print("[FULL Mutation Fingerprint DataFrame Generation Completed]")
    except Exception as e:
        print(f"Mutation Fingerprint Generation Failed: {e}")
        raise

# [4] Model training and optimization
def keystone_model_training(ml_modelname=model_name):
    # full run will take around 6 hours - latest optimized hyperparameters will be saved in model_name_stats.txt files
    test = KeyStone(ml_modelname)
    test.train_models()

def LookingGlass_Demo(ml_model=model_name, genefile='video_test.fasta', output_filename='video_screen_test'):
    test_module = LookingGlass(ml_model, genefile)
    if __name__ == "__main__":
        test_module.predict_file(output_filename)

def Repair_Gene(ml_model=model_name, pathogenic_route='video_test.fasta', outfile_name='video_test_results'):
    if __name__ == "__main__":
        module = ReGen(ml_model, pathogenic_route, outfile_name)
        module.repair()

# [Commands]
LookingGlass_Demo()