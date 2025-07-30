from a01_KeyStone import KeyStone
from a03_LookingGlass import LookingGlass
from a04_ReGen import ReGen


def keystone_p1_demo(ml_modelname='ReGen_v2'):
    test = KeyStone(ml_modelname)
    test.naive_dataframe()
    test.decompress_genome()
    test.context_dataframe()


def keystone_p2_demo(ml_modelname='ReGen_v2'):
    # this process takes around 1 hour 50 generally depending on background tasks
    # [Generating DNA mutation fingerprints]: 100%|██████████| 378862/378862 [13:19<00:00, 473.69it/s]
    # [Generating AA chain mutation fingerprints]: 100%|██████████| 378862/378862 [1:30:27<00:00, 69.81it/s]
    test = KeyStone(ml_modelname)

    try:
        if __name__ == "__main__":
            success = test.generate_fp()
            if success:
                print("[[Fingerprint DataFrame Generation Completed]]")
    except Exception as e:
        print(f"Mutation Fingerprint Generation Failed: {e}")
        raise


def keystone_p3_demo(ml_modelname='ReGen_v2'):
    test = KeyStone(ml_modelname)
    test.train_models()


def LookingGlass_Demo(fasta_filename='test.fasta', ml_model='ReGen_v2', output_filename='Screen_test_1'):
    test_module = LookingGlass(fasta_filename, ml_model)
    if __name__ == "__main__":
        test_module.predict_file(output_filename)


def Repair_Gene(pathogenic_gene_file='benchmark_fasta', ml_model='ReGen_v2', outfile_name='benchmark_repair_test'):
    if __name__ == "__main__":
        module = ReGen(pathogenic_gene_file, ml_model, outfile_name)
        module.repair()


Repair_Gene()