import gzip
import pickle as pkl
from pyfaidx import Fasta

from a02_1_CompositeDNA_Toolkit import *
from a02_2_CompositeProt_Toolkit import *

from DataSift import DataSift
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay



class KeyStone:
    """
    Access and clean data from variant_summary.txt.gz file
    Process this file and the DNA sequences
    Generate a DNA fingerprint with various statistics
    Train ML model on DNA fingerprint and ClinicalSignificance labels as 'y'
    """

    def __init__(self, model_name):
        self.root = Path(__file__).parent.resolve()
        with open(self.root / 'config.yaml', 'r') as outfile:
            self.cfg = yaml.safe_load(outfile)

        # Navigation
        self.clinvar_data = Path(self.cfg['database_folder']) / self.cfg['clinvar_data']
        self.genome_gz = Path(self.cfg['database_folder']) / self.cfg['GRCh38_gz']
        self.genome_decomp = Path(self.cfg['database_folder']) / self.cfg['GRCh38_fna']

        self.naivefile_df_outpath = Path('database') / f"{self.cfg['ref_alt_df']}.csv"
        self.context_df_outpath = Path('database') / f"{self.cfg['context_df']}.pkl"

        # final dataframe for model training
        self.final_df_path = Path('database') / f"{self.cfg['full_variant_df']}.pkl"


        # model storage
        self.model_name = model_name
        self.model_storage = Path(self.cfg['model_folder'])
        self.model_path = self.model_storage / f"{self.model_name}"

        # optimal features
        self.optimal_features = []

       # create full path in one go with parents=True
        self.model_path.mkdir(parents=True, exist_ok=True)

        # dataframe settings
        self.flanksize = self.cfg['flank_size']


    # ====[FILTER DATASET]====
    # Gets ClinVar data variants and filters them to get confident classifications of Benign and Pathogenic Variants
    # After more data cleaning we align the variants with its surrounding context of +500bp both sides (configurable)
    # Now we have a dataframe with 500bp context-elaborated variants

    def naive_dataframe(self):
        """
        Generates a balanced and shuffle dataframe containing molecular fingerprint-like data characterizing
        various changes in DNA sequences resulting from mutations
        :return: REF_ALT_df.csv numerically characterizes DNA alignment data, structural and mutation changes
        """
        with gzip.open(self.clinvar_data, "rt") as outfile:
            df = pd.read_csv(outfile, sep="\t", low_memory=False)

        # Columns of interest
        # Clinical Significance, REF / ALT alleles, positions, Chromosome
        # First, filter out the clinical significance columns
        model_columns = ['Assembly', 'Chromosome', 'ChromosomeAccession', 'PositionVCF', 'ReferenceAlleleVCF', 'AlternateAlleleVCF', 'ClinicalSignificance']

        df2 = df[model_columns]

        # drop rows that have 'na' or NaN in any column
        df3 = df2.dropna()
        no_na_df = df3[~df3.map(lambda x: isinstance(x, str) and x.lower() == 'na').any(axis=1)]

        # drop Un and Mt chromosomes + filter for GRCh38 assembly genomes - keep data quality consistent
        no_MTUn_df = no_na_df[
            (~no_na_df['Chromosome'].str.contains('MT')) &
            (~no_na_df['Chromosome'].str.contains('Un')) &
            (~no_na_df['Assembly'].str.contains('GRCh37'))
        ]

        # drop clinical significance ratings like uncertain, conflicting .etc
        clinical_filter = ["Pathogenic", "Benign"]
        binary_clinical_df = no_MTUn_df[no_MTUn_df.ClinicalSignificance.isin(clinical_filter)]

        # save dataframe
        # 378862 rows of GRCh38 assembly gene varant data rows
        binary_clinical_df.to_csv(self.naivefile_df_outpath, index=False)

    def decompress_genome(self):
        with gzip.open(self.genome_gz, 'rb') as f_in, open(self.genome_decomp, 'wb') as f_out:
            print("decompressing")
            f_out.write(f_in.read())
            print(f"decompressed to {self.genome_decomp}")

    def context_dataframe(self):
        """
        Processes the variant data and queries a separate data file for the surrounding nucleotides (known as context)
        The size of context is configurable in config
        :return:
        """
        df = pd.read_csv(self.naivefile_df_outpath)
        genome = Fasta(self.genome_decomp)

        new_df = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Building {self.flanksize} Flanks on ClinVar Variant Alleles..."):
            # obtain local strings
            chromosome = row['Chromosome']
            chromosome_id = row['ChromosomeAccession']
            pos = row['PositionVCF']   # this is where the mutation starts
            ref_allele = row['ReferenceAlleleVCF']
            alt_allele = row['AlternateAlleleVCF']
            clinsig = row['ClinicalSignificance']

            chromosome_length = len(genome[chromosome_id])

            # define whole window - bound checks
            flank_start = max(1, pos - self.flanksize)
            flank_end = min(chromosome_length, pos + len(ref_allele) + self.flanksize -1)

            # define the flank windows - bound checks
            flank1_start = flank_start
            flank1_end = max(flank_start, pos - 1)  # -1 to not include variant
            flank2_start = min(chromosome_length, pos + len(ref_allele))
            flank2_end = flank_end

            # ensure the ranges are valid
            if flank1_end < flank1_start:
                flank1_end = flank1_start - 1  # create an empty flank1
            if flank2_start > flank2_end:
                flank2_start = flank2_end + 1


            try:
                ref_context_seq = str(genome[chromosome_id][flank_start -1: flank_end].seq).upper()
            except KeyError:
                print(f"Chromosome {chromosome} | {chromosome_id} not found in FASTA. Skipping")
                continue

            # calculate positions of variant within context
            variant_relative_start = pos - flank_start
            variant_relative_end = variant_relative_start + len(ref_allele)


            # sanity check - check if extracted reference matches provided allele
            extracted_ref = ref_context_seq[variant_relative_start:variant_relative_end]
            if extracted_ref != ref_allele:
                print(f"Warning: Ref allele mismatch at {chromosome_id}: {pos}")

            flank1 = str(genome[chromosome_id][flank1_start - 1: flank1_end])
            flank2 = str(genome[chromosome_id][flank2_start - 1: flank2_end])

            new_df.append({
                "Chromosome": chromosome,
                "ReferenceAlleleVCF": ref_allele,
                "AlternateAlleleVCF": alt_allele,
                "Flank_1": flank1.upper(),
                "Flank_2": flank2.upper(),
                "ClinicalSignificance": clinsig
            })

        # make the new dataframe
        new_df = pd.DataFrame(new_df)

        # convert chromosome to numeric values: chromosome is categorical but if it is a mix a numbers and strings, we handle it as separate categories
        new_df.loc[:, 'Chromosome'] = new_df['Chromosome'].apply(lambda x:
                                                                                         23 if x == 'X' else
                                                                                         24 if x == 'Y' else
                                                                                         int(x))

        with open(self.context_df_outpath, 'wb') as outfile:
            pkl.dump(new_df, outfile)


    # ====[FEATURE ENGINEERING --> GENERATE SHORT AND STRUCTURAL VARIANT DATAFRAMES]===
    # pass data on to my two 1000+ line classes for feature extraction and engineering
    # one analyzes the DNA sequences and the other the most-likely translated protein
    # concatenate the DNA and Prot dataframe for each variant type
    # One dataframe for short variants and one for structural variants
    # Now we can train our models
    def generate_fp(self):
        """
        Builds dataframe for variants -> DNA and AA profiles at once
        This process will take a fairly long time - I am trying to implement multicore optimizations,
        it's just a little tricky rn with the class variables holding some crucial elements
        :return: Dataframe in database//VARIANT_df.pkl
        """
        with open(self.context_df_outpath, 'rb') as infile:
            df = pkl.load(infile)

        # note: modules support multiprocessing context managers
        # ==[DNA data]==
        # 12:35 to 13:20, dropped from 1.5 hours
        with CompositeDNA() as dna_module:
            dna_df = dna_module.gen_DNAfp_dataframe(df)

        # ==[Protein data]==
        # 1.5 hours, dropped from a prospective 7-9 hours
        with CompositeProt() as prot_module:
            prot_df = prot_module.gen_AAfp_dataframe(df)



        # [[Save DataFrame]]
        # ensure alignment - in case something goes wrong with one of them
        dna_df.index = df.index
        prot_df.index= df.index

        variant_final_df = pd.concat([dna_df, prot_df], axis=1)
        with open(self.final_df_path, 'wb') as outfile:
            pkl.dump(variant_final_df, outfile)
        return True


    def train_models(self):
        """
        Call on this to train the models
        1) Feature optimization
        2) Hyperparameter optimization
        3) Model saving
        """
        # load data and train models
        with open(self.final_df_path, 'rb') as infile:
            variant_dataframe = pkl.load(infile)

        self.optimized_model(variant_dataframe)


    def optimized_model(self, df):
        y_label = 'ClinicalSignificance'
        X = df.drop(y_label, axis=1)
        y = df[y_label]

        X = X.apply(pd.to_numeric, errors= 'coerce')

        label_map = {'Benign': 0, 'Pathogenic': 1}
        y = y.map(label_map)

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size = 0.2,
                                                            stratify=y,
                                                            random_state=42)

        # == Hyperparameter Optimization ==
        # class_counts = y.value_counts()
        # scale_pos_weight = class_counts[0] / class_counts[1]
        # study = optuna.create_study(direction='maximize')
        # study.optimize(lambda trial: self.objective(trial, X_train, y_train, scale_pos_weight), n_trials=100)
        # best_params = study.best_params

        best_params = {'n_estimators': 1936,
                       'max_depth': 10,
                       'learning_rate': 0.041977875319094894,
                       'subsample': 0.8691093047813849,
                       'colsample_bytree': 0.9973783186852718,
                       'reg_alpha': 0.20907871533405323,
                       'reg_lambda': 1.6124970064334614,
                       'gamma': 0.35865668074613577,
                       'scale_pos_weight': 1.8996291716997067}

        self.evaluate_save(best_params, X_train, y_train, X_test, y_test)

    def evaluate_save(self, parameters, X_train, y_train, X_test, y_test):
        import os
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score

        print(f"Optimal Hyperparameters: {parameters}")

        strat_fold = StratifiedKFold(n_splits = 5, shuffle=True)

        model = XGBClassifier(**parameters)

        roc_scores, pr_scores, fn_counts, fp_counts = [], [], [], []

        # k-fold cross validation
        for train_idx, val_idx in strat_fold.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_tr, y_tr)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba, pos_label=1)

            f1_scores = 2 * (precision * recall) / (precision + recall)

            optimal_idx = np.argmax(f1_scores)
            optimal_t = thresholds[optimal_idx]
            y_pred_optimal = (y_pred_proba >= optimal_t).astype(int)
            cm = confusion_matrix(y_val, y_pred_optimal, labels=[0, 1])

            fn_counts.append(cm[1, 0])
            fp_counts.append(cm[0, 1])
            roc_scores.append(roc_auc_score(y_val, y_pred_proba))
            pr_scores.append(auc(recall, precision))

        print(f"Cross Validation Results: Mean ROC AUC: {np.mean(roc_scores):.4f}, Mean PR AUC: {np.mean(pr_scores):.4f}")
        print(f"Mean FNs: {np.mean(fn_counts):.2f}, Mean FPs: {np.mean(fp_counts):.2f}")

        # Train on full data and evaluate on test set
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate F1 specifically for pathogenic class
        pathogenic_f1 = f1_score(y_test, (y_pred_proba >= 0.5).astype(int), pos_label=1)

        # AUC cores
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba, pos_label=1)
        pr_auc = auc(recall, precision)

        # Find optimal threshold for pathogenic detection
        f1_scores = 2 * (precision * recall) / (precision + recall)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]

        # Apply optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)


        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Precision-Recall AUC: {pr_auc:.4f}")
        print(f"Pathogenic F1-Score: {pathogenic_f1:.4f}")
        print(f"Optimal threshold for pathogenic detection: {optimal_threshold:.3f}")

        print("Performance with optimal threshold:")
        print(classification_report(y_test, y_pred_optimal))

        cm = confusion_matrix(y_test, y_pred_optimal, labels=[0, 1])
        print("Confusion Matrix:")
        print(cm)

        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Pathogenic'])
        display.plot()

        savepath = self.model_path / f"{self.model_name}.pkl"
        if os.path.exists(savepath):
            print(f"Model {self.model_name} already exists. Skipping save to prevent overwrite")
        else:
            with open(savepath, 'wb') as outpath:
                pkl.dump(model, outpath)


    # OPTUNA HYPERPARAMETER OPTIMIZATION
    @staticmethod
    def objective(trial, X_train, y_train, scale_pos_weight):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5 * scale_pos_weight, 2 * scale_pos_weight),
            'objective': 'binary:logistic',
            'eval_metric': 'mlogloss',
            'n_jobs': -1,
            'random_state': 42
        }

        model = XGBClassifier(**params)

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='average_precision')
        return scores.mean()


    def extract_first5(self):
        with open(self.context_df_outpath, 'rb') as infile:
            df = pkl.load(infile)

        print(df.head(5).to_string())







