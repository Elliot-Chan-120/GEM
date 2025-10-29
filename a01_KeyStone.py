import gzip
import pickle as pkl
from pyfaidx import Fasta

from a02_1_CompositeDNA_Toolkit import *
from a02_2_CompositeProt_Toolkit import *
from a02_3_DNAMatrix_Toolkit import *
from a02_4_ProtMatrix_Toolkit import *

import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score, precision_recall_curve, auc, f1_score



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
        self.database = Path(self.cfg['database_folder'])  # main database folder
        self.datacore = self.database / self.cfg['datacore_folder']
        self.clinvar_data = self.datacore / self.cfg['clinvar_data']
        self.genome_gz = self.datacore / self.cfg['GRCh38_gz']
        self.genome_decomp = self.datacore / self.cfg['GRCh38_fna']

        self.naivefile_df_outpath = self.database / f"{self.cfg['ref_alt_df']}.csv"
        self.context_df_outpath = self.database / f"{self.cfg['context_df']}.pkl"

        # dataframe saves
        self.composite_dataframe = self.database / f"{self.cfg['composite_df']}.pkl"
        self.dna_profile_df = self.database / f"{self.cfg['dna_profile']}.pkl"
        self.prot_profile_df = self.database / f"{self.cfg['prot_profile']}.pkl"

        self.dna_pwm_profile_df = self.database / f"{self.cfg['dna_pwm_profile']}.pkl"
        self.aa_pwm_profile_df = self.database / f"{self.cfg['aa_pwm_profile']}.pkl"

        self.hmm_profile_df = self.database / f"{self.cfg['hmm_profile']}.pkl"

        # final dataframe for model training
        self.final_df_path = self.database / f"{self.cfg['full_variant_df']}.pkl"


        # model storage
        self.model_name = model_name
        self.model_storage = Path(self.cfg['model_folder'])
        self.model_path = self.model_storage / f"{self.model_name}"


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
                "ReferenceAlleleVCF": ref_allele.upper(),
                "AlternateAlleleVCF": alt_allele.upper(),
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

    # [1] - extract most probable protein sequences for downstream biochemical analysis and motif scanning
    def protein_extraction(self):
        with open(self.context_df_outpath, 'rb') as infile:
            df = pkl.load(infile)
            # load up default dataframe, pass it to composite prot to extract all the protein sequences first

        with CompositeProt() as prot_module:
            composite_df = prot_module.gen_AAseqs(df)

        with open(self.composite_dataframe, 'wb') as outfile:
            pkl.dump(composite_df, outfile)

        print(composite_df.columns)
        return True

    # [2]
    def generate_dna_profile(self):
        """
        Builds dataframe for variants -> DNA and AA profiles at once
        This process will take a fairly long time - I am trying to implement multicore optimizations,
        it's just a little tricky rn with the class variables holding some crucial elements
        :return: Dataframe in database//VARIANT_df.pkl
        """
        with open(self.composite_dataframe, 'rb') as infile:
            df = pkl.load(infile)

        # note: modules support multiprocessing context managers
        # ==[DNA data]==
        # 12:35 to 13:20, dropped from 1.5 hours
        with CompositeDNA() as dna_module:
            dna_df = dna_module.gen_DNAfp_dataframe(df)
        dna_df.index = df.index  # ensure alignment for downstream processing

        with open(self.dna_profile_df, 'wb') as outfile:
            pkl.dump(dna_df, outfile)
        return True

    # [3]
    def generate_prot_profile(self):
        with open(self.composite_dataframe, 'rb') as infile:
            df = pkl.load(infile)

        # ==[Protein data]==
        # 30 minutes
        with CompositeProt() as prot_module:
            prot_df = prot_module.gen_AAfp_dataframe(df)
        prot_df.index = df.index

        with open(self.prot_profile_df, 'wb') as outfile:
            pkl.dump(prot_df, outfile)
        return True

    # [4]
    def generate_dnapwm_profile(self):
        with open(self.composite_dataframe, 'rb') as infile:
            df = pkl.load(infile)

        # managed to optimize processing time from 3-4 hours to 20-30 minutes with 4x the motifs...
        with DNAMatrix() as dna_pwm_module:
            dna_pwm_df = dna_pwm_module.gen_DNAPWM_dataframe(df)
        dna_pwm_df.index = df.index  # ensure index alignment

        with open(self.dna_pwm_profile_df, 'wb') as outfile:
            pkl.dump(dna_pwm_df, outfile)
        return True


    def generate_aapwm_profile(self):
        with open(self.composite_dataframe, 'rb') as infile:
            df = pkl.load(infile)

        # managed to optimize processing time from 3-4 hours to 20-30 minutes with 4x the motifs...
        with ProtMatrix() as aa_pwm_module:
            aa_pwm_df = aa_pwm_module.gen_AAPWM_dataframe(df)
        aa_pwm_df.index = df.index  # ensure index alignment

        with open(self.aa_pwm_profile_df, 'wb') as outfile:
            pkl.dump(aa_pwm_df, outfile)
        return True


    def generate_hmm_profile(self):
        with open(self.composite_dataframe, 'rb') as infile:
            df = pkl.load(infile)

        with CompositeDNA() as dna_pwm_module:
            hmm_df = dna_pwm_module.gen_HMM_dataframe(df)
        hmm_df.index = df.index

        with open(self.hmm_profile_df, 'wb') as outfile:
            pkl.dump(hmm_df, outfile)

        return True


    def get_final_dataframe(self):
        filepaths = [self.dna_profile_df, self.prot_profile_df,
                     self.dna_pwm_profile_df, self.aa_pwm_profile_df,
                     self.hmm_profile_df]

        for p in filepaths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(f"File {path} not created yet")
            if path.stat().st_size == 0:
                raise ValueError(f"File {path} is empty")

        # Remember this order for LookingGlass and ReGen
        # dna, prot, dnapwm, protpwm
        dfs = [
            pd.read_pickle(self.dna_profile_df),
            pd.read_pickle(self.prot_profile_df),
            pd.read_pickle(self.dna_pwm_profile_df),
            pd.read_pickle(self.aa_pwm_profile_df),
            pd.read_pickle(self.hmm_profile_df)
        ]

        variant_final_df = pd.concat(dfs, axis=1)
        variant_final_df.to_pickle(self.final_df_path)
        return True


    def train_models(self):
        """
        Call on this to train the models
        1) Feature optimization -> not possible yet still need to tweak DataSift before using it in this pipeline
        2) Hyperparameter optimization
        3) Model saving
        """

        # load data and train models
        with open(self.final_df_path, 'rb') as infile:
            variant_dataframe = pkl.load(infile)

        useless_columns = ['ref_protein_list', 'alt_protein_list',
                           'non_ambiguous_ref', 'non_ambiguous_alt',
                           'ref_protein_length', 'alt_protein_length']

        variant_dataframe = variant_dataframe.drop(useless_columns, axis=1)

        self.optimized_model(variant_dataframe)


    def optimized_model(self, df):
        # from DataSift import DataSift
        y_label = 'ClinicalSignificance'
        df = df.loc[:, ~df.columns.duplicated()]

        X = df.drop(y_label, axis=1)
        y = df[y_label]

        for label in X.columns:
            print(label)

        X = X.apply(pd.to_numeric, errors= 'coerce')

        label_map = {'Benign': 0, 'Pathogenic': 1}

        y = y.map(label_map)

        # Hopefully I figure out how to make this actually work well later
        # refined_features = DataSift(XGBClassifier(),
        #                             df,
        #                             y_label,
        #                             label_map).d_sift()

        # X = X[refined_features]

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size = 0.2,
                                                            stratify=y,
                                                            random_state=42)

        # == Hyperparameter Optimization ==
        # tuner = optuna.samplers.TPESampler(n_startup_trials=50, seed=42)  # will learn from previous trials
        # pruner = optuna.pruners.MedianPruner(n_startup_trials=30, n_warmup_steps=10)
        #
        # class_counts = y.value_counts()
        # scale_pos_weight = class_counts[0] / class_counts[1]
        #
        # study = optuna.create_study(direction='maximize',
        #                             sampler = tuner,
        #                             pruner = pruner)
        #
        # study.optimize(lambda trial: self.objective(trial, X_train, y_train, scale_pos_weight), n_trials=175)
        # best_params = study.best_params

        best_params = {'n_estimators': 1674,
                       'max_depth': 10,
                       'learning_rate': 0.034561112430304776,
                       'subsample': 0.9212141915845736,
                       'colsample_bytree': 0.6016405698933265,
                       'colsample_bylevel': 0.9329109895929816,
                       'reg_alpha': 0.7001202050122113,
                       'reg_lambda': 3.1671750288760134,
                       'gamma': 1.0033930419124446,
                       'min_child_weight': 9,
                       'scale_pos_weight': 1.6075244983571118}

        self.evaluate_save(best_params, X_train, y_train, X_test, y_test)

    def evaluate_save(self, parameters, X_train, y_train, X_test, y_test):

        content = ""
        content += f"Optimal Hyperparameters: {parameters}\n"

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

        content += f"Cross Validation Results: Mean ROC AUC: {np.mean(roc_scores):.4f}, Mean PR AUC: {np.mean(pr_scores):.4f}\n"
        content += f"Mean FNs: {np.mean(fn_counts):.2f}, Mean FPs: {np.mean(fp_counts):.2f}\n"

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


        content += f"ROC AUC: {roc_auc:.4f}\n"
        content += f"Precision-Recall AUC: {pr_auc:.4f}\n"
        content += f"Pathogenic F1-Score: {pathogenic_f1:.4f}\n"
        content += f"Optimal threshold for pathogenic detection: {optimal_threshold:.3f}\n"

        content += "Performance with optimal threshold:\n"
        content += classification_report(y_test, y_pred_optimal)

        cm = confusion_matrix(y_test, y_pred_optimal, labels=[0, 1])
        content += "Confusion Matrix:\n"
        content += str(cm)
        content += "\n----------------------------------------------\n"

        feature_importances = model.feature_importances_
        featimp_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": feature_importances,
        })

        featimp_df = featimp_df.sort_values(by="Importance", ascending=True)

        for _, row in featimp_df.iterrows():
            content += f"{row['Feature']}: {row['Importance']:.4f}\n"

        savepath = self.model_path / f"{self.model_name}.pkl"
        statpath = self.model_path / f"{self.model_name}_stats.txt"
        statpath.write_text(content)
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
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight',
                                                    0.8 * scale_pos_weight,
                                                    2.0 * scale_pos_weight),
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'eval_metric': 'aucpr',
            'n_jobs': -1,
            'random_state': 42
        }

        model = XGBClassifier(**params)
        f1_scorer = make_scorer(f1_score, pos_label=1)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=f1_scorer)
        return scores.mean()


    def extract_first5(self):
        with open(self.context_df_outpath, 'rb') as infile:
            df = pkl.load(infile)

        print(df.head(5).to_string())

