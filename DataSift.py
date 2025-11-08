from pathlib import Path
import pandas as pd
import numpy as np
import json

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone


# ultimate goal - improve model performance and training time via selecting optimal features
# to maximize performance and training efficiency

# base XGB - # straightforward stratified kfold cross validation
# preprocesses dataframe for low variance features + sequentially prunes the least important features
# monitors model performance after stratified k-fold cv and...
# returns feature list that brought about the greatest model performance

# Enhanced Feature saving with Metadata - model names, timestamps, n_features, features
#   - allow users to access the features via function + model name, this will be transferable to future projects
#   - feature configs?



class DataSift:
    def __init__(self, classifier_name,
                 classifier,
                 dataframe, y_label, label_map,
                 variance_space=None,
                 optimize_variance=True,
                 max_runs=None,
                 test_size=0.2,
                 random_state=42,
                 cv_splits=10,
                 patience=3):


        # Binary Variance Search settings
        if variance_space is None:
            variance_space = [0, 0.3]  # defaults
        self.model_name = classifier_name
        # intialize config savepath if not already
        self.config_path = Path("DataSift_configs") / f"{self.model_name}.json"
        self.config_path.parent.mkdir(parents = True, exist_ok=True)

        self.optimize_variance = optimize_variance
        self.max_runs = max_runs
        if self.max_runs is None:
            self.max_runs = 10


        # model + data input
        self.classifier = classifier
        self.dataframe = dataframe
        self.y_label = y_label
        self.label_map = label_map

        # setup variables
        self.variance_space = variance_space
        if self.variance_space is None:
            self.variance_space = [0.0, 0.3]

        # model training variables
        self.test_size = test_size
        self.random_state = random_state
        self.cv_splits = cv_splits
        self.patience = patience

        # train test data
        self.X_train = None
        self.y_train = None

        # stratified cv
        self.skf_cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)

        # baseline metrics
        self.pr_auc = 0
        self.roc_auc = 0
        self.pathogenic_f1 = 0
        self.base_composite = self.pr_auc + self.roc_auc + self.pathogenic_f1

        self.optimal_variance = 0


    def VarianSift(self):
        """
        Eliminates features according to variance - guided by a binary search-style algorithm
        :return:
        """
        X = self.dataframe.drop(self.y_label, axis=1)
        y = self.dataframe[self.y_label]
        X = X.apply(pd.to_numeric, errors='coerce')
        y = y.map(self.label_map)

        # variance filtering - columns with a variance below threshold -> drop
        feature_variances = X.var()

        # split into train and test for binary variance space search
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            stratify=y,
                                                            random_state=self.random_state)

        if self.optimize_variance:
            print(f"Variance Optimization Status: {self.optimize_variance} \nInitializing Variance Sift...")
            optimal_var_threshold = self.binary_variance_search(feature_variances, X_train, y_train)
        else:
            print(f"Variance Optimization Status: {self.optimize_variance} -- Using default threshold {self.variance_space[0]}")
            optimal_var_threshold = self.variance_space[0]


        low_variance_features = feature_variances[feature_variances < optimal_var_threshold].index
        X_train_filtered = X_train.drop(columns=low_variance_features)

        self.X_train = X_train_filtered
        self.y_train = y_train

        self.optimal_variance = optimal_var_threshold

        return self.X_train, self.y_train


    def binary_variance_search(self, feature_variances, X_train, y_train):
        low = self.variance_space[0]
        high = self.variance_space[1]

        retain_counter = 0

        iteration = 0

        best_threshold = low
        best_composite = -np.inf

        history = []

        print(f"Variance Sift Search Initialized: searching between {low} and {high}")

        while iteration < self.max_runs and retain_counter < self.patience:
            iteration += 1

            mid = (low + high) / 2

            step = max(0.01, (high-low) / 10)

            left_shift = max(low, mid-step)
            right_shift = max(high, mid+step)
            # less - variance   (left) [low ----- mid ----- high] (right)   more variance

            # get score composites
            base_comp = self.binary_test(mid, feature_variances, X_train, y_train)
            left_comp = self.binary_test(left_shift, feature_variances, X_train, y_train)
            right_comp = self.binary_test(right_shift, feature_variances, X_train, y_train)

            # track the highest score
            for threshold, composite in [(mid, base_comp), (left_shift, left_comp), (right_shift, right_comp)]:
                if composite > best_composite:
                    best_composite = composite
                    best_threshold = threshold

            history.append((mid, base_comp))

            print(f"VarSift Iteration {iteration} | Best Score: {best_composite} | Best Threshold: {best_threshold}")

            # decide search direction
            if base_comp >= left_comp and base_comp >= right_comp:
                if left_comp > right_comp:
                    high = mid
                    print(f"Moving upper bound to {mid}")
                else:
                    low = mid
                    print(f"Moving lower bound to {mid}")
            elif left_comp > base_comp:
                high = mid
                print(f"Moving upper bound to {mid}")
            else:
                low = mid
                print(f"Moving lower bound to {mid}")

            if len(history) >= 3:
                recent_scores = [tup[1] for tup in history [-3:]]
                if max(recent_scores) - min(recent_scores) < 0.005:
                    print(f"Plateau - stopping early")
                    return best_threshold

        return best_threshold


    def binary_test(self, variance, feature_variances, X_train, y_train):
        columns = feature_variances[feature_variances >= variance].index
        test_X = X_train[columns]
        a_roc, a_pr, a_f1 = self.cross_validation(test_X, y_train)
        composite = a_roc + a_pr + a_f1

        return composite


    def Data_Sift(self):
        """
        Processes dataframe by variances then uses a iterates backwards, eliminating the least important features sequentially until performance wanes -> most efficient feature combination
        :return:
        """
        self.VarianSift()
        best_features = self.FeatureSift()

        if best_features:
            self.save_feature_config(best_features)
        else:
            self.save_feature_config(feature_list)

        return True

    def FeatureSift(self):
        """
        Utilizes backward sequencing to determine optimal feature combination for model performance (hopefully)
        Eliminates the least important ones sequentially
        :return:
        """
        print(f"Feature Sift Initialized")
        # now we have base evaluation metrics and a feature dataframe with increasing importance values
        base_roc, base_prc, base_f1, FeatImp_df = self.importance_df()
        base_composite = base_roc + base_prc + base_f1
        feature_list = FeatImp_df['Feature'].to_list()

        best_features = []
        best_roc = base_roc
        best_prc = base_prc
        best_f1 = base_roc
        best_composite = base_composite

        best_idx = 0
        early_stop_counter = 0


        for feature_removal_count in range(len(feature_list)):
            new_feature_list = feature_list[feature_removal_count + 1:]  # we do +1 because we already did the first round as base
            X_t_new, y_t_new = self.X_train[new_feature_list], self.y_train

            # get metrics from new list + generate new composite score
            new_roc, new_prc, new_f1 = self.cross_validation(X_t_new, y_t_new)
            new_composite = new_roc + new_prc + new_f1

            # first need to check if nothing has decreased by more than 1%
            if ((new_composite <= best_composite - 0.01) or
                    (new_roc < best_roc - 0.01) or
                    (new_prc <= best_prc - 0.01) or
                    (new_f1 <= best_f1 - 0.01)):
                print(f"[|Performance break encountered|]: returning optimal feature list with the following metrics"
                      f"Composite: {new_composite} | ROC_AUC {new_roc} | PR_AUC {new_prc} | F1 {new_f1}")
                break
            elif new_composite >= best_composite:  # check if it is better than the best stats and overwrite them
                best_features = new_feature_list
                best_roc = new_roc
                best_prc = new_prc
                best_f1 = new_f1
                best_composite = new_composite
                best_idx = feature_removal_count
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # status check per iteration
            print(f"FeatureSift Iteration [{feature_removal_count}] Composite: {new_composite:.5f} | ROC_AUC {new_roc:.5f} | PR_AUC {new_prc:.5f} | F1 {new_f1:.5f} | Best_idx = {best_idx}")

            if best_features and early_stop_counter == self.patience:
                print(f"[|Patience threshold exceeded|]: breaking early with the following stats & returning optimal feature list:"
                      f"[{best_idx}] Composite: {best_composite} | ROC_AUC {best_roc} | PR_AUC {best_prc} | F1 {best_f1}")
                break

        if best_features:
            return best_features
        else:
            return feature_list

    def importance_df(self):
        """
        :return: Feature importance dataframe for feature refinement loop + starting stats
        0: average_roc -> 1: average_prc -> 2: average_f1 -> 3: feature importance df
        """
        return self.cross_validation(self.X_train, self.y_train, importance_flag=True)

    def cross_validation(self, X_train, y_train, importance_flag=False):
        """
        Performs stratified cv + outputs roc & pr AUCs, & f1 scores <- evaluation metrics for model performance
        \nIf importance_flag is on - will output feature importance dataframe, do this once for the setup
        :return: 0: average_roc -> 1: average_prc -> 2: average_f1
        """
        model = clone(self.classifier)
        roc_scores, pr_scores, f1_scores = [], [], []
        importance_accumulator = np.zeros(X_train.shape[1])

        # k-fold cross validation
        for train_idx, val_idx in self.skf_cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_tr, y_tr)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba, pos_label=1)

            # evaluation metrics
            f1_set = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_f1 = np.max(f1_set)
            f1_scores.append(best_f1)
            roc_scores.append(roc_auc_score(y_val, y_pred_proba))
            pr_scores.append(auc(recall, precision))

            # add importances for downstream average calculation
            importance_accumulator += model.feature_importances_

        # get overall metrics
        average_roc = np.mean(roc_scores)
        average_prc = np.mean(pr_scores)
        average_f1 = np.mean(f1_scores)

        if not importance_flag:
            return average_roc, average_prc, average_f1
        else:
            # average importances utilized - different folds give different importance values
            feature_importances = importance_accumulator / self.cv_splits
            featimp_df = pd.DataFrame({
                "Feature": self.X_train.columns,
                "Importance": feature_importances,
            })
            featimp_df = featimp_df.sort_values(by="Importance", ascending=True)
            return average_roc, average_prc, average_f1, featimp_df


    def save_feature_config(self, features):
        from datetime import datetime

        config = {
            'ModelName': self.model_name,
            'Time': datetime.now().isoformat(),
            'Optimal_var_threshold': self.optimal_variance,
            'n_features': len(features),
            'features': features,
        }

        # json save configs
        with open(self.config_path, 'w') as outfile:
            json.dump(config, outfile, indent=2)

        print(f"{len(features)} features saved to {self.config_path}")
        return self.config_path



class SiftControl:
    def __init__(self):
        self.folder_path = Path("DataSift_configs")
        self.filepath = None
        self.config = None

    def LoadConfig(self, model_name):
        self.filepath = self.folder_path / f"{model_name}.json"
        try:
            file_size = self.filepath.stat().st_size
            print(f"{file_size}")
        except FileNotFoundError:
            print(f"Error: File not found at {self.filepath}")

        with open(self.filepath, 'r') as confile:
            self.config = json.load(confile)

        return self.config

    def check(self):
        if self.config is None:
            raise FileNotFoundError(f"Config not detected: call LoadConfig before utilizing Config-dependent functions")
        else:
            return True

    def LoadSift(self):
        if self.check:
            return self.config['features']
        else:
            raise ValueError("Config diagnostic failed")

    def SiftData(self, X_dataframe):
        features = self.LoadSift

        missing_features = set(features) - set(X_dataframe.columns)
        if missing_features:
            raise ValueError(f"Missing {len(missing_features)}: {list(missing_features)}")

        # extra features are fine, the Sift will ignore it anyway
        return X_dataframe[features]

