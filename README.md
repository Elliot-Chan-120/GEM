# GEM
[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&color=5E81AC&width=435&height=40&lines=Benign+by+Design)](https://git.io/typing-svg)
- developed by: Elliot Chan
- contact: elliotchan120@gmail.com

> A sophisticated ML-enhanced end-to-end gene variant pathogenicity prediction and repair system.
> Aiming to accelerate gene variant classification and gene therapy target identification via intelligent, guided mutations.


## What is this GEM?
GEM (Gene Edit Machine) is a machine learning-powered platform designed to classify and optimize gene variants for reduced pathogenicity. 
What makes this a GEM?
- integrates ClinVar and GRCh38 human genome assembly data to build detailed, context-inclusive sequences
- uses biochemical data and research-backed feature extraction at both the DNA and protein level
- trains optimized XGBoost models to classify benign and pathogenic mutation variants
- suggests guided edits to reduce variant pathogenicity

It includes:
- *LookingGlass*: Variant classifier based on custom FASTA inputs
- *ReGen*: Performs a combination of guided edits to increase the likelihood of benign classification, with intelligently placed stochastic mutations to escape performance plateaus


## [1] Technical Peek into GEM's Architecture
**KeyStone: Integrating Database Querying, Bioinformatics Analyses and ML pipelines**

*Extracts genomic context, generates both DNA and Protein-based biochemical + structural features, and trains optimized models for clinical variant interpretation*
  - Data processing
    - Quality Filtering: removes uncertain classifications and focuses on high-confidence Benign/Pathogenic variants
    - Context Extraction: configurable flanking sequence windows around variants
  - Feature Engineering is done through classes CompositeDNA and CompositeProt, elaborated on below
  - Machine Learning Optimization and Evaluation
    - XGB classifiers undergo Optuna hyperparameter tuning (pre-computed optimal settings provided in pipeline)
    - Stratified k-fold validation, weighted sampling, threshold optimization
    - ROC-AUC, PR-AUC, F1-scores, Confusion Matrices, and Reports on trained model provided in model_folder

\
**CompositeDNA: High-performance comprehensive DNA Analysis**

*Conducts thorough DNA analyses to build a numerical profile on the following features:*
- Sequence Alignment and Comparison
  - Needleman-Wunsch global alignment, Smith-Waterman local alignment, Sequence identity and Coverage metrics
- Structural Analysis
  - Thermodynamic stability analysis using nearest-neighbour parameters
  - GC content and CpG island disruption analysis 
  - Shannon entropy for sequence complexity
  - Hairpin structure prediction
  - Microsatellite detection
- Mutation Detection
  - SNV (Single Nucleotide Variant) identification
  - Insertion / Deletion analysis with frameshift detection
  - Tandem duplication discovery
- Regulatory Element Analysis
  - Transcription factor motif disruptions
  - DNA instability motif analysis
  - Start/stop codon changes

\
**CompositeProt: High-performance comprehensive Amino-Acid Chain / Protein Analysis**

*Performs smart ORF detection to uncover the highest-probability protein sequence upon which the following features are extracted:*
- Translation and ORF Detection
  - Intelligent Open Reading Frame identification
  - Kozak motif sequence and ORF length-based scoring for translation efficiency
  - Bidirectional sequence analysis (accounts for reverse complement)
  - Ambiguous nucleotide handling
- Physicochemical Properties
  - Molecular weight calculations
  - Net charge at physiological pH
  - Isoelectric point determination
  - Aliphatic index
  - Protein half-life predictions
- Structural Analysis
  - Secondary structure propensity analysis (BioPython-aided)
  - Proline content assessment
  - Global and local protein sequence alignment with BLOSUM64 matrix
- Feature Interaction Engineering
  - Composite scores combining multiple properties
  - Interaction terms (e.g. charge-hydrophobicity)
  - Relative metrics normalized by protein length

\
**ReGen: Therapeutic Optimization**

*Utilizes a sophisticated algorithm system to optimize variants towards benign classification*
- Guided Evolution: intelligent sequence optimization using both guided and stochastic mutations
- Multi-copy evolution: configurable parallel optimization of multiple sequence variants for balance between exploration / exploitation
- Adaptive strategy: switches between guided and random mutations based on optimization progress
- Comprehensive tracking: monitors the best variants per iteration and threshold-crossing variants


## [1.1] Design Rationale
### Why XGB
XGB models were chosen due to being tree-based and faring well under non-linear high-dimensional data. I found them to be faster to train than RandomForest, speaking from (hours of) experience.

### Context Matters 
The problem with ClinVar's data on its own is that for a project like this, the ref and alt allele vcfs (variant sequences) alone are not sufficient to make accurate predictions of pathogenicity.
Surrounding context is highly significant in understanding the impact a mutation has. One simple insertion / deletion can shift a reading frame and drastically alter the resulting protein.
Therefore, cross-referencing ClinVar variant location data with GCF assembly and complete FASTA to obtain a configurable amount of flanking context sequences was the optimal choice. 

### Protein Analysis Strategy 
Protein analysis was a big hurdle, as I had find a way to balance processing power with biological accuracy. For every string of DNA (above a certain size), there are 6 ways to read and translate proteins from it, along what we call Open Reading Frames.
Naive approaches would involve brute-force scanning of all 6 reading frames, then scanning all possible ORFs, extracting all sequences, then conducting analyses on all of them then doing that again for variant comparison.
For 370k+ sequences of minimum length 1001 base pairs, this would nuke my computer. The solution to this would be to preprocess the entire sequence, scanning for start and stop codons.

Valid potential reading frames that started and ended with valid start and stop codons and had no in-frame stop codons inside were then saved, then scanned for a Kozak motif.
A Kozak motif generally looks like this (gccA/Gcc[AUG]G) around the [start codon] ATG / AUG (mRNA), where the capital letters represent the most important positions. It is a powerful predictor of translation initiation, so if you find it, there is a high chance protein translation will initiate immediately after.
Generally, the largest ORF is the one that will encode the protein. By searching for the Kozak motif and getting the length of all potential ORFs, we can generate a score for each, where the highest-scoring ORF is one we can safely assume will encode the protein.

### Flexible Multiprocessing
Both CompositeDNA and CompositeProt were optimized with multiprocessing by initializing a multiprocessing pool upon class instantiation, terminating automatically when under a context manager (with xyz as CompositeDNA/Prot(core_num=max_cores-2)).
This sped up protein analysis time from a predicted 8-9 hours to 1:50.
What makes them flexible is that pool persistence is encoded into them, allowing for them to be loaded up once, many dataframes passed onto them for analysis, then terminated at will rather than automatically via a context manager. 
This saved an immense amount of overhead in ReGen, as dataframes would be altered then rerun through their fingerprinting pipelines.

### ReGen's Adaptive Logic
A look at the config file or ReGen's logic will reveal a "retain_counter". What this is for is to keep track of the amount of times a mutation did not result in an increase in benignity. 
Once this passes a config-defined threshold (retain_threshold), the variant will undergo a certain amount of random mutations, and the highest-scoring one is allowed to pass on under a more lenient error threshold.
The threshold decreases as retain-count increases, becoming more lenient the more the variant demonstrates ridigity. This, combined with the stochastic mutations is designed to break through plateaus in performance, finding the mutation route around it.


## [2] Workflow
First, you're going to need to download ClinVar's variant data from the link below.
- Go to: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/ then click on "variant_summary.txt.gz"

Then, you're going to download the GCF GRCh38 human genome assembly file as well as its assembly report
- Go to: https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/

Then download the following:
- GCF_000001405.40_GRCh38.p14_genomic.fna.gz
- GCF_000001405.40_GRCh38.p14_assembly_report.txt

Move all of those into the 'database' folder. Now we are ready to do your first run. 

All of this can be run from 00_main.py.

```python
from a01_KeyStone import KeyStone
from a03_LookingGlass import LookingGlass
from a04_ReGen import ReGen

# Parse and filter for high-confidence data on Benign and Pathogenic variants from ClinVar
# Unzips human genome file, and uses the assembly report to help cross-reference variant locations
# Using the parsed locations, acquires flanking regions of configurable bps (default = 500bp)
# Neatly formats everything into a dataframe for downstream operations
def keystone_p1_demo(ml_modelname='ReGen_v2'):
    test = KeyStone(ml_modelname)
    test.naive_dataframe()
    test.decompress_genome()
    test.context_dataframe()

    
# With the previously built dataframe, runs the DNA and Protein Analysis toolkits on them to generate mutation profile dataframe
def keystone_p2_demo(ml_modelname='ReGen_v2'):
    # this process takes around 1:50 minutes generally depending on background tasks
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


# Initializes an XGB model and optimizes its hyperparameters with Optuna
def keystone_p3_demo(ml_modelname='ReGen_v2'):
    test = KeyStone(ml_modelname)
    test.train_models()


# When provided with a correctly-formatted special FASTA file (example provided in gene_databank folder), will classify each variant as benign or pathogenic
def LookingGlass_Demo(fasta_filename='test_fasta', ml_model='ReGen_v2', output_filename='Screen_test_1'):
    test_module = LookingGlass(fasta_filename, ml_model)
    if __name__ == "__main__":
        test_module.predict_file(output_filename)


# Utilizes an intelligent sequence of guided mutations and plateau-breaking stochastic mutations to optimize variants towards benign classification
def Repair_Gene(pathogenic_gene_file='benchmark_fasta', ml_model='ReGen_v2', outfile_name='benchmark_repair_test'):
    if __name__ == "__main__":
        module = ReGen(pathogenic_gene_file, ml_model, outfile_name)
        module.repair()
```


## [2.1] Current Model Stats - ReGen_v2
```
Model: ReGen_v2

Optimal Hyperparameters: {'n_estimators': 1936, 'max_depth': 10, 'learning_rate': 0.041977875319094894, 'subsample': 0.8691093047813849, 'colsample_bytree': 0.9973783186852718, 'reg_alpha': 0.20907871533405323, 'reg_lambda': 1.6124970064334614, 'gamma': 0.35865668074613577, 'scale_pos_weight': 1.8996291716997067}
Cross Validation Results: Mean ROC AUC: 0.8744, Mean PR AUC: 0.8684
Mean FNs: 5601.00, Mean FPs: 6627.00
ROC AUC: 0.8783
Precision-Recall AUC: 0.8739
Pathogenic F1-Score: 0.7885
Optimal threshold for pathogenic detection: 0.513
Performance with optimal threshold:
              precision    recall  f1-score   support

           0       0.83      0.81      0.82     41774
           1       0.78      0.80      0.79     34814

    accuracy                           0.81     76588
   macro avg       0.80      0.81      0.80     76588
weighted avg       0.81      0.81      0.81     76588

Confusion Matrix:
[[33884  7890]
 [ 6987 27827]]
```


## [3] LookingGlass example FASTA Input and Results

Users must provide FASTA sequences in this format to gene_databank:
```
Remember that flanking context sequences (ideally 500bp) must be provided
If flanking sequences are under 500, nothing's going to break, but the lack of context will most likely bottleneck the model's prediction power.

Below is a cut-off version of the pretend gene in gene_databank

>chr'X'| ref / alt / flank1 / flank2 | gene_name

>chr9|ref|BenchmarkGene1
ACTGACTGCATGACTAGCTACGACTGACTGACTCAGTACGACTGACTACGTACGATCGTCGCTGCATGCTCAG

>chr9|alt|BenchmarkGene1
ACTGCTGACTCAGTCAGACTGCATGACTACGTCAGTACGCATGCA

>chr9|flank1|BenchmarkGene1
ATCGATCGTACGCGCTAGTAGCTACGATCGAGATCGATCGACTGGCCGCTATATATCGCCGATCTGATGATCG

>chr9|flank2|BenchmarkGene1
ACTGGCGCCGCGGGGATATCTCTCTCTTAGCGCGCCGATATCGCACCATCACGACTAGTCAGTCGTCCGCGCC
```

*LookingGlass output from the test gene file*
```
Name,Predicted_Class,Prob_Benign,Prob_Pathogenic
benchmarkgene1,1,0.0025193095,0.9974807
benchmarkgene2,1,0.0074431896,0.9925568
```

## ReGen example Input and Results
Note: Users need to insert a FASTA file of the same custom format in the ReGen_input folder
```
================================================================================
ReGen Analysis Results: ReGen_V1 | benchmark_fasta | benchmarkgene1
================================================================================

ORIGINAL VARIANT STATS: 
Sequence: ACTGACTGCATGACTAGCTACGACTGACTGACTCAGTACGACTGACTACGTACGATCGTCGCTGCATGCTCAGTACGACTGACGAC
Benign % chance: 0.251931

ANALYSIS SUMMARY:
|- Starting Score: 0.002519
|- Original Length: 86 bp
|- Final Variants: 1
|- Benign Threshold Variants: 0
|- ReGen config: 50 iterations, 1 copies

MAX BENIGN VARIANTS PER ITERATION:
--------------------------------------------------
Score: 0.8367717266082764 | Length: 92 bp
Benign % increase: 0.5848407745361328
   Sequence:
    ACTGCTGACTCAGTCAGACTGCATGACTACGTCAGTACGCATGCATGACTGCATGCATCAGTACGTGCTGCTGCATGCGG
    CGCCCCCAAAAA

Score: 1.2857317924499512 | Length: 91 bp
Benign % increase: 1.0338008403778076
   Sequence:
    ACTGCGACTCAGTCAGACTGCATGACTACGTCAGTACGCATGCATGACTGCATGCATCAGTACGTGCTGCTGCATGCGGC
    GCCCCCAAAAA

Note: Top performing genes from every iteration are listed here, I'm skipping the other 47...

Score: 5.8156609535217285 | Length: 92 bp
Benign % increase: 5.563730001449585
   Sequence:
    ACTGGACTCATCAGACTGCGACTACGTCAGTACGCATGCATGACTGCATGCATCAGTACGTGCTGCTGCATGCGGCGCCC
    CCATGCATGACC

FINAL VARIANTS:
--------------------------------------------------
Score: 5.8156609535217285 | Length: 92 bp
Benign % increase: 5.563730001449585   Sequence: 
    ACTGGACTCATCAGACTGCGACTACGTCAGTACGCATGCATGACTGCATGCATCAGTACGTGCTGCTGCATGCGGCGCCC
    CCATGCATGACC
```
Multiple copies can be run simultaneously, I just personally chose to run with 1 for ease of debugging, as it was easier to keep track of the 1.


## Configuration
All parameters are configurable via 'config.yaml', this is a demo section:
```yaml
# ========================================
# Mutation Profile Fingerprinting Settings
# ========================================

# [DNA similarity matrix scores]
match: 1
mismatch: -1

# [K-mer size limits]
k_min: 2  # size limits on possible kmers
k_max: 7

# [Ambiguous string generation number]
variant_num: 3  # how many DNA variants we are willing to consider for ambiguous sequences

# [Functional Consequence Settings]
splice_site_window: 8
motif_window: 4

# [DNA repeat settings]
min_repeat_length: 2  # minimum size of repeat subsequences
max_repeat_length: 10
min_repeats: 5  # minimum times a subseq is repeated for its region to be considered a microsatellite

# [Protein settings]
protein_substitution_matrix: "blosum62.mat"
minimum_protein_length: 10
protein_gap_penalty: -8
```


## Prerequisites and Dependencies
- Python 3.7+
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - biopython
  - tqdm
  - optuna
  - pyfaidx
  
  
## References:
- Chromosome densities:
  - https://www.cshlp.org/ghg5_all/section/dna.shtml#:~:text=Gene%20density%20varies%20greatly%20among,x-axis%20of%20the%20figure.
- Nearest-neighbour thermodynamic parameters:
  - https://pubs.acs.org/doi/10.1021/bi951907q
- Kozak motif info:
  - https://www.sciencedirect.com/topics/biochemistry-genetics-and-molecular-biology/kozak-consensus-sequence
- AA Molecular Weights
  - https://www.thermofisher.com/ca/en/home/references/ambion-tech-support/rna-tools-and-calculators/proteins-and-amino-acids.html#:~:text=GGG%20Gly%20(G),amino%20acids%20x%20110%20Da
- Isoelectric Points
  - https://www.peptideweb.com/images/pdf/pKa-and-pI-values-of-amino-acids.pdf
- Hydrophobicities
  - https://www.sigmaaldrich.com/CA/en/technical-documents/technical-article/protein-biology/protein-structural-analysis/amino-acid-reference-chart?srsltid=AfmBOor2d8A7dJO8Hd-g3cDQimh8Z9IXz-jEcHic7DaGYH1OXi5oUriO
- Half Lives 
  - https://resources.qiagenbioinformatics.com/manuals/clcgenomicsworkbench/650/Estimated_half_life.html



Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
