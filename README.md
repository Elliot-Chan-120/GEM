# GEM
[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&color=5E81AC&width=435&height=40&lines=Benign+by+Design)](https://git.io/typing-svg)
- developed by: Elliot Chan
- contact: e224chan@uwaterloo.ca

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
  - Chromosome density
  - Length changes
  - Thermodynamic stability analysis using nearest-neighbour parameters
  - GC content and CpG island disruption analysis 
  - Shannon entropy for sequence complexity
  - Hairpin structure prediction
  - Microsatellite detection
  - Intron detections
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
  - Hydrophobicity changes
  - Aliphatic index
  - Protein half-life predictions
- Structural Analysis
  - Secondary structure propensity analysis (BioPython-aided)
  - Proline content assessment
  - Global and local protein sequence alignment with BLOSUM64 matrix
- Feature Interaction Engineering includes:
  - Composite scores combining multiple properties
  - Interaction terms (e.g. charge-hydrophobicity)
  - Relative metrics normalized by protein length

**DNAMatrix - regulatory motif analysis**

*Captures how variants disrupt DNA-level regulatory architecture through multi-dimensional scoring of transcriptional and post-transcriptional control.*

Scans variety of motifs found in 3 regulatory domains: 

- initiation signals: initiator, TATA box, Kozak Sequence
- transcription factors: CTCF, CAAT, SP1, NF-kB, AP1, CREB
- post-transcriptional regulatory elements: 5’/3’ splice sites, branch points, polyadenylation signals

Each motif is scored using position-weight matrices with a Gaussian-weighted distance decay function that assigns higher impact to disruptions near the variant site. The module calculates three complementary metrics across individual motif, domain, and total DNA levels:

- raw motif count changes
- position-weighted score shifts
- cluster composite scores to detect synergistic effects when multiple binding sites are disrupted within close proximity. The distance threshold is configurable (30bp default).

By aggregating individual motif disruptions into domain-level cluster scores, the system identifies coordinated regulatory breakdowns that simple count-based methods would miss. 

ProtMatrix - protein motif analysis

*Extends variant impact assessment to the protein level, analyzing amino acid changes and disruptions in post-translational modification sites and protein-protein interaction domains. Profiles four major regulatory systems:*

- phosphorylation sites: CDK, cAMP-PKA, CK2, Tyrosine Kinase
- glycosylation sites: N-glycosylation consensus sequences
- ubiquitination signals: D-box, KEN-box degrons
- interaction domains: SH2 family, 14-3-3 binding sites, pdz domains, nuclear localization and export signals

Since proteins sequences are extrapolated without genomic coordinates but from translation likelihood calculations, the module employs PWM scanning combined with regex pattern matching for canonical motifs. Similar cluster composite scoring is implemented, capturing the disruption of spatially clustered PTM sites or binding domains.

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
Protein analysis was a big hurdle, as I had find a way to balance processing power with biological accuracy. For every string of DNA (above a certain size), there are 6 ways to read and translate proteins from it, along what we call Reading Frames. Each reading frame usually contains multple Open Reading Frames (ORFs), characterized by start and stop codons, which is where the proteins actually get made.
Naive approaches to analyzing the protein level would involve brute-force scanning of all 6 reading frames, then scanning for all possible ORFs, extracting all sequences, then conducting analyses on all of them then doing that again for variant comparison. So there's 6 RFs, most likely multiple ORFs in each, and we probably have to compare the average of those results between 2 sequences 380k+ times. This would, at best, nuke my computer. The solution to this would be to preprocess the entire sequence first, then determine its protein translation efficiency.

I started off with scanning for start and stop codons, tracking the positions where they appeared. Valid potential reading frames that started and ended with in-frame start and stop codons (% 3 == 0) and had no in-frame stop codons inside were saved, then scanned for a Kozak motif. A Kozak motif generally looks like this "gccA/Gcc[AUG]G" around the [start codon] ATG / AUG (mRNA), where the capital letters represent the most important positions. It is a powerful predictor of translation initiation, so if you find it, there is a high chance protein translation will initiate immediately after. Generally, the largest ORF is the one that will encode the protein. By searching for the Kozak motif and getting the length of all potential ORFs, we can generate a score for each, where the highest-scoring ORF is one we can safely assume will encode the protein.

### Motif Analysis Strategy
The previous version focused on extracting predictive features relying on sequence conservation and direct protein structure disruption. However, I felt that it was missing a critical later: regulatory context.
A variant in a highly conserved region might be benign if it doesn't disrupt any motifs contributing to regulatory logic, while a variant in another region could be pathogenic if it breaks a critical cluster / regulatory hub.
The main challenge I found here was finding a way to accurately capture these effects on a scale, not just binary to really capture relevant, diverse biological signals.

I chose to detect disruptions of a suite of motifs on both the DNA and protein level via Position Weight Matrices, which represent the degenerate nature of motifs in real biological systems.
To further amplify the signals' biological meaning, I developed the toolkit with a Gaussian-weighted distance decay scoring mechanism combined with a cluster-composite scoring system. 
This meant that disruptions to motifs occurring close the mutation site's window were scored higher than those further away. 

Regulatory elements rarely function in isolation, in many cases breaking one regulatory site is less severe than compromising an entire regulatory hub, hence the implementation of a cluster composite scoring algorithm.
Unlike simply counting motifs, this identifies synergistic disruption, when variants don't remove individual sites but break down coordinated regulatory 'clusters'.

### Flexible Multiprocessing
Both CompositeDNA and CompositeProt were optimized with multiprocessing by initializing a multiprocessing pool upon class instantiation, terminating automatically when under a context manager (with xyz as CompositeDNA/Prot(core_num=max_cores-2)).
This sped up protein analysis time from a predicted 8-9 hours to 1:30.
What makes them flexible is that pool persistence is encoded into them, allowing for them to be loaded up once, many dataframes passed onto them for analysis, then terminated at will rather than automatically via a context manager. 
This saved an immense amount of overhead in ReGen, where strings were constantly being altered and rerun through their fingerprinting pipelines for repeated predictions.

### ReGen's Adaptive Logic
A look at the config file or ReGen's logic will reveal a "retain_counter". What this is for is to keep track of the amount of times a mutation did not result in an increase in benignity. 
Once this passes a config-defined threshold (retain_threshold), the variant will undergo a certain amount of random mutations, and the highest-scoring one is allowed to pass on under a more lenient error threshold.
The threshold decreases as retain-count increases, becoming more lenient the more the variant demonstrates ridigity. This, combined with the stochastic mutations is designed to break through plateaus in performance, finding the mutation route around it. I would recommend setting this to 1, but you can leave it at 3 to see how it works once a plateau is hit.


## [2] Installation and Workflow
### *Installation and Setup*
- Go to the Releases Tab on the right hand side of this GitHub page
- Download the latest release as a .zip file
- Unzip the file on your computer
- Open/Load up the folder using your preferred Python IDE (PyCharm, VSCode, or others)
- In your python terminal, type in "pip install" followed by the list of required packages in the Prerequisites and Dependencies section of this README.md

Now you're ready to set up the project's database.

First, you're going to need to download ClinVar's variant data from the link below.
- Go to https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/ then click on "variant_summary.txt.gz"

Then, you're going to download the GCF GRCh38 human genome assembly file as well as its assembly report.
- Go to https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/ then download the following:
- GCF_000001405.40_GRCh38.p14_genomic.fna.gz
- GCF_000001405.40_GRCh38.p14_assembly_report.txt

Move all of those into the 'database' folder. Now we are ready to do your first run. 


### *Workflow*

All of this is on 00_main.py.
```python
from a01_KeyStone import KeyStone
from a03_LookingGlass import LookingGlass
from a04_ReGen import ReGen

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

```


## [2.1] Current Model Stats - ReGen_v4
```
Model: ReGen_v4

Cross Validation Results: Mean ROC AUC: 0.8920, Mean PR AUC: 0.8874
Mean FNs: 5478.60, Mean FPs: 5833.60
ROC AUC: 0.8939
Precision-Recall AUC: 0.8900
Pathogenic F1-Score: 0.7995
Optimal threshold for pathogenic detection: 0.480
Performance with optimal threshold:
              precision    recall  f1-score   support

           0       0.84      0.82      0.83     41774
           1       0.79      0.81      0.80     34814

    accuracy                           0.82     76588
   macro avg       0.81      0.82      0.81     76588
weighted avg       0.82      0.82      0.82     76588
Confusion Matrix:
[[34247  7527]
 [ 6586 28228]]
```


## [3] LookingGlass example FASTA Input and Results

Users must provide FASTA sequences in this format to gene_databank:
```
>chr'X'| ref / alt / flank1 / flank2 | gene_name

Remember that flanking context sequences (ideally 500bp) must be provided
If flanking sequences are under 500, nothing's going to break, but the lack of context will most likely bottleneck the model's prediction power.

BenchmarkGene1 is the first pathogenic gene variant in our dataset: ID NC_000007.14
I have trimmed it down for this README.md, the full sequence is in the file

>chr9|ref|BenchmarkGene1
GCTGCTGGACCTGCC

>chr9|alt|BenchmarkGene1
G

>chr9|flank1|BenchmarkGene1
ACACCTGTAATCCCAGCACCTCGGGAGGCCAAGGCAGGAGGATCTTGAGGCCAGGAGTTCAAGACCAGCC

>chr9|flank2|BenchmarkGene1
CTGCTTGACGGCGGTGCTGGACCTGCAGCTCAGGTGGGCCCCTCACCCTCTGCCAGCGCTGCGTCT
```

*LookingGlass output from the test gene file*
```
Name,Predicted_Class,Prob_Benign,Prob_Pathogenic
benchmarkgene1,1,0.026314199,0.9736858
benchmarkgene2,1,0.031000197,0.9689998
```

## ReGen example Input and Results
Note: Users need to insert a FASTA file of the same custom format in the ReGen_input folder
```
================================================================================
ReGen Analysis Results: ReGen_v4 | benchmark_fasta | benchmarkgene1
================================================================================

ORIGINAL VARIANT STATS: 
Ref Sequence: GCTGCTGGACCTGCC
Alt Sequence: G
Benign % chance: 2.631420

ANALYSIS SUMMARY:
|- Starting Score: 0.026314
|- Original Length: 15 bp
|- Final Variants: 1
|- Benign Threshold Variants: 1
|- ReGen config: 50 iterations, 1 copies

MAX BENIGN VARIANTS PER ITERATION:
--------------------------------------------------
Score: 67.49681830406189 | Length: 3 bp
Benign % increase: 64.86539840698242
   Sequence:
    GTA

Score: 71.39513492584229 | Length: 6 bp
Benign % increase: 68.76371502876282
   Sequence:
    GTATTA

Note: Top performing genes from every iteration are listed here, I'm skipping the others as there was no improvement


Score: 82.05875158309937 | Length: 3 bp
Benign % increase: 79.4273316860199
   Sequence:
    TTA

BENIGN THRESHOLD VARIANTS: <- these are the variants that surpassed user-defined threshold
--------------------------------------------------
Score: 82.05875158309937 | Length: 3 bp
Benign % increase: 79.4273316860199
   Sequence:
    TTA

FINAL VARIANTS:
--------------------------------------------------
Score: 82.05875158309937 | Length: 3 bp
Benign % increase: 79.4273316860199
   Sequence: 
    TTA
```
Multiple copies can be run simultaneously, I just personally chose to run with 1 for ease of debugging, as it was easier to keep track of the 1.
More copies and iterations generally will get better results.

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
