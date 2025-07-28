import re

def custom_parse(fasta_filepath):
    """
    Parses FASTA file custom-made for this program
    \n Custom Format: chr'X' | Type (Ref, Alt, Flank_1/2) | Name / Identifier
    \n Where X is a valid chromosome number with X = 23 and Y = 24
    :param fasta_filepath:
    :return:
    """
    dataframe = []
    with open(fasta_filepath, 'r') as outfile:
        FASTAFile = [l.strip() for l in outfile.readlines()]
    FASTAstrings = {}
    FASTALabel = ''

    # Separate the fasta formatted DNA strings into labels and contents
    # This is so we can further sort them by the contents of their labels
    for line in FASTAFile:
        if line.startswith('>'):
            FASTALabel = line.lower()
            FASTAstrings[FASTALabel] = ''
        else:
            FASTAstrings[FASTALabel] += line

    current_chr = None
    current_name = None
    df_dict = {}

    for label, string in FASTAstrings.items():
        match = re.match(r'^>chr(\w+)\|(\w+)\|(\w+)$', label)
        if not match:
            raise ValueError(
                f"Invalid Header: {label}. DNA headers must follow the following format: chr'X'| Type (Ref, Alt, Flank_1/2) | Name / Identifier")
        chrom, tag, name = match.groups()

        if (current_chr is not None and current_name is not None
                and (chrom != current_chr or name != current_name)):
            # add chromosome and clinical significance - reset dictionary
            df_dict['Name'] = str(current_name)
            df_dict['Chromosome'] = int(current_chr)
            df_dict['ClinicalSignificance'] = 'N/A'
            dataframe.append(df_dict)
            df_dict = {}

        if 'ref' in tag:
            df_dict['ReferenceAlleleVCF'] = str(string)
        elif 'alt' in tag:
            df_dict['AlternateAlleleVCF'] = str(string)
        elif 'flank_1' in tag or 'flank1' in tag:
            df_dict['Flank_1'] = str(string)
        elif 'flank_2' in tag or 'flank2' in tag:
            df_dict['Flank_2'] = str(string)
        else:
            raise ValueError(f"Improper label: {tag}")
        current_chr = chrom
        current_name = name

    if df_dict:
        df_dict['Name'] = str(current_name)
        df_dict['Chromosome'] = int(current_chr)
        df_dict['ClinicalSignificance'] = 'N/A'
        dataframe.append(df_dict)

    return dataframe