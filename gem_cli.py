import argparse
import sys

from src.gem.a03_LookingGlass import LookingGlass
from src.gem.a04_ReGen import ReGen
from src.gem.paths import *

def validate_fasta(filepath, flag):
    path = Path(filepath)
    gene_databank_path = GENE_DATABANK / path

    if path.exists():
        return str(path)

    if flag == 'screen' or flag == 'repair':
        if gene_databank_path.exists():
            return str(gene_databank_path)

    # file not found
    print(f"Error: input filepath '{filepath}' not found")
    print(f"Searched:")
    print(f" - {gene_databank_path}")
    print(f"\n ensure custom FASTA file is in correct folder location")
    sys.exit(1)

def valid_name(name):
    if name == "clinicmod":
        return "ClinicalModel"
    elif name == "discmod":
        return "DiscriminatorModel"
    else:
        raise ValueError("Name must be one of: clinicmod / discmod")

def gem_screen(args):
    input_file = validate_fasta(args.input, "screen")
    model_name = valid_name(args.model)
    output_filename = args.output

    try:
        if __name__ == "__main__":
            LG = LookingGlass(model_name, genefile_route=input_file)
            print(f"\nScreening variant file: {input_file}")
            LG.predict_file(outfile_name=output_filename)

    except Exception as e:
        print(f"\nError during screening: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def gem_repair(args):
    input_file = validate_fasta(args.input, "repair")
    model_name = valid_name(args.model)
    output_filename = args.output
    its=args.iterations if args.iterations is not None else 10
    cops=args.copies if args.copies is not None else 1

    try:
        if __name__ == "__main__":
            RG = ReGen(model_name,
                       pathogenic_route=input_file,
                       outfile_name=output_filename,
                       iterations=its,
                       copies=cops)
            RG.repair()

    except Exception as e:
        print(f"\nError during repair: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def entry_point():
    parser = argparse.ArgumentParser(
        prog = 'GEM',  # name of program
        description = "GEM: Gene Edit Machine - Variant Pathogenicity Prediction and Reduction",
        formatter_class = argparse.RawDescriptionHelpFormatter,  # customizes help output
        epilog=  # text to display before argument
        """
        Parameters:
            input == input gene filepath
            
            --model clinicmod == Clinical Model
            --model discmod == Discriminator Model
            
            --iterations  == Regen Parameters / Run Options
            --copies
            
            output == output filename
        
        Command List:
            will fill this out after mentally recovering
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # ==============
    # screen command
    # ==============
    screen_parser = subparsers.add_parser(
        'screen',
        help='Screen input variant for pathogenicity prediction: requires input and output filenames',
        description='Screens input variant for pathogenicity prediction'
    )

    screen_parser.add_argument(
        'input',
        type=str,
        help='Input custom FASTA file w/ variants (provide filename inside gene_databank or full path)'
    )

    screen_parser.add_argument(
        '--model',
        type=str,
        default='clinicmod',
        choices=['clinicmod', 'discmod'],
        help='Model to use: "clinicmod" (lowest FNs), "discmod" (balanced pure discrimination)',
    )

    screen_parser.add_argument(
        'output',
        type=str,
        default=None,
        help='Output filename prefix'
    )

    # ==============
    # repair command
    # ==============
    repair_parser = subparsers.add_parser(
        'repair',
        help='Repair input variant for pathogenicity reduction: requires input and output filenames',
        description='Screen input variant for pathogenicity reduction & potential gene therapy target identification'
    )

    repair_parser.add_argument(
        'input',
        type=str,
        help='Input custom FASTA file w/ variants (provide filename inside gene_databank or full path)'
    )

    repair_parser.add_argument(
        '--model',
        type=str,
        default='clinicmod',
        choices=['clinicmod', 'discmod'],
        help='Model to use: "clinicmod" (lowest FNs), "discmod" (balanced pure discrimination)',
    )

    repair_parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Number of pathogenicity reduction iterations [default=10]'
    )

    repair_parser.add_argument(
        '--copies',
        type=int,
        default=1,
        help='Number of parallel sequence copies for pathogenicity reduction [default=1]'
    )

    repair_parser.add_argument(
        'output',
        type=str,
        default=None,
        help='Output filename prefix'
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == 'screen':
        gem_screen(args)
    elif args.command == 'repair':
        gem_repair(args)


if __name__ == "__main__":
    entry_point()

# python gem_cli.py [screen / repair] --i input.fasta --model [clinicmod / discmod] --o output_name








