"""
CLI interface to perform inference using DISNet

This script provides a command line interface for performing inference using DISNet.
It has three main commands, each with its own set of options:

download: Download the DISNet model.
    --output-file, -o: The path to save the downloaded model file.

convert: Convert a trained PyTorch model to torchscript.
    --input-file, -i: The path to the trained PyTorch model file.
    --output-file, -o: The path to save the converted model file.
    --to-torchscript, -ts: If specified, convert the model to torchscript format.

inference: Perform inference using a trained DISNet model.
    --model-file, -m: The path to the trained model file.
    --model-format, -f: The format of the model file. Must be one of `torch`,
                        or `torchscript`.
    --input-file, -i: The path to the input image file.
    --output-file, -o: The path to save the output segmentation mask file.

Examples
--------
Download the DISNet model:
    python cli.py download --output-file /path/to/model.pt

Convert a trained PyTorch model to torchscript:
    python cli.py convert \
                  --input-file /path/to/input.pt \
                  --output-file /path/to/output.pt \
                  --to-torchscript

Run inference on an input image file called input.png using a trained model file
converted to torchscript called model.pt, and save the output segmentation mask as
output.png:
    python cli.py inference \
                  --model-file /path/to/model.pt \
                  --model-format torchscript \
                  --input-file /path/to/input.png \
                  --output-file /path/to/output.png
"""

import argparse

import src.isnet.model_utils as model_utils
from src.isnet.inference import Inference


def add_model_parser(subparsers: argparse._SubParsersAction) -> None:
    """Create the parser for the `model` command"""
    model_parser = subparsers.add_parser("download", help="Download the model")

    model_parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        required=True,
        help="The pytorch model file",
    )


def add_convert_parser(subparsers: argparse._SubParsersAction) -> None:
    """Create the parser for the `convert` command"""
    model_parser = subparsers.add_parser(
        "convert",
        help="Convert a trained pytorch model to torchscript",
    )

    model_group = model_parser.add_argument_group(
        "model conversion options",
        "Select --to-torchscript",
    )
    model_mutually_exclusive_group = model_group.add_mutually_exclusive_group(
        required=True,
    )
    model_mutually_exclusive_group.add_argument(
        "--to-torchscript",
        "-ts",
        action="store_true",
        help="Convert the downloaded model to torchscript",
    )
    
    model_parser.add_argument(
        "--input-file",
        "-i",
        type=str,
        required=True,
        help="The trained torch input file",
    )
    model_parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        required=True,
        help="The output file",
    )


def add_inference_parser(subparsers: argparse._SubParsersAction) -> None:
    """Create the parser for the `inference` command"""
    inference_parser = subparsers.add_parser("inference", help="Perform inference")

    inference_parser.add_argument(
        "--model-file",
        "-m",
        type=str,
        required=False,
        default=None,
        help="The path to the trained model",
    )
    inference_parser.add_argument(
        "--model-format",
        "-f",
        type=str,
        required=True,
        help="The format of the model. Can be `torch`, or `torchscript`",
    )
    inference_parser.add_argument(
        "--input-file",
        "-i",
        type=str,
        required=True,
        help="The path to the input image file",
    )
    inference_parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        required=True,
        help="The path to the output segmentation mask file",
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        An object containing the parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="CLI interface to perform inference using DISNet",
    )
    subparsers = parser.add_subparsers(dest="command")

    add_model_parser(subparsers)
    add_convert_parser(subparsers)
    add_inference_parser(subparsers)

    args = parser.parse_args()
    return args


def run_inference(
    input_file_path: str,
    output_file_path: str,
    model_file_path: str = None,
    model_format: str = "torchscript",
) -> None:
    """
    Run image segmentation inference using a trained model on an input image and save
    the segmentation mask.

    Parameters
    ----------
    input_file_path : str
        The path to the input image file.
    output_file_path : str
        The path to the output segmentation mask file.
    model_file_path : str, optional
        The path to the trained model. If None, it downloads automatically the model.

        Defaults to None
    model_format : {"torchscript", "torch"}
        The format of model to load.

        Defaults to "torchscript"
    """
    inference = Inference(model_format, model_file_path)
    inference.infer(input_file_path, output_file_path)


def main():
    """Parse command line arguments and execute corresponding command."""
    args = parse_args()

    if args.command == "download":
        model_utils.download_model(output_file_path=args.output_file)

        print(f"Downloaded model saved: {args.output_file}")
    elif args.command == "convert":
        output_format = "torchscript"

        model_utils.convert_model_to(
            output_format=output_format,
            input_file_path=args.input_file,
            output_file_path=args.output_file,
        )

        print(f"Converted model saved: {args.output_file}")
    elif args.command == "inference":
        run_inference(
            input_file_path=args.input_file,
            output_file_path=args.output_file,
            model_file_path=args.model_file,
            model_format=args.model_format,
        )


if __name__ == "__main__":
    main()
