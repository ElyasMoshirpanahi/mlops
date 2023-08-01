"""Common utility functions for the model"""

from pathlib import Path

import torch

from src.isnet.isnet import ISNetDIS


def download_model(
    output_file_path: str,
    url_id: str = "1XL4XwU2FmAM7nYDy4sZDwVwOemEWWGPR",
    quiet: bool = False,
) -> None:
    """
    Downloads the model from google drive

    Parameters
    ----------
    output_file_path : str
        The path to the ouput file
    url_id : str
        The id of the google drive url
    quiet : bool, optional
        If true shows rich output when downloading the model.
        If false, the download is quiet

        Defaults to false
    """
    import gdown  # import only with dev dependencies

    gdown.download(id=url_id, output=output_file_path, quiet=quiet)


def load_model(model_format: str, model_file_path: str = None) -> ISNetDIS:
    """
    Loads the saved model in the given format, from the artifacts folder

    Parameters
    ----------
    model_format : {"torch", "torchscript"}
        The format of model to load.
    model_file_path : str, optional
        The path to the saved model. If set to None, it will search the parent
        folder `artifacts` for the pre-trained model, with the name: model.ckpt,
        model.pt. If this is not found it will download it and convert it
        to the right format specified in the `model_format` parameter.

        Defaults to None.

    Returns
    -------
    ISNetDIS
        The network with the loaded model
    """
    # check if model artifact exists
    extensions = {"torch": "ckpt", "torchscript": "pt"}

    if model_file_path is None:
        artifacts_folder = Path(__file__).resolve().parent / "artifacts"
        model_file_path = artifacts_folder / f"model.{extensions[model_format]}"

        if not model_file_path.exists():  # if artifact doesn't exist , download it
            ckpt_file_path = artifacts_folder / "model.ckpt"
            download_model(str(ckpt_file_path))

            if model_format != "torch":  # convert model if needed
                convert_model_to(
                    model_format,
                    str(ckpt_file_path),
                    str(model_file_path),
                )
                ckpt_file_path.unlink()

    # create architecure and load model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_format == "torch":
        model = ISNetDIS()
        model.load_state_dict(
            torch.load(model_file_path, map_location=torch.device(device)),
        )

        if torch.cuda.is_available():
            model = model.cuda()

        model.eval()
    elif model_format == "torchscript":
        model = torch.jit.load(
            model_file_path,
            map_location=torch.device(device),
        )

        if torch.cuda.is_available():
            model = model.cuda()

    return model


def convert_model_to(
    output_format: str,
    input_file_path: str,
    output_file_path: str,
):
    """
    Converts a torch model to torchscript for optimized inference

    Parameters
    ----------
    output_format : {"torchscript"}
        The format of output model
    input_file_path : str
        The path to the pytorch trained model
    output_file_path : str
        The path to the torchscript model

    Raises
    ------
    Exception
        If output_format is in the accepted format
    """
    # create network and load model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ISNetDIS()
    model.load_state_dict(
        torch.load(input_file_path, map_location=torch.device(device)),
    )
    model.eval()

    # compile and save
    if output_format == "torchscript":
        compiled_model = torch.jit.script(model)
        compiled_model.save(output_file_path)
    else:
        raise Exception(
            f"The output format must be either: 'torchscript', but got: \
            {output_format}",
        )
