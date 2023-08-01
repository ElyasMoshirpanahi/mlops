"""Inference class for ISNetDIS"""

from typing import Tuple
import imageio
import numpy as np
import torch
import torch.nn.functional as functional
from torchvision.transforms.functional import normalize
import src.isnet.model_utils as model_utils
import base64
    

class Inference:
    """Performs inference using the pre-trained ISnet model"""

    def __init__(
        self,
        model_format: str = "torchscript",
        model_file_path: str = None,
    ) -> None:
        """
        Initializes the Inference class by loading the segmentation model.

        Parameters
        ----------
        model_format : {"torchscript", "torch"}
            The format of model to load.

            Defaults to "torchscript"
        model_file_path : str, optional
            The path to the saved model. If set to None, it will search the parent
            folder `artifacts` for the pre-trained model, with the name: model.ckpt,
            model.pt. If this is not found it will download it and convert it
            to the right format specified in the `model_format` parameter.

            Defaults to None.
        """
        self.model = model_utils.load_model(model_format, model_file_path)
        self.model_format = model_format

    def _preprocess_image(
        self,
        image_path: str,
    ) -> Tuple[Tuple[int, int], torch.Tensor]:
        """
        Preprocesses the input image before running inference.

        Parameters
        ----------
        image_path : str
            The path to the input image.

        Returns
        -------
        A tuple containing:
        - orig_width_height : tuple
            The original width and height of the input image.
        - image : torch.Tensor
            The preprocessed input image as a PyTorch tensor.
        """
        image = imageio.v3.imread(image_path)

        # ensure that images has three dimensions
        if len(image.shape) < 3:
            image = image[:, :, np.newaxis]
        elif len(image.shape) >= 3 and image.shape[2] > 3:  # discard alpha channel
            image = image[:, :, :3]

        orig_width_height = image.shape[0:2]

        # convert to tensor with the right shape
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        image_tensor = functional.interpolate(
            torch.unsqueeze(image_tensor, 0),
            [1024, 1024],
            mode="bilinear",
            align_corners=False,
        ).type(torch.uint8)

        image = torch.divide(image_tensor, 255.0)
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

        return orig_width_height, image

    def infer(self, image_path: bytes) -> bytes:
        """
        Runs inference on the input image and saves the output to the specified path.

        Parameters
        ----------
        image_path : str
            The path to the input image.
        output_image_path : str
            The path where the output image will be saved.
        """

        with torch.no_grad():
            # preprocess image
            orig_width_height, image = self._preprocess_image(image_path)

            if torch.cuda.is_available():
                image = image.cuda()

            # run the network
            result = self.model(image)

            # interpolate to original width and height dimensions
            result = torch.squeeze(
                functional.interpolate(
                    result[0][0],
                    orig_width_height,
                    mode="bilinear",
                    align_corners=False,
                ),
                0,
            )

            # normalize segmentation mask output between 0 and 1
            max = torch.max(result)
            min = torch.min(result)
            result = (result - min) / (max - min)

            # save the result
            output_array = (
                (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
            )


            image_bytes = imageio.imwrite(imageio.RETURN_BYTES, output_array, format='PNG')

            return image_bytes

            # imageio.imsave(
            #     output_image_path,
            #     output_array,
            # )

            
            # 