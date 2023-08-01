# # src/tests/test_model.py

import torch
import os
from src.isnet.inference import Inference
import base64

# Run sample inference
image_path = 'src/tests/bike.png'
out_path   = 'src/tests/bike_seg.png'

def test_inference_model_instantiation():

  inference = Inference()

  # Check model loaded
  assert inference.model is not None
  print("Model loaded successfully")




def test_inference_model_output():

  inference = Inference()

  # Check if model segements the img without problem
  inference.infer(image_path, out_path)

  # Validate output image exists
  assert os.path.exists(out_path)

  print("Model Segmented images successfully")
