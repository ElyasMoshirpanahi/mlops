# # src/tests/test_model.py
from src.isnet.inference import Inference
import base64

# # Run sample inference
# image_path = 'src/tests/bike.png'

def test_inference_model_instantiation():
  inference = Inference()
  # Check model loaded
  assert inference.model is not None
  print("Model loaded successfully")




# def test_inference_model_output():
#   inference = Inference()
#   image_bytes = open(image_path,"rb").read()
#   # Check if model segements the img without problem
#   output_bytes = inference.infer(image_bytes)
#   # Call your segmentation function with the image bytes
#   base64_result = base64.b64encode(output_bytes).decode('utf-8')        

#   # Check that the result is a non-empty base64 string
#   assert base64_result!= ''
#   assert isinstance(base64_result, str) 
  
#   print("Model Segmented images successfully")






 
    
