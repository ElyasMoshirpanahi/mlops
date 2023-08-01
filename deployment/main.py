from typing import Union
from fastapi import FastAPI, File, UploadFile, HTTPException

from fastapi.responses import JSONResponse
from src.isnet.inference import Inference
from io import BytesIO
from datetime import datetime as dt
import base64
import os
import logging
from io import BytesIO



app = FastAPI()
inference = Inference()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

@app.get("/")
def refer_to_docs():
    return "Fastapi is running! ,checkout the docs at /docs url of whatever domain you are using now!"


@app.post("/segment")
async def segment(image: UploadFile = File(...)):
    try:
        #Checking for none or Null
        if not image.filename:
          response= {"error": "invalid or no image"} 
          return JSONResponse(content=response, status_code=422)


        #Checking file format
        file_ext = image.filename.split(".")[-1]
        if file_ext.lower() not in ALLOWED_EXTENSIONS:
          response={"error": "invalid file format. Allowed formats: png, jpg, jpeg"}
          return JSONResponse(content=response,status_code=422)

        # Read to stream :New approach without file saving
        image_stream = await image.read()
        
        #Processing the segmenation on the image
        image_bytes = inference.infer(image_stream)
        
        image_base64= base64.b64encode(image_bytes)        

        #Sending back the response
        response = {"image":image_base64}

        return response


        # return JSONResponse(content=final_img,status_code=200) #would raise a json serilization error!

    except Exception as e:

        #Reporting back the errors
        response={"error": e.args[0]}
        return JSONResponse(content=response,status_code=500)
    


@app.post("/test_image")
async def get_image(image: UploadFile = File(...)):
    try:
        if not is_allowed_file(image.filename):
            raise HTTPException(status_code=422, detail="Invalid file format. Allowed formats: png, jpg, jpeg")

        contents = await image.read()
        b64_string = base64.b64encode(contents).decode("utf-8")
        return {"image": b64_string}

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# @app.post("/test_image")
# async def get_image(image: UploadFile = File(...)):


#   try:
    
#       if not image.filename:
#         return {"error": "invalid or no image"} , 422
      

#       file_ext = image.filename.split(".")[-1]

#       if file_ext.lower() not in ALLOWED_EXTENSIONS:
#         return {"error": "Invalid file format. Allowed formats: png, jpg, jpeg"}, 422


#       contents = await image.read()
#       b64_string = base64.b64encode(contents).decode('utf-8')
#       return {"image": b64_string},200

#   except Exception as e:
#     return {"error": e, "message": e.args},500



#-----------------------Helper functions-----------------------#

# def gen_img_path():
#   s = str(dt.today()).replace(":","_").replace(" ","")
#   return  (f"in_{s}.png",f"out_{s}.png")


def is_allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
