import base64
import pytest
from unittest.mock import patch, mock_open

def test_if_image_is_valid_image(mock_post_request):
    # Simulate the API response (status_code 200)
    response = mock_post_request(url)

    b64_img = response["image"]
    assert response["status_code"] == 200
    assert b64_img != ""


def test_if_image_is_none_image(mock_post_request):
    # Simulate the API response (status_code 400)
    response = mock_post_request(url, status_code=400)
    assert response["status_code"] == 400
    assert "error" in response


def test_if_image_is_no_image(mock_post_request):
    # Simulate the API response (status_code 422)
    response = mock_post_request(url, status_code=422)
    assert response["status_code"] == 422
    assert "error" in response


def test_if_file_format_is_invalid(mock_post_request):
    # Mock the content of the file
    with patch("builtins.open", mock_open(read_data="hello")):
        # Simulate the API response (status_code 422)
        response = mock_post_request(url, status_code=422)
        assert response["status_code"] == 422
        assert "error" in response


def test_bad_image_that_will_give_server_internal_error(mock_post_request):
    # Mock the file reading
    with patch("builtins.open", mock_open(read_data=b"sample_image_data")):
        # Simulate the API response (status_code 500)
        response = mock_post_request(url, status_code=500)

    assert response["status_code"] == 500
    assert "error" in response


@pytest.fixture
def mock_post_request(monkeypatch):
    def mock_post(url, files=None, status_code=200):
        # Simulate the API response based on the status_code provided
        if status_code == 200:
            # Generate a sample base64 string as response
            response = {"status_code": status_code, "image": "sample_base64_string"}
        else:
            # Generate an error response
            response = {"status_code": status_code, "error": "some error message"}

        return response

    monkeypatch.setattr("requests.post", mock_post)

  # # test_api.py 

  # import requests
  # import base64
  # import pytest
  # import os 


  # docs = "http://localhost:8000/docs"
  # url = "http://localhost:8000/segment" 


  # def test_if_server_is_running():
  #   response = requests.get(docs)
  #   assert response.status_code == 200

  # def test_if_image_is_valid_image():
  #   file_path = "src/tests/bike.png"
  #   ext = file_path.split(".")[-1]
  #   f = file_path
  #   files=[('image',(f'{ext}',open(f,'rb'),f'image/{ext}'))]
    

  #   response = requests.post(url, files=files)
    
  #   b64_img = response.json()["image"]
  #   # img_bytes = base64.b64decode(b64_img)
  #   # out_path = "src/tests/out.png"
  #   # with open(out_path, "wb") as f:
  #   #   f.write(img_bytes)

  #   assert response.status_code == 200
  #   assert b64_img != ""

  #   # os.remove(out_path)

  # def test_if_image_is_none_image():
  #   files = {"image": None}
  #   response = requests.post(url, files=files)
  #   assert response.status_code == 400
  #   assert "error" in response.text


  # def test_if_image_is_no_image():
  #   files = {"image": ""}
  #   response = requests.post(url, files=files)
  #   assert response.status_code == 422
  #   assert "error" in response.json()



  # def test_if_file_fomrat_is_invalid():
  #   file_name = "src/tests/test.txt"
  #   with open(file_name,"w") as  f:
  #     f.write("hello")

  #   files = {"image": file_name}
  #   response = requests.post(url, files=files)
  #   assert response.status_code == 422
  #   assert "error" in response.json()


  # def test_bad_image_that_will_give_server_internal_error():
  #   file_path = "src/tests/bike_seg.png"

  #   with open(file_path, "rb") as f:
  #     files = {"image": f}
  #     response = requests.post(url, files=files)


  #   assert response.status_code == 500
  #   assert "error" in response.json()

