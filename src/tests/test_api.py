from fastapi.testclient import TestClient
from fastapi import status
from deployment.main import app

client  = TestClient(app=app)


def test_if_server_is_running():
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK


# def test_if_image_is_valid_image():
#     file_path = "src/tests/bike.png"
#     ext = file_path.split(".")[-1]
#     f = file_path
#     files=[('image',(f'{ext}',open(f,'rb'),f'image/{ext}'))]

#     response = client.post("/segment", files=files)
    
#     assert response.status_code == status.HTTP_200_OK
#     assert "error" != response.json()



def test_if_image_is_none_image():
    files = {"image": None}
    response = client.post("/segment", files=files)
    assert response.status_code == 400
    assert "error" in response.text


def test_if_image_is_no_image():
    files = {"image": ""}
    response = client.post("/segment", files=files)
    assert response.status_code == 422
    assert "error" in response.json()



def test_if_file_fomrat_is_invalid():
    file_name = "src/tests/test.txt"
    with open(file_name,"w") as  f:
        f.write("hello")

    files = {"image": file_name}
    response = client.post("/segment", files=files)
    assert response.status_code == 422
    assert "error" in response.json()


def test_bad_image_that_will_give_server_internal_error():
    file_path = "src/tests/bike_seg.png"

    with open(file_path, "rb") as f:
        files = {"image": f}
        response = client.post("/segment", files=files)


    assert response.status_code == 500
    assert "error" in response.json()
