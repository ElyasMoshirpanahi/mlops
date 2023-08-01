export PYTHONPATH=$(pwd) &&
pip install -r requirements.txt &&
python src/cli.py download --output-file src/isnet/artifacts/model.pt &&
python -m uvicorn deployment.main:app --host 0.0.0.0 --port 8000 