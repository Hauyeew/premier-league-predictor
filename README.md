cd /PLPredictor
python update_data.py
python model/predict.py          # optional check
uvicorn api:app --reload         # if the API is not already running
