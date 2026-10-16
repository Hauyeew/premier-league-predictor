cd /Users/haoyiwang/Desktop/PLPredictator\
python3 update_data.py\
python3 model/predict.py          # optional check\
uvicorn api:app --reload         # if the API is not already running\
cd website && npm run dev
