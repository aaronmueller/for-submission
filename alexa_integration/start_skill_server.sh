export FLASK_APP=alexa_server.py
export FLASK_ENV=development
export FLASK_RUN_CERT=adhoc

source activate dialog

flask run -p 2020

# flask run -p 2020 -m internal:hred -mf /Users/alexandradelucia/discourse-hw3/parlai_internal/zoo/movie_hred/hred_model.ckpt.checkpoint --no-cuda

