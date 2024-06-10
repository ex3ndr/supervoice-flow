set -e
accelerate launch ./train.py
# while true; do
#     accelerate launch ./train.py || true
# done