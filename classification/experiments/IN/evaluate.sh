STUDENT="$1"
ckpt="$2"

python -u IN/main.py \
 --is_train=false \
 --student="$STUDENT" \
 --resume="$ckpt"