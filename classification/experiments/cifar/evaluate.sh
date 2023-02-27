STUDENT="$1"
ckpt="$2"

python -u cifar/main.py \
 --is_train=false \
 --student="$STUDENT" \
 --resume="$ckpt"