STUDENT="$1"
TEACHER="$2"
random="${3:-$RANDOM}"

exp_name="IN"
running_name="$STUDENT"-"$TEACHER"-"$random"
mkdir -p output/"$exp_name"/"$running_name"/log
python -u IN/main.py \
 --init_lr 0.1 --batch_size 256 \
 --output_dir=output/"$exp_name"/"$running_name" \
 --config=experiments/"$exp_name"/"$STUDENT"-"$TEACHER".yaml \
 --student="$STUDENT" --teacher="$TEACHER" \
 --random_seed="$random" \
 &>output/"$exp_name"/"$running_name"/log/output.log &