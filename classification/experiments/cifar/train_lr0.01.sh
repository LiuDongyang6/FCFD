STUDENT="$1"
TEACHER="$2"
random="${3:-$RANDOM}"

exp_name="cifar"
running_name="$STUDENT"-"$TEACHER"-"$random"
mkdir -p output/"$exp_name"/"$running_name"/log
python -u cifar/main.py \
 --init_lr 0.01 --batch_size 64 \
 --output_dir=output/"$exp_name"/"$running_name" \
 --config=experiments/"$exp_name"/"$STUDENT"-"$TEACHER".yaml \
 --student="$STUDENT" --teacher="$TEACHER" \
 --teacher_ckpt=pretrain/cifar_teachers/"$TEACHER"_vanilla/ckpt_epoch_240.pth \
 --random_seed="$random" \
 &>output/"$exp_name"/"$running_name"/log/output.log &