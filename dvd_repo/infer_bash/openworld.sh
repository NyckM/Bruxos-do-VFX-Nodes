CKPT='ckpt'

# You could increase the resolution here but expect slower inference speed.
HEIGHT=480
WIDTH=640

# You might increase these two number to ensure more stable scale variation but expect slower inference speed
WINDOW_SIZE=81
OVERLAP=21

# INPUT_VIDEO='demo/drone.mp4'

# python test_script/test_single_video.py --ckpt $CKPT --input_video $INPUT_VIDEO  --height $HEIGHT --width $WIDTH --window_size $WINDOW_SIZE --overlap $OVERLAP

INPUT_VIDEO='demo/robot_navi.mp4'

python test_script/test_single_video.py --ckpt $CKPT --input_video $INPUT_VIDEO  --height $HEIGHT --width $WIDTH --window_size $WINDOW_SIZE --overlap $OVERLAP