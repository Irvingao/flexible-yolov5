CUDA_VISIBLE_DEVICES=0 
set PYTHONPATH="/root/ghz_ws/flexible-yolov5:$PYTHONPATH"
python scripts/train_seg.py --seg --batch-size 10 --epochs 25 --dataset cityscapes
# --weights 'run/coco/checkpoint'