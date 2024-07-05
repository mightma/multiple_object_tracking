python3 tracking/track.py \
    --yolo-model ../zmot/train-yolov8l-1cls-smtj/litter_1cls_yolov8l/weights/best.pt \
    --tracking-method botsort \
    --source ../zmot/videos/AM01-1.mp4 \
    --project ../outputs \
    --name boxmot-track-video-litter-1cls-yolov8l-botsort \
    --save-txt \
    --save
