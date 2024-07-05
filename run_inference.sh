# Examples of inferencing detection model
# yolo detect predict model={path/to/model/checkpoint.pt} \
#     source={path/to/source} imgsz=640 project=outputs name={folder_name_containing_results}
# where {source} could be path to image, video or directory containing images

# yolo detect predict model=zmot/train-yolov8l-1cls-smtj/litter_1cls_yolov8l/weights/best.pt source=zmot/1000_samples/val2017 imgsz=640 project=outputs name=detect-val2017-litter-1cls-yolov8l

yolo detect predict model=zmot/train-yolov8l-1cls-smtj/litter_1cls_yolov8l/weights/best.pt source=zmot/videos/AM01-1.mp4 imgsz=640 project=outputs name=detect-video-litter-1cls-yolov8l