# Examples of inferencing multi-objects tracking model
# yolo track model={path/to/model/checkpoint.pt} \
#     source={path/to/video} imgsz=640 project=outputs name={folder_name_containing_results}

# yolo track model=zmot/train-yolov8l-1cls-smtj/litter_1cls_yolov8l/weights/best.pt source=zmot/videos/AM01-1.mp4 imgsz=640 project=outputs name=track-video-litter-1cls-yolov8l

# yolo track model=zmot/train-yolov8l-1cls-smtj/litter_1cls_yolov8l/weights/best.pt source=zmot/videos/AM01-1.mp4 imgsz=640 project=outputs name=track-video-litter-1cls-yolov8l-botsort tracker=trackers/botsort.yaml

# yolo track model=zmot/train-yolov8l-1cls-smtj/litter_1cls_yolov8l/weights/best.pt source=zmot/videos/AM01-1.mp4 imgsz=640 project=outputs name=track-video-litter-1cls-yolov8l-bytetrack tracker=trackers/bytetrack.yaml

yolo track model=zmot/train-yolov10l-1cls/litter_1cls_yolov10l/weights/yolov10l.pt source=zmot/videos/AM01-1.mp4 imgsz=640 project=outputs name=track-video-litter-1cls-yolov10l-botsort
