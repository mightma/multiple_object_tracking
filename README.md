# Multiple Objects Tracking with YOLO

This repository allows you to fine-tune YOLO models like YOLO-V8 and YOLO-V10 on your customized dataset, and then use tracking methods like BoTSORT or DeepOCSORT to track objects in videos. Combined with a general Re-ID model, you can count the number of objects in the video.

The entire pipeline consists of two main parts: 1) an image object detection model and 2) a tracking method. The image object detection model is trained on an image object detection dataset, enabling it to detect each object of interest. The tracking method then uses a Re-ID model to connect objects across adjacent frames. If you don't have a dataset to train the Re-ID model, you can use a general Re-ID model. Thus, the only model you need to train is the image object detection model.

This project heavily relies on Ultralytics and BOXMOT. For more details about these two toolboxes, see the following links:
* [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
* [https://github.com/mikel-brostrom/boxmot](https://github.com/mikel-brostrom/boxmot)

## Prerequisites

### SageMaker Notebook Setup

The scripts in this repository are designed to run on Amazon SageMaker. To run them, you need to set up a SageMaker Notebook Instance on AWS. Search for SageMaker on the AWS console, select the SageMaker service, and then choose Notebook - Notebook Instances from the left sidebar. Click the "Create Notebook Instance" button in the top-right corner to enter the configuration page. During the configuration, make sure to select an appropriate notebook instance type. Since you'll be running inference scripts on the notebook, we recommend using the `g5.2xlarge` instance type.

### Environment Setup

Clone this repository and install the dependencies:

```shell
git clone https://github.com/mightma/multiple_object_tracking
cd multiple_object_tracking/boxmot
pip install poetry
poetry install --with yolo  # Install boxmot + yolo dependencies
poetry shell  # Activate the newly created environment with the installed dependencies
```

## Training Object Detection Models

Ultralytics provides rich and convenient scripts to train object detection models on your customized dataset. The first step is to reorganize your dataset in a format that can be loaded by Ultralytics.

### Construct Dataset

The input dataset is stored at `./zmot/1000_samples/positive/`, where each image has a JSON-formatted annotation provided by the customer. Use the following script to convert the JSON annotations to a text format:

```shell
python3 construct_dataset_anno.py
```

As shown in the script, the dataset is split into a training set and a validation set with a 0.1 split ratio, stored at `./zmot/1000_samples/train2017` and `./zmot/1000_samples/val2017`, respectively. Each image has a text-formatted annotation like:

```plain
label x y w h
label x y w h
```

Where `label` is the class ID, and `x`, `y`, `w`, `h` are normalized between 0 and 1.

There is a YAML file at `boxmot/litter_1cls_dataset.yaml` that you can modify as needed.

#### Filter Out Small Objects in the Dataset

After constructing the dataset, you can filter out small objects with the following script:

```shell
bash run_preprocess_dataset.sh
```

The corresponding YAML file is at `boxmot/litter_1cls_thr1e-3_dataset.yaml`.

However, since the customer-provided dataset contains many small objects, even a very small threshold of 0.001 could filter out nearly half of the dataset, which could degrade the performance of the image object detection model. This is not recommended.

### Training

We strongly recommend training the model with a SageMaker Training Job. You can start a training job with [start_training_job.ipynb](start_training_job.ipynb). Before that, you need to upload the constructed dataset to S3, and the trained model will also be stored in S3. See the notebook for more details.

The training will start with the script at [boxmot/train.py](boxmot/train.py), where you can change the model type, dataset, and other hyperparameters.

## Inference

### Image Object Detection

Directly detect objects at the image level using the YOLO CLI:

```
yolo detect predict model={path/to/model/checkpoint.pt} \
    source={source} \
    imgsz=640 \
    project=outputs \
    name={folder_name_containing_results}
```

Where `{source}` could be the path to an image, video, or a directory containing images. See examples in [run_inference.sh](run_inference.sh). The visualized results will be stored in `outputs/{folder_name_containing_results}`.

Reference:
* [https://docs.ultralytics.com/modes/predict/](https://docs.ultralytics.com/modes/predict/)

### Multiple Objects Tracking

Download your trained model from S3 to your local machine.

#### Directly Track Objects at the Image Level Using the YOLO CLI

```
yolo track model={path/to/model/checkpoint.pt} \
    source={path/to/video} \
    imgsz=640 \
    project=outputs \
    name={folder_name_containing_results}
```

See examples in [run_inference_tracking.sh](run_inference_tracking.sh). The visualized results will be stored in `outputs/{folder_name_containing_results}`.

Reference:
* [https://docs.ultralytics.com/modes/track/](https://docs.ultralytics.com/modes/track/)

#### Another Way is to Use BOXMOT

```shell
cd boxmot
python3 tracking/track.py \
    --yolo-model {path/to/model/checkpoint.pt} \
    --tracking-method botsort \
    --source {path/to/video} \
    --project ../outputs \
    --name folder_name_containing_results \
    --save-txt \
    --save
```

The visualized results will be stored in `outputs/{folder_name_containing_results}`. You can filter out small objects by providing the `--wh_thr` argument, which will filter out objects whose height or width is smaller than the specified value in pixels. See more examples in [boxmot/run_tracking.sh](boxmot/run_tracking.sh).

BOXMOT provides rich support for state-of-the-art trackers like `BoTSORT`, `DeepOCSORT`, etc. See the official BOXMOT repository for more details.

To count the number of objects, the above script will generate a label text file for each frame of the input video. The text file consists of rows in the following format:

```plain
class_id x y w h {instance_id}
```

`{instance_id}` is optional and represents the ID of the instance if it has appeared before. For example:

```plain
0 0.504105 0.741854 0.0197888 0.0226776 11
0 0.339809 0.969106 0.0310313 0.0600562
```

You can count the number of instances by counting each box without an instance ID. We provide an example in [count_instances.py](count_instances.py). Run the script with:

```shell
python3 count_instances.py \
    --input_dir outputs/{folder_name_containing_results}
```
