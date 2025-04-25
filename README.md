windlesssecurity@raspberrypi:~/hailo_model_zoo $ yolo export model=best.pt format=onnx imgz=320
Traceback (most recent call last):
  File "/home/windlesssecurity/.local/bin/yolo", line 8, in <module>
    sys.exit(entrypoint())
             ^^^^^^^^^^^^
  File "/home/windlesssecurity/.local/lib/python3.11/site-packages/ultralytics/cfg/__init__.py", line 914, in entrypoint
    check_dict_alignment(full_args_dict, overrides)
  File "/home/windlesssecurity/.local/lib/python3.11/site-packages/ultralytics/cfg/__init__.py", line 499, in check_dict_alignment
    raise SyntaxError(string + CLI_HELP_MSG) from e
SyntaxError: 'imgz' is not a valid YOLO argument. Similar arguments are i.e. ['imgsz=640'].

    Arguments received: ['yolo', 'export', 'model=best.pt', 'format=onnx', 'imgz=320']. Ultralytics 'yolo' commands use the following syntax:

        yolo TASK MODE ARGS

        Where   TASK (optional) is one of frozenset({'detect', 'classify', 'pose', 'segment', 'obb'})
                MODE (required) is one of frozenset({'train', 'track', 'benchmark', 'val', 'predict', 'export'})
                ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
                    See all ARGS at https://docs.ultralytics.com/usage/cfg or with 'yolo cfg'

    1. Train a detection model for 10 epochs with an initial learning_rate of 0.01
        yolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01

    2. Predict a YouTube video using a pretrained segmentation model at image size 320:
        yolo predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

    3. Val a pretrained detection model at batch-size 1 and image size 640:
        yolo val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640

    4. Export a YOLO11n classification model to ONNX format at image size 224 by 128 (no TASK required)
        yolo export model=yolo11n-cls.pt format=onnx imgsz=224,128

    5. Ultralytics solutions usage
        yolo solutions count or in ['crop', 'blur', 'workout', 'heatmap', 'isegment', 'visioneye', 'speed', 'queue', 'analytics', 'inference', 'trackzone'] source="path/to/video.mp4"

    6. Run special commands:
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        yolo solutions help

    Docs: https://docs.ultralytics.com
    Solutions: https://docs.ultralytics.com/solutions/
    Community: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    
