from sahi.utils.yolov5 import (
    download_yolov5s6_model,
)

# import required functions, classes
from sahi.model import Yolov5DetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predictor import get_prediction, get_sliced_prediction, predict
from nanoslice import NanoSlice
# download YOLOV5S6 model to 'models/yolov5s6.pt'
yolov5_model_path = 'models/yolov5s6.pt'
download_yolov5s6_model(destination_path=yolov5_model_path)

# download test images into demo_data folder
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data/small-vehicles1.jpeg')
download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png', 'demo_data/terrain2.png')
detection_model = NanoSlice(
    model_path = "/home/thor/nanodet_model/mode.ckpt",
    # model: Optional[Any] = None,
    config_path = "/home/thor/nanodet_model/config.yml",
    load_at_init = True,
    image_size = 416,
    device="cpu"
)


'''
detection_model = Yolov5DetectionModel(
    model_path=yolov5_model_path,
    confidence_threshold=0.3,
    device="cpu", # or 'cuda:0'
)
'''
result = get_sliced_prediction("mcd180.jpeg", detection_model)
import pdb;pdb.set_trace()
print(result)
