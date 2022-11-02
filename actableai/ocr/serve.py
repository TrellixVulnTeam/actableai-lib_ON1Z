from PIL import Image
from io import BytesIO
import base64

from actableai.serve.abstract_serve import AbstractRayDeployment
import starlette.requests

class AAIOCRExtractor(AbstractRayDeployment):
    def __init__(self):
        """
        TODO write documentation
        """
        from vietocr.tool.predictor import Predictor
        from vietocr.tool.config import Cfg

        config = Cfg.load_config_from_name("vgg_transformer")
        # config['weights'] = './weights/transformerocr.pth'
        config[
            "weights"
        ] = "https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA"
        config["cnn"]["pretrained"] = False
        config["device"] = "cuda:0"
        config["predictor"]["beamsearch"] = False

        self.detector = Predictor(config)

    def predict(self, img_byte):
        """
        TODO write documentation
        """
        img = Image.open(img_byte)
        return self.detector.predict(img)

    def __call__(self, request: starlette.requests.Request):
        request_body = request.json()
        img_base64 = request_body["img"]
        img = BytesIO(base64.b64decode(img_base64))
        return self.predict(img)
