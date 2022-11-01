from PIL import Image


from actableai.serve.abstract_serve import AbstractRayDeployment


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

    def predict(self, img_path: str) -> str:
        """
        TODO write documentation
        """
        img = Image.open(img_path)
        return self.detector.predict(img)
