class AAISentimentExtractor:
    """
    TODO write documentation
    """

    @classmethod
    def deploy(
        cls,
        num_replicas,
        ray_options,
        device,
        BERT_DIR,
        EXTRACT_MODEL_DIR,
        CLASSIFICATION_MODEL_DIR,
    ):
        """
        TODO write documentation
        """
        from ray import serve

        return serve.deployment(
            cls,
            name=cls.__name__,
            num_replicas=num_replicas,
            ray_actor_options=ray_options,
            init_args=(device, BERT_DIR, EXTRACT_MODEL_DIR, CLASSIFICATION_MODEL_DIR),
        ).deploy()

    @classmethod
    def get_handle(cls):
        """
        TODO write documentation
        """
        from ray import serve

        return cls.get_deployment().get_handle()

    @classmethod
    def get_deployment(cls):
        """
        TODO write documentation
        """
        from ray import serve

        return serve.get_deployment(cls.__name__)

    def __init__(
        self, device, BERT_DIR, EXTRACT_MODEL_DIR, CLASSIFICATION_MODEL_DIR
    ) -> None:
        """
        TODO write documentation
        """
        import torch
        import os
        from actableai.third_parties.spanABSA.bert.sentiment_modeling import (
            BertForSpanAspectExtraction,
            BertForSpanAspectClassification,
        )
        import actableai.third_parties.spanABSA.bert.tokenization as tokenization
        from actableai.third_parties.spanABSA.bert.modeling import BertConfig
        from actableai.third_parties.spanABSA.span_absa_helper import load_model

        self.device = (
            device  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.BERT_DIR = BERT_DIR  # "/app/superset/prediction/models/bert-base-uncased"
        self.BERT_INIT_MODEL_DIR = os.path.join(self.BERT_DIR, "pytorch_model.bin")
        self.EXTRACT_MODEL_DIR = (
            EXTRACT_MODEL_DIR  # "/app/superset/prediction/models/extract.pth.tar"
        )
        self.CLASSIFICATION_MODEL_DIR = (
            CLASSIFICATION_MODEL_DIR  # "/app/superset/prediction/models/cls.pth.tar"
        )

        n_gpu = torch.cuda.device_count()
        bert_config = BertConfig.from_json_file(
            os.path.join(self.BERT_DIR, "bert_config.json")
        )

        self.extract_model = load_model(
            BertForSpanAspectExtraction,
            bert_config,
            self.BERT_INIT_MODEL_DIR,
            self.EXTRACT_MODEL_DIR,
            self.device,
            n_gpu,
        )
        self.cls_model = load_model(
            BertForSpanAspectClassification,
            bert_config,
            self.BERT_INIT_MODEL_DIR,
            self.CLASSIFICATION_MODEL_DIR,
            self.device,
            n_gpu,
        )

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=os.path.join(self.BERT_DIR, "vocab.txt"), do_lower_case=True
        )

    def predict(self, X, row_id=None):
        """
        TODO write documentation
        """
        from actableai.third_parties.spanABSA.span_absa_helper import (
            detect as abas_detect,
        )

        results = []
        for sentence in X:
            terms, label = abas_detect(
                sentence,
                self.tokenizer,
                self.extract_model,
                self.cls_model,
                self.device,
            )
            res = {"keyword": terms, "row_id": row_id, "sentiment": label}
            results.append(res)
        return results


if __name__ == "__main__":
    sentences = [
        "The bread is top notch as well.",
        "Certainly not the best sushi in New York, however, it is always fresh, and the place is very clean, sterile.",
        "I love the drinks, esp lychee martini, and the food is also VERY good.",
        "In fact, this was not a Nicoise salad and wa`s barely eatable.",
        "While there's a decent menu, it shouldn't take ten minutes to get your drinks and 45 for a dessert pizza.",
        "Our waiter was horrible; so rude and disinterested.",
        "We enjoyed ourselves thoroughly and will be going back for the desserts ....",
        "I definitely enjoyed the food as well.#",
        "WE ENDED UP IN LITTLE ITALY IN LATE AFTERNOON.",
    ]

    m = AAISentimentExtractor()
    results = m.predict(sentences)

    print(results)
