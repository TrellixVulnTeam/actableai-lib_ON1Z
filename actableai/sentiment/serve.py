import traceback
import logging
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class AAISentimentExtractor:
    """
    TODO write documentation
    """

    @classmethod
    def deploy(
        cls,
        num_replicas,
        ray_options,
        pyabsa_checkpoint,
        device,
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
            init_args=(pyabsa_checkpoint, device),
        ).deploy()

    @classmethod
    def get_handle(cls):
        """
        TODO write documentation
        """
        return cls.get_deployment().get_handle()

    @classmethod
    def get_deployment(cls):
        """
        TODO write documentation
        """
        from ray import serve

        return serve.get_deployment(cls.__name__)

    def __init__(self, checkpoint, device) -> None:
        """
        TODO write documentation
        """
        import multi_rake
        from pyabsa import APCCheckpointManager

        self.rake = multi_rake.Rake(min_freq=1)
        self.sent_classifier = APCCheckpointManager.get_sentiment_classifier(
            checkpoint=checkpoint,
            auto_device=device,
        )

    def predict(self, X, rake_threshold=1.0):
        """
        TODO write documentation
        """
        keywords, candidates = self.rake.apply_sentences(X)
        keywords = set([kw[0].text for kw in keywords if kw[1] >= rake_threshold])
        results = [
            {"keyword": [], "sentiment": [], "confidence": []} for i in range(len(X))
        ]
        annotated_sents = []
        extracted_candidates = []
        for candidate in candidates:
            if candidate.text in keywords:
                sent = X[candidate.sentence_id]
                annotated_sent = (
                    sent[: candidate.start_position]
                    + "[ASP]"
                    + sent[candidate.start_position : candidate.end_position]
                    + "[ASP]"
                    + sent[candidate.end_position :]
                )
                annotated_sents.append(annotated_sent)
                extracted_candidates.append(candidate)

        samples = []
        for sent in annotated_sents:
            samples.extend(self.sent_classifier.dataset.parse_sample(sent))
        self.sent_classifier.dataset.process_data(samples)
        self.sent_classifier.infer_dataloader = DataLoader(
            dataset=self.sent_classifier.dataset,
            batch_size=self.sent_classifier.opt.eval_batch_size,
            shuffle=False)

        try:
            predictions = self.sent_classifier._infer(print_result=False)
            sentiments, confidences = [], []
            for s in predictions:
                sentiments.extend(s["sentiment"])
                confidences.extend(s["confidence"])

            assert len(sentiments) == len(extracted_candidates)
            assert len(sentiments) == len(confidences)

            for candidate, s, conf in zip(extracted_candidates,
                                          sentiments,
                                          confidences):
                results[candidate.sentence_id]["keyword"].append(candidate.text)
                results[candidate.sentence_id]["sentiment"].append(s.lower())
                results[candidate.sentence_id]["confidence"].append(conf)
        except Exception:
            logger.error(
                "Error in analyzing text: %s\n%s" % (text, traceback.format_exc()))

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
