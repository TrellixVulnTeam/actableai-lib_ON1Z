import traceback
import logging
from typing import Optional, List

from actableai.serve.abstract_serve import AbstractRayDeployment

logger = logging.getLogger(__name__)


class AAISentimentExtractor(AbstractRayDeployment):
    """
    TODO write documentation
    """
    def __init__(
        self,
        checkpoint,
        device,
        flair_pos_model_path: Optional[str] = None,
        flair_pos_supported_language_codes: Optional[List[str]] = None,
    ) -> None:
        """
        TODO write documentation
        """
        import multi_rake
        from multi_rake.pos_tagger import MultilingualPOSTagger
        from pyabsa import APCCheckpointManager

        tagger = MultilingualPOSTagger(
            flair_pos_model_path, flair_pos_supported_language_codes
        )
        self.rake = multi_rake.Rake(min_freq=1, tagger=tagger)
        self.sent_classifier = APCCheckpointManager.get_sentiment_classifier(
            checkpoint=checkpoint, auto_device=device
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

                try:
                    sentiment = self.sent_classifier.infer(
                        annotated_sent, print_result=False
                    )
                    results[candidate.sentence_id]["keyword"].append(candidate.text)
                    results[candidate.sentence_id]["sentiment"].append(
                        sentiment["sentiment"][0].lower()
                    )
                    results[candidate.sentence_id]["confidence"].append(
                        sentiment["confidence"][0]
                    )
                except Exception:
                    logger.error(
                        "Error in analyzing sentence: %s\n%s"
                        % (annotated_sent, traceback.format_exc())
                    )

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
