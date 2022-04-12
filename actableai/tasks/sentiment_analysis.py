from actableai.tasks import TaskType
from actableai.tasks.base import AAITask


class AAISentimentAnalysisTask(AAITask):
    """
    Sentiment Analysis Task
    """

    @AAITask.run_with_ray_remote(TaskType.SENTIMENT_ANALYSIS)
    def run(self, df, target, batch_size=32):
        """
        TODO write documentation
        """
        try:
            import math
            import time
            import traceback
            import ray
            from ray import serve
            from nltk.tokenize import sent_tokenize
            from actableai.sentiment.SpanABSA import AAISentimentExtractor

            start = time.time()

            # To resolve any issues of acces rights make a copy
            df = df.copy()

            # Get serve handle
            span_absa_handle = AAISentimentExtractor.get_handle()

            row_ids, sentences = [], []
            for i, doc in enumerate(df[target].to_list()):
                if doc is not None:
                    for paragraph in doc.split("</br>"):
                        for sent in sent_tokenize(paragraph):
                            row_ids.append(i)
                            sentences.append(sent)

            # Call the deployed model batch by batch
            result = []
            for batch_index in range(math.ceil(len(sentences) / batch_size)):
                sentence_batch = sentences[
                    batch_index * batch_size : (batch_index + 1) * batch_size
                ]
                result += ray.get(
                    span_absa_handle.options(method_name="predict").remote(
                        sentence_batch
                    )
                )

            data = []
            for i, re in enumerate(result):
                for kw, sentiment in zip(re["keyword"], re["sentiment"]):
                    data.append(
                        {
                            "keyword": kw,
                            "sentiment": sentiment,
                            "sentence": sentences[i],
                            "row": row_ids[i],
                        }
                    )

            return {"data": data, "status": "SUCCESS", "runtime": time.time() - start}
        except Exception as err:
            return {
                "status": "FAILURE",
                "message": str(err),
                "trace": traceback.format_exc(),
            }
