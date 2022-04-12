#!/usr/bin/env python
# coding: utf-8

# !pip install torch==1.7.1
# !pip install allennlp==2.1.0

import os


from actableai.third_parties.spanABSA.absa.run_base import bert_load_state_dict
from actableai.third_parties.spanABSA.squad.squad_utils import _get_best_indexes

import collections
import torch


BERT_DIR = "sentiment/bert/bert-base-uncased"
BERT_INIT_MODEL_DIR = os.path.join(BERT_DIR, "pytorch_model.bin")
DATA_DIR = "sentiment/data/absa"
JOINT_MODEL_DIR = "sentiment/out/joint/01/checkpoint.pth.tar"
EXTRACT_MODEL_DIR = "extract.pth.tar"
CLASSIFICATION_MODEL_DIR = "cls.pth.tar"

# # Load model
def load_model(Model, bert_config, bert_init_model_dir, model_dir, device, n_gpu):
    model = Model(bert_config)
    model = bert_load_state_dict(
        model, torch.load(bert_init_model_dir, map_location=device)
    )
    model = model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    checkpoint = torch.load(model_dir, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


# # Evaluation Simplify For Production
def list_object(obj):
    print({k: getattr(obj, k) for k in obj.__dir__() if not k.startswith("__")})


max_seq_length = 96
max_term_num = 25  # number of terms in one sentence


class SourceData:
    def __init__(self, sentence, tokenizer):
        self.sentence = sentence
        self.tokens = tokenizer.tokenize(sentence)

    def __str__(self):
        return self.sentence

    def __repr__(self):
        return self.__str__()


class ExtractInferenceFeature:
    def __init__(
        self, source_data, tokens, input_ids, input_mask, segment_ids, token_to_orig_map
    ):
        self.source_data = source_data
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        # segment_ids used in squad_utils to do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        # it's not used yet
        self.segment_ids = segment_ids
        self.token_to_orig_map = token_to_orig_map

    def __str__(self):
        return str({"tokens": self.tokens})

    def __repr__(self):
        return self.__str__()


# Feature Extraction
def convert_source_data_to_feature(source_data, tokenizer, max_seq_length):
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(source_data.tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    # Account for [CLS] and [SEP] with "- 2"
    if len(all_doc_tokens) > max_seq_length - 2:
        all_doc_tokens = all_doc_tokens[0 : (max_seq_length - 2)]

    tokens = []
    token_to_orig_map = {}
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)

    for index, token in enumerate(all_doc_tokens):
        token_to_orig_map[len(tokens)] = tok_to_orig_index[index]
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return ExtractInferenceFeature(
        source_data, tokens, input_ids, input_mask, segment_ids, token_to_orig_map
    )


_ExtractInferencePrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"]
)

_ExtractInferenceResult = collections.namedtuple(
    "ExtractInferenceResult",
    ["terms", "start_indexes", "end_indexes", "original_feature"],
)


def predict_extract_inference(
    model,
    feature: ExtractInferenceFeature,
    device,
    n_best_size=20,
    max_answer_length=12,
    logit_threshold=7.5,
    use_heuristics=True,
):
    input_ids = torch.tensor(feature.input_ids, dtype=torch.long).unsqueeze(0)
    input_mask = torch.tensor(feature.input_mask, dtype=torch.long).unsqueeze(0)
    segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long).unsqueeze(0)

    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)

    with torch.no_grad():
        batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)

    start_logits = batch_start_logits[0].detach().cpu().tolist()
    end_logits = batch_end_logits[0].detach().cpu().tolist()

    start_indexes = _get_best_indexes(batch_start_logits[0], n_best_size)
    end_indexes = _get_best_indexes(batch_end_logits[0], n_best_size)

    predictions = []

    for start_index in start_indexes:
        for end_index in end_indexes:
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions.
            if start_index >= len(feature.tokens):
                continue
            if end_index >= len(feature.tokens):
                continue
            if start_index not in feature.token_to_orig_map:
                continue
            if end_index not in feature.token_to_orig_map:
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue
            start_logit = start_logits[start_index]
            end_logit = end_logits[end_index]
            if start_logit + end_logit < logit_threshold:
                continue

            predictions.append(
                _ExtractInferencePrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=start_logit,
                    end_logit=end_logit,
                )
            )

    if use_heuristics:
        predictions = sorted(
            predictions,
            key=lambda x: (
                x.start_logit + x.end_logit - (x.end_index - x.start_index + 1)
            ),
            reverse=True,
        )
    else:
        predictions = sorted(
            predictions,
            key=lambda x: (x.start_logit + x.end_logit),
        )

    results = []
    for i, pred_i in enumerate(predictions):
        tok_tokens = feature.tokens[pred_i.start_index : (pred_i.end_index + 1)]
        tok_text = " ".join(tok_tokens)

        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")
        tok_text = tok_text.replace(" # # ", "")
        results.append(
            _ExtractInferenceResult(
                terms=tok_text,
                start_indexes=pred_i.start_index,
                end_indexes=pred_i.end_index,
                original_feature=feature,
            )
        )

    final_texts = []
    final_results = []
    for result in sorted(results, key=lambda x: len(x.terms), reverse=True):
        tok_text = result.terms
        # avoid result is a partitial text from another result
        is_not_partitial = True
        for final_text in final_texts:
            if tok_text in final_text:
                is_not_partitial = False
                break
        if is_not_partitial:
            final_texts.append(tok_text)
            final_results.append(result)
            is_not_partitial = True

    return final_results


# Polarity Classifier
class PolarityClassifierFeature:
    def __init__(
        self,
        source_data,
        tokens,
        input_ids,
        input_mask,
        segment_ids,
        start_indexes,
        end_indexes,
        start_indexes_padding,
        end_indexes_padding,
    ):
        self.source_data = source_data
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        # segment_ids used in squad_utils to do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        # it's not used yet
        self.segment_ids = segment_ids
        self.start_indexes = start_indexes
        self.end_indexes = end_indexes
        self.start_indexes_padding = start_indexes_padding
        self.end_indexes_padding = end_indexes_padding


def extract_inference_result_to_feature(extract_inference_results, max_term_num):
    start_indexes = [result.start_indexes for result in extract_inference_results]
    end_indexes = [result.end_indexes for result in extract_inference_results]

    start_indexes_padding = [index for index in start_indexes]
    end_indexes_padding = [index for index in end_indexes]

    while len(start_indexes_padding) < max_term_num:
        start_indexes_padding.append(0)
        end_indexes_padding.append(0)

    feature = extract_inference_results[0].original_feature

    return PolarityClassifierFeature(
        source_data=feature.source_data,
        tokens=feature.tokens,
        input_ids=feature.input_ids,
        input_mask=feature.input_mask,
        segment_ids=feature.segment_ids,
        start_indexes=start_indexes,
        end_indexes=end_indexes,
        start_indexes_padding=start_indexes_padding,
        end_indexes_padding=end_indexes_padding,
    )


id_to_label = {0: "other", 1: "neutral", 2: "positive", 3: "negative", 4: "conflict"}


def predict_cls(model, feature: PolarityClassifierFeature, device):
    input_ids = torch.tensor(feature.input_ids, dtype=torch.long).unsqueeze(0)
    input_mask = torch.tensor(feature.input_mask, dtype=torch.long).unsqueeze(0)
    segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long).unsqueeze(0)
    span_starts = torch.tensor(
        feature.start_indexes_padding, dtype=torch.long
    ).unsqueeze(0)
    span_ends = torch.tensor(feature.end_indexes_padding, dtype=torch.long).unsqueeze(0)

    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    span_starts = span_starts.to(device)
    span_ends = span_ends.to(device)

    with torch.no_grad():
        batch_cls_logits = model(
            "inference",
            input_mask,
            input_ids=input_ids,
            token_type_ids=segment_ids,
            span_starts=span_starts,
            span_ends=span_ends,
        )

    cls_logits = batch_cls_logits[0].detach().cpu().numpy().argmax(axis=1).tolist()

    results = []
    for _, cls_logit in zip(feature.start_indexes, cls_logits):
        results.append(id_to_label[cls_logit])

    return results


# Final Detection
def detect(sentence, tokenizer, extract_model, cls_model, device):
    example = SourceData(sentence, tokenizer)
    feature = convert_source_data_to_feature(example, tokenizer, max_seq_length)
    final_result = predict_extract_inference(
        extract_model,
        feature,
        device,
        n_best_size=20,
        max_answer_length=12,
        logit_threshold=7.5,
        use_heuristics=True,
    )
    if len(final_result) == 0:
        return [], []
    extract_inference_feature = extract_inference_result_to_feature(
        final_result, max_term_num
    )
    label = predict_cls(cls_model, extract_inference_feature, device)
    return [t.terms for t in final_result], label
