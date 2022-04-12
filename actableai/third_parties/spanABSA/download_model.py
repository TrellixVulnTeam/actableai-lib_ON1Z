import os
from actableai.third_parties.spanABSA import bucket

bucket = bucket.bucket


def download_spanabsa(spanabs_dir="/app/superset/prediction/models/"):
    try:
        bert_file = "bert-base-uncased.zip"
        cls_file = "cls.pth.tar"
        extr_file = "extract.pth.tar"
        bert_model = spanabs_dir + bert_file
        cls_model = spanabs_dir + cls_file
        extr_model = spanabs_dir + extr_file
        os.makedirs(spanabs_dir)
    except:
        print("Folder satisfied")

    if not os.path.isfile(bert_model):
        print("Starting download!")
        bucket.download_file("SpanABSA/bert-base-uncased.zip", bert_model)
        print("Unziping!")
        os.system(f"unzip -o {bert_model} -d {spanabs_dir}")

    if not os.path.isfile(cls_model):
        print("Starting download cls!")
        bucket.download_file("SpanABSA/cls.pth.tar", cls_model)

    if not os.path.isfile(extr_model):
        print("Starting download extract!")
        bucket.download_file("SpanABSA/extract.pth.tar", extr_model)


if __name__ == "__main__":
    download_spanabsa()
