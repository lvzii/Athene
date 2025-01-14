import nlpertools
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import evaluate
import numpy as np
from transformers import AutoTokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

id2label = ["O", "B-entity1", "I-entity1", "B-entity2", "I-entity2", "B-null", "I-null"]
label2id = {i: idx for idx, i in enumerate(id2label)}


def main(pt_path):
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        for name in ["entity1", "entity2", "null"]:
            tp, fn, fp, tn = 0, 0, 0, 0

            for sentence_idx in range(len(true_labels)):
                label_list = true_labels[sentence_idx]
                pred_list = true_predictions[sentence_idx]
                true_entities_list = set()
                pred_entities_list = set()
                for idx, i in enumerate(label_list):
                    if i == f"B-{name}":
                        end_idx = idx + 1
                        while end_idx < len(label_list) and label_list[end_idx] == f"I-{name}":
                            end_idx += 1
                        true_entities_list.add((idx, end_idx))
                    if pred_list[idx] == f"B-{name}":
                        end_idx = idx + 1
                        while end_idx < len(label_list) and pred_list[end_idx] == f"I-{name}":
                            end_idx += 1
                        pred_entities_list.add((idx, end_idx))
                tp += len(true_entities_list.intersection(pred_entities_list))
                fp += len(pred_entities_list - true_entities_list)
                fn += len(true_entities_list - pred_entities_list)
            print(name)
            try:
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f = 2 * p * r / (p + r)
                print("p", p)
                print("r", r)
                print("f", f)
            except:
                print(0)

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["text"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples["ner_tag"]):
            label = [label2id[k] for k in label]
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    df = pd.DataFrame(nlpertools.load_from_jsonl(r"./data/train.json"))
    nong_train = Dataset.from_pandas(df)
    df = pd.DataFrame(nlpertools.load_from_jsonl(r"./data/test.json"))
    nong_test = Dataset.from_pandas(df)

    nong = DatasetDict()
    nong["train"] = nong_train
    nong["test"] = nong_test
    tokenizer = AutoTokenizer.from_pretrained(pt_path)

    tokenized_nong = nong.map(tokenize_and_align_labels, batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    seqeval = evaluate.load("seqeval")

    model = AutoModelForTokenClassification.from_pretrained(
        pt_path, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="output/{}".format(pt_path.split("/")[-1]),
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=7,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_nong["train"],
        eval_dataset=tokenized_nong["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    # pt_paths = ["../pretrained_model/bert-base-chinese", "SIKU-BERT/sikuroberta", "SIKU-BERT/sikubert"]
    # pt_path = "SIKU-BERT/sikuroberta"
    pt_path = "SIKU-BERT/sikubert"
    # for pt_path in pt_paths:
    main(pt_path)
