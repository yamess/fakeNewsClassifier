import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
import os
import copy
import torch
from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments
from fakenews.fakedataset import TxtDataset
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import AdamW
from fakenews import config
from fakenews.model import BertClassifier
from fakenews.engine import train, evaluate

from transformers import WEIGHTS_NAME, CONFIG_NAME

def main(data):
    # X = data.text.values
    X = data[["title", "text"]]
    y = data.target.values

    if os.path.exists(config.CHECKPOINT):
        tokenizer = BertTokenizer.from_pretrained(config.OUTPUT_DIR)
    else:
        tokenizer = BertTokenizer.from_pretrained(config.BERT_TOKENIZER_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    trainDataset = TxtDataset(
        title=X_train.title.values,
        text=X_train.text.values,
        target=y_train,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )
    trainDataLoader = DataLoader(
        dataset=trainDataset,
        sampler=RandomSampler(trainDataset),
        batch_size=config.TRAIN_BATCH_SIZE
    )

    testDataset = TxtDataset(
        title=X_test.title.values,
        text=X_test.text.values,
        target=y_test,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )
    testDataLoader = DataLoader(
        dataset=testDataset,
        sampler=SequentialSampler(testDataset),
        batch_size=config.VALID_BATCH_SIZE
    )

    best_test_acc = 0.0
    best_epoch = 1
    # output_model_file = os.path.join(config.OUTPUT_DIR, WEIGHTS_NAME)
    # output_config_file = os.path.join(config.OUTPUT_DIR, CONFIG_NAME)

    if os.path.exists(config.OUTPUT_MODEL_FILE):
        # model = BertForSequenceClassification.from_pretrained(config.OUTPUT_DIR)
        state_dict = torch.load(config.OUTPUT_MODEL_FILE)
        model = BertClassifier(config.BERT_MODEL_PATH,config.DROPOUT,n_class=1)
        model.load_state_dict(state_dict)
        model.to(config.DEVICE)
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

        print("Loading model from last checkpoint...")
        state = torch.load(config.CHECKPOINT)
        best_test_acc = state['best_test_acc']
        best_epoch = state['epoch']
        print(f"Best test Accuracy so far: {best_test_acc:.3f} at Epoch {best_epoch}\n")
    else:
        # model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        model = BertClassifier(config.BERT_MODEL_PATH,config.DROPOUT,n_class=1)
        # print the number of parameter of the model before freezing
        print(
            f"Nbr of parameters before freezing bert layers: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        # Freezing the bert layers so that we will only train the top layer (classifier)
        for name,param in model.named_parameters():
            if name.startswith('bert'):
                param.requires_grad = False
        # Print the number of parameters of the model after freezing the bert layers
        print(
            f"Nbr of parameters after freezing bert layers: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        model.to(config.DEVICE)
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):
        real_epoch = best_epoch + epoch
        train_loss, train_acc = train(model, optimizer, trainDataLoader)
        test_loss, test_acc = evaluate(model, testDataLoader)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(
                f"Epoch {real_epoch: <{5}} | Train loss {train_loss:8.3f}| Train acc {train_acc:8.3f} | Valid loss {test_loss:8.3f} | Valid acc {test_acc:8.3f} | + "
            )

            # SAVING MODEL
            model_to_save = model.module if hasattr(model, 'module') else model

            torch.save(model_to_save.state_dict(), config.OUTPUT_MODEL_FILE)   # Saving the model
            tokenizer.save_pretrained(config.OUTPUT_DIR)        # Saving the pretrained tokenizer vocab

            checkpoint = {
                'epoch': real_epoch,
                'best_test_acc': best_test_acc,
            }
            torch.save(checkpoint, config.CHECKPOINT)
        else:
            print(
                f'Epoch {real_epoch: <{5}} | Train loss {train_loss:8.3f}| Train acc {train_acc:8.3f} | Valid loss {test_loss:8.3f} | Valid acc {test_acc:8.3f} |'
            )
    print(f"The best Model Accuracy: {best_test_acc}")
    print("The best Model has been saved")


if __name__ == '__main__':
    df = pd.read_csv("data.csv")
    main(df)
