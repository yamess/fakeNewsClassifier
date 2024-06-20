from tqdm import tqdm
from fakenews import config
import torch.nn.functional as F
from torch import nn
import torch
from sklearn.metrics import accuracy_score

 
def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)


def train(model, optimizer, data_loader):
    total_loss = 0
    correct = 0
    total = 0

    model.train()
    for idx, d in enumerate(tqdm(data_loader)):
        ids = d["input_ids"].to(config.DEVICE)
        mask = d["mask"].to(config.DEVICE)
        token_type_ids = d["token_type_ids"].to(config.DEVICE)
        y_true = d["target"].to(config.DEVICE)

        optimizer.zero_grad()
        logits = model(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        logits = logits.view(-1)
        y_true = y_true.view(-1)

        loss = loss_fn(logits, y_true)
        loss.backward()
        optimizer.step()

        y_pred = torch.sigmoid(logits).cpu().detach().numpy() > 0.5
        y_true = y_true.cpu().detach().numpy()

        y_pred = y_pred.astype(int)     # Convert the boolean to int
        y_true = y_true.astype(int)     # Convert float to int

        total_loss += loss.item()
        correct += (y_pred == y_true).sum()
        total += len(y_true)

    avg_loss = total_loss/len(data_loader)
    acc = correct/total
    return avg_loss, acc

def evaluate(model, data_loader):
    total_loss = 0
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for _, d in enumerate(tqdm(data_loader)):
            ids = d["input_ids"].to(config.DEVICE)
            mask = d["mask"].to(config.DEVICE)
            token_type_ids = d["token_type_ids"].to(config.DEVICE)
            y_true = d["target"].to(config.DEVICE)

            logits = model(
                input_ids=ids,
                attention_mask=mask,
                token_type_ids=token_type_ids
            )
            logits = logits.view(-1)
            y_true = y_true.view(-1)

            loss = loss_fn(logits, y_true)

            y_pred = torch.sigmoid(logits).cpu().detach().numpy() > 0.5
            y_true = y_true.cpu().detach().numpy()

            y_pred = y_pred.astype(int)  # Convert the boolean to int
            y_true = y_true.astype(int)  # Convert float to int

            total_loss += loss.item()
            correct += (y_pred == y_true).sum()
            total += len(y_true)

        avg_loss = total_loss / len(data_loader)
        acc = correct/total
    return avg_loss, acc

def preprocess(title, text, tokenizer):
    
    inputs = tokenizer.encode_plus(
        str(title) + ". " + str(text),
        add_special_tokens=True,
        max_length=config.MAX_LEN,
        pad_to_max_length=True,
        return_token_type_ids=True,
        return_attention_mask=True
    )
    
    input_ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    
    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        "attention_mask": torch.tensor(mask, dtype=torch.long).unsqueeze(0),
        "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)
    }
    
    return out

def predict(inputs, model):
    """
    1: Means Real news
    0: Means Fake News
    """
    model.eval()
    with torch.no_grad():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"]
        )
        prob = torch.sigmoid(logits).cpu().detach().numpy().item()
        y_pred = int(prob> 0.5)
        logits = logits.cpu().detach().numpy().item()
        y_pred_text = "Real News" if y_pred==0 else "True News"

        predictions = {
            "logits": logits,
            "predictionValue": y_pred,
            "predictionText": y_pred_text,
            "predictionProbability": prob
        }
        return predictions

def onnx_predict(inputs, session):
    """
    1: Means Real news
    0: Means Fake News
    """
    with torch.no_grad():
        logits = session.run(
            None, 
            {
                "input_ids":inputs["input_ids"].detach().cpu().numpy(),
                "attention_mask":inputs["attention_mask"].detach().cpu().numpy(),
                "token_type_ids":inputs["token_type_ids"].detach().cpu().numpy()
            }
        )
        logits = torch.tensor(logits).flatten()
        
        prob = torch.sigmoid(logits).item()
        y_pred = int(prob> 0.5)
        y_pred_text = "Fake News" if y_pred==0 else "Real News"
        logits = logits.item()
        
        predictions = {
            "logits": logits,
            "predictionValue": y_pred,
            "predictionText": y_pred_text,
            "predictionProbability": prob
        }
        return predictions