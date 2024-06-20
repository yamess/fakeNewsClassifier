import torch

class TxtDataset:
    def __init__(self, title, text, target, tokenizer, max_len=512):
        self.title = title
        self.text = text
        self.target = target
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text + self.title)

    def __getitem__(self, item):
        text = str(self.title[item]) + ". " + str(self.text[item])
        # target = int(self.target[item])

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            return_attention_mask=True
        )

        input_ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target": torch.tensor(self.target[item], dtype=torch.float)
        }
        return out

