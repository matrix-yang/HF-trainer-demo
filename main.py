from data_process import CustomDatasetDataset
from modeling_cls import CustomModel
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
import torch


def collate_fn(examples):
    pixel_values = torch.stack([example['x'] for example in examples])
    labels = torch.tensor([example['labels'] for example in examples])
    max_len = max([example['texts']['attention_mask'].sum() for example in examples])
    input_ids = torch.stack([example['texts']['input_ids'][0][:max_len] for example in examples])
    token_type_ids = torch.stack([example['texts']['token_type_ids'][0][:max_len] for example in examples])
    attention_mask = torch.stack([example['texts']['attention_mask'][0][:max_len] for example in examples])
    input_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
    return {"x": pixel_values, "labels": labels, "input_dict": input_dict}


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        pix_values = inputs["x"]
        text = inputs["input_dict"]
        target = inputs["labels"]
        outputs = model(pix_values, text)
        loss = F.binary_cross_entropy(outputs, target)

        return (loss, outputs) if return_outputs else loss


if __name__ == '__main__':

    model = CustomModel()
    train_file = '/ms/FM/ydq/vision_cls/cls_data_train.yaml'
    test_file = '/ms/FM/ydq/vision_cls/cls_data_test.yaml'
    custom_dataset = CustomDatasetDataset(train_file)
    test_dataset = CustomDatasetDataset(test_file)
    # 冻结vision tower
    for name, para in model.named_parameters():
        if name.startswith('model.'):
            para.requires_grad = False

    training_args = TrainingArguments(
        output_dir='test_trainer',
        learning_rate=1e-4,
        per_device_train_batch_size=128,
        num_train_epochs=1,
        evaluation_strategy="epoch",
        remove_unused_columns=False,
        bf16=True,
        warmup_steps=200,
        save_steps=300,
        save_total_limit=5,
        logging_strategy='steps',
        logging_steps=10
    )

    trainer = MyTrainer(
        model,
        training_args,
        train_dataset=custom_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
    )

    trainer.train()