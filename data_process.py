from torch.utils.data import Dataset
from PIL import Image
import yaml
import jsonlines
import json
import random
import torch
import timm
from transformers import BertTokenizer, BertModel


def load_data(in_file, file_format=''):
    if not file_format:
        if in_file.endswith(('jsonl', 'jsonline', 'jsonlines')):
            file_format = 'jsonl'
        elif in_file.endswith('json'):
            file_format = 'json'
        elif in_file.endswith('csv'):
            file_format = 'csv'
        elif in_file.endswith(('xls', 'xlsx')):
            file_format = 'xls'
        elif in_file.endswith(('yaml', 'yml')):
            file_format = 'yaml'
        else:
            assert False, f'unknown file format for {in_file}'

    if file_format == 'jsonl':
        with jsonlines.open(in_file) as reader:
            data = [obj for obj in reader]
            return data
    elif file_format == 'json':
        with open(in_file, 'r', encoding='utf-8') as fp:
            return json.load(fp)
    elif file_format == 'csv':
        df = pd.read_csv(in_file)
        data = []
        for _, row in df.iterrows():
            data.append(row.to_dict())

        return data
    elif file_format == 'xls':
        df = pd.read_excel(in_file)
        data = []
        for _, row in df.iterrows():
            data.append(row.to_dict())

        return data
    elif file_format == 'yaml':
        with open(in_file, 'r', encoding='utf-8') as fp:
            files = yaml.safe_load(fp)

        data = []
        for f in files['data']:
            data += load_data(f)

        return data
    else:
        assert False, f'unknown file format {file_format} for {in_file}'


class CustomDatasetDataset(Dataset):
    def __init__(self, in_file):
        # in_file = '/huawei-data/FM/lhb/llava/configs/test_data.yaml'
        # in_file = '/ms/FM/ydq/vision_cls/cls_data.yaml'
        model_cfg = timm.get_pretrained_cfg('vit_large_patch14_reg4_dinov2.lvd142m')
        data_cfg = timm.data.resolve_data_config(pretrained_cfg=model_cfg.__dict__)
        data_cfg['input_size'] = (3, 384, 384)
        transform = timm.data.create_transform(**data_cfg, is_training=False)
        self.data = load_data(in_file)
        self.transform = transform
        bert_path = '/huawei-data/FM/checkpoints/hfl/chinese-roberta-wwm-ext'
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.generate_data()

    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return len(self.data_pairs)

    def generate_data(self):
        pairs = []
        for d in self.data:
            img_path = d['image']
            text = d['conversations'][-1]['value']
            if '|' in text:
                text = text.split('|')[1]
            pairs.append((img_path, text, 1.0))

        text_bag = list(set([t for i, t, l in pairs]))

        false_pairs = []
        for i, t, l in pairs:
            false_text = self.get_false_text(t, text_bag)
            false_pairs.append((i, false_text, 0.0))

        self.data_pairs = pairs + false_pairs

    def get_false_text(self, true_text, text_bag):

        text = random.choice(text_bag)
        while text == true_text:
            text = random.choice(text_bag)
            # print(f'false {text} \ntrue {true_text}')
        return text

    def __getitem__(self, idx):
        """
        获取数据集中的样本。
        :param idx: 样本的索引。
        """
        img_path, text, label = self.data_pairs[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            # print(text)
        input_dict = self.tokenizer.batch_encode_plus([text], padding='max_length', max_length=50,
                                                      return_tensors='pt')
        return {'x': image, "labels": label, "texts": input_dict}


def collate_fn(examples):
    pixel_values = torch.stack([example['x'] for example in examples]).cuda()
    labels = torch.tensor([example['labels'] for example in examples]).cuda()
    max_len = max([example['texts']['attention_mask'].sum() for example in examples])
    input_ids = torch.stack([example['texts']['input_ids'][0][:max_len] for example in examples]).cuda()
    token_type_ids = torch.stack([example['texts']['token_type_ids'][0][:max_len] for example in examples]).cuda()
    attention_mask = torch.stack([example['texts']['attention_mask'][0][:max_len] for example in examples]).cuda()
    input_dict = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}
    return {"x": pixel_values.cuda(), "labels": labels.cuda(), "input_dict": input_dict}
