import os
import time
import pandas as pd
import jieba  #中文文本操作库
import jieba.analyse as analyse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel


class CSVProcessor():
    def __init__(self,csv_file_path):
        self.df= pd.read_csv(csv_file_path)
 
    def generate_label_mapping(self, label_column):
        '''
        参数：
        csv_file_path(str): 字符格式的csv文件路径
        label_column(str): csv表格中标签列的字符格式列名

        功能:读取csv并根据读取到的标签信息,给每一个标签分配数字,返回生成的DataFrame和标签映射字典
        '''
        unique_labels = self.df[label_column].unique()
        label_mapping = {label: i for i, label in enumerate(unique_labels)}
        return label_mapping
    
 
    def label_numerization(self, label_mapping, label_column):
        '''
        参数：
        label_mapping(dict):标签映射字典
        label_column(str): csv表格中标签列的字符格式列名
        
        功能:根据输入将每个样本的标签映射为数字,填在df['label_num']中
        '''
        self.df['label_num'] = self.df[label_column].map(label_mapping)


    def fill_nan_with_value(self, fill_value='无'):
        '''
        参数：
        fill_value:填充内容
        '''
        self.df.fillna(fill_value, inplace=True)

    
    def str_length_normalization(self, column_name, length, fillchar='0'):
        self.df[column_name] = self.df[column_name].str[:length].str.pad(width=length, side='right', fillchar=fillchar)


    @staticmethod
    def merge_columns(row, columns_to_merge):
        values = [str(row[column]) for column in columns_to_merge if not pd.isnull(row[column])]
        return '/'.join(values)

    def apply_merge_to_columns(self, columns_to_merge):
        '''
        参数：
        columns_to_merge(list): 需要合并的列的字符串列名组成的数组

        功能:根据输入将需要各列合并,返回合并结果
        '''
        merged_result = self.df.apply(lambda row: self.merge_columns(row, columns_to_merge), axis=1)
        return merged_result

        
    @staticmethod
    def chinese_word_cut(text):
        seg_list = jieba.cut(text, cut_all=False)
        return ' '.join(seg_list)
    
    def apply_chinese_word_cut(self, column_name):
        '''
        参数：
        column_name(str): 需要对中文内容分词的字符串列名

        功能:根据输入将内容分词,返回分词结果
        '''
        cut_word = self.df[column_name].apply(self.chinese_word_cut)
        return cut_word


    @staticmethod
    def key_word_extract(text, top_keywords):
        return " ".join(analyse.extract_tags(text, topK=top_keywords, withWeight=False, allowPOS=()))

    def apply_keyword_extraction(self, column_name, top_keywords):
        '''
        参数：
        column_name(str): 需要对中文提取关键词的字符串列名
        top_keywords(int): 关键词提取结果的词数上限

        功能:根据输入将内容分词,返回关键词提取结果
        '''
        key_word = self.df[column_name].apply(lambda text: self.key_word_extract(text, top_keywords))
        return key_word


class TextTokenizer():
    def __init__(self, model_name, max_length):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def tokenize_dataframe(self, text_list):
        input_ids_list = []
        attention_mask_list = []
        
        for text in text_list:
            encoding = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_length,
                                          padding='max_length', truncation=True, return_tensors='pt')
            input_ids_list.append(encoding['input_ids'].squeeze().tolist())
            attention_mask_list.append(encoding['attention_mask'].squeeze().tolist())

        return input_ids_list, attention_mask_list


class TrainDataset(Dataset):
    def __init__(self, input_ids_tensor, attention_mask_tensor, label_tensor):
        self.input_ids_tensor = input_ids_tensor
        self.attention_mask_tensor = attention_mask_tensor
        self.label_tensor = label_tensor
        
    def __len__(self):
        return len(self.input_ids_tensor)
    
    def __getitem__(self, idx):
        input_ids = self.input_ids_tensor[idx]
        attention_mask = self.attention_mask_tensor[idx]
        label_tensor = self.label_tensor[idx]

        return input_ids, attention_mask, label_tensor

    def prepare_dataloaders(self, train_ratio, val_ratio, batch_size):
        '''
        参数:
        train_ratio(float):训练数据占总数据比例  如:0.8
        val_ratio(float):验证数据占总数据的比例
        batch_size:略

        功能:根据输入读取词张量和标签张量,按规定比例，随机生成并返回训练、验证和测试数据加载器
        '''

        dataset = TensorDataset(self.input_ids_tensor, self.attention_mask_tensor, self.label_tensor)
        total_size = self.__len__()
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        
        train_loader = DataLoader(train_dataset,batch_size,
                                    shuffle=True,num_workers=0,pin_memory=True)
        val_loader = DataLoader(val_dataset,batch_size,
                                    shuffle=False,num_workers=0,pin_memory=True)
        test_loader = DataLoader(test_dataset,batch_size,
                                    shuffle=False,num_workers=0,pin_memory=True)
        
        return train_loader, val_loader, test_loader
    

#已完成-----------------------------------------------------------------------------------------------------------------------------
class BERTVectorizer(nn.Module):
    def __init__(self, model_name, num_classes, device):
        super(BERTVectorizer, self).__init__()
        self.model_name = model_name
        self.num_labels = num_classes
        self.device = torch.device(device)
        self.bert_model = BertModel.from_pretrained(self.model_name).to(self.device)
        self.classifier = nn.Linear(1024, num_classes)  
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits


class CNNVectorizer():
    def __init__(self, max_length, model_name, num_labels, device):
        self.max_length = max_length
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = torch.device(device)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = BertModel.from_pretrained(self.model_name).to(self.device)
        self.conv_layer = nn.Conv1d(in_channels=1024, out_channels=128, kernel_size=3)
        self.classifier = nn.Linear(128, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = pooled_output.permute(0, 2, 1)  # 为了匹配卷积层的输入维度
        conv_output = self.conv_layer(pooled_output)
        conv_output = conv_output.squeeze(-1)
        logits = self.classifier(conv_output)
        return logits

#待修改-----------------------------------------------------------------------------------------------------------------------------

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, scheduler, epoches, model_save_path, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.epoches = epoches
        self.model_save_path = model_save_path
        self.best_val_acc = 0.0
        self.device = torch.device(device)
        self.model.to(self.device)

    def train_epoch(self,epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for _, (input_ids, attention_mask, labels) in enumerate(self.train_loader):
            input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / len(self.train_loader)
        train_acc = correct / total_samples
        return train_loss, train_acc

    def validation(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for input_ids, attention_mask, labels in self.val_loader:
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        val_loss = total_loss / len(self.val_loader)
        val_acc = correct / total_samples
        return val_loss, val_acc

    def save_model(self, model_path):
        if not os.path.exists('model'):
            os.makedirs('model')
        torch.save(self.model.state_dict(), model_path)

    def train(self):
        for epoch in range(self.epoches):
            start = time.time()
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validation()
            lr = self.optimizer.param_groups[0]['lr']
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                best_model_path = os.path.join('model', self.model_save_path)
                self.save_model(best_model_path)
            
            print('Epoch [{}/{}] Train Loss: {:.6f} Train Acc: {:.2f} Val Loss: {:.6f} Val Acc: {:.2f} Learning Rate: {:.6f}'.format(
                epoch + 1, self.epoches, train_loss, train_acc, val_loss, val_acc, lr))

            self.scheduler.step(val_acc)  # Pass validation accuracy to the scheduler
            end = time.time()
            time_taken = end - start
            m, s = divmod(int(time_taken), 60)
            print('\tTime: {:02d}:{:02d}'.format(m, s))


# ----------------------------------------------------------------------------------------------------------------------------
class ModelEvaluator:
    def __init__(self, model, test_loader, label_mapping, device):
        self.model = model
        self.test_loader = test_loader
        self.label_mapping = label_mapping
        self.device = torch.device(device)
    
    def test_accuracy(self):
        self.model.eval()
        correct = 0
        total = 0
        torch.cuda.empty_cache()

        with torch.no_grad():
            for input_ids, attention_mask, labels in self.test_loader:
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)

                outputs = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print('Accuracy of the network on the test items: %.2f %%' % accuracy)

    def accuracy_of_label(self):
        classes = [key for key in self.label_mapping.keys()]
        model = self.model.to(self.device)
        model.eval()

        num_classes = len(classes)
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        with torch.no_grad():
            for input_ids, attention_mask, labels in self.test_loader:
                input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)

                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                c = (predicted == labels).squeeze()
                for i in range(len(input_ids)):
                    label = labels[i]
                    class_correct[label] += c[i]
                    class_total[label] += 1

        for i in range(num_classes):
            if class_total[i] != 0:
                accuracy = 100 * class_correct[i] / class_total[i]
                print('Accuracy of %5s : %.2f %%' % (classes[i], accuracy))
            else:
                print('Accuracy of %5s : No samples in the test set' % (classes[i]))


class Prediction:
    def __init__(self, model, model_path, device):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, unlabeled_data_loader):
        predictions = []
        with torch.no_grad():
            for input_ids, attention_mask in unlabeled_data_loader:
                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                outputs = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        return predictions
    

class PredictionDataset(TrainDataset):
    '''
    用于预测的数据集构造操作
    '''
    def __init__(self, input_ids_tensor, attention_mask_tensor):
        self.input_ids_tensor = input_ids_tensor
        self.attention_mask_tensor = attention_mask_tensor
        
    def __len__(self):
        return len(self.input_ids_tensor)
    
    def __getitem__(self, idx):
        input_ids = self.input_ids_tensor[idx]
        attention_mask = self.attention_mask_tensor[idx]
        return input_ids, attention_mask
    
    def prepare_dataloader(self, batch_size):
        '''
        参数:
        text_save_path:词张量字典储存路径
        batch_size:略

        功能:根据输入读取词张量，生成并返回数据加载器
        '''

        dataset = TensorDataset(self.input_ids_tensor, self.attention_mask_tensor)

        predict_loader = DataLoader(dataset,batch_size,
                                       shuffle=True,num_workers=0,pin_memory=True)
        
        return predict_loader
    