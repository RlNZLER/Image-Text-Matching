import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import BertModel, BertTokenizer

class ImageTextDataset(Dataset):
    def __init__(self, filepath, tokenizer, image_dir='data/images', image_size=(224, 224)):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.image_dir = image_dir
        self.image_transform = Compose([
            Resize(image_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.data_index = self._create_index()

    def _create_index(self):
        data_index = []
        with open(self.filepath, 'r') as f:
            for line_number, line in enumerate(f):
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    data_index.append({
                        'line_number': line_number,
                        'image_path': parts[0],
                        'caption': parts[1],
                        'label': 1 if parts[2] == 'match' else 0
                    })
        return data_index

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        data_row = self.data_index[idx]
        image_path = os.path.join(self.image_dir, data_row['image_path'])
        caption = data_row['caption']
        label = data_row['label']
        
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)
        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        label = torch.tensor(label).long()
        
        return image, text, label, caption


class ImageTextMatchingModel(nn.Module):
    def __init__(self):
        super(ImageTextMatchingModel, self).__init__()
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.image_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, 768)
        self.classifier = nn.Linear(768 * 2, 2)

    def forward(self, images, input_ids, attention_mask):
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        image_features = self.image_model(images)
        combined_features = torch.cat((text_features, image_features), dim=1)
        logits = self.classifier(combined_features)
        return logits


def train_model(model, dataloader, optimizer, criterion, epochs=5, device='cpu'):
    model.train()
    for epoch in range(epochs):
        for images, texts, labels, _ in dataloader:
            images = images.to(device)
            input_ids = texts['input_ids'].squeeze(1).to(device)
            attention_mask = texts['attention_mask'].squeeze(1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images, input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


def evaluate_model(model, dataloader, device='cpu'):
    model.eval()
    similarity_scores = []
    for images, texts, _, captions in dataloader:
        images = images.to(device)
        input_ids = texts['input_ids'].squeeze(1).to(device)
        attention_mask = texts['attention_mask'].squeeze(1).to(device)

        with torch.no_grad():
            logits = model(images, input_ids, attention_mask)
            probs = F.softmax(logits, dim=1)
            predicted_labels = torch.argmax(probs, dim=1)

        # Caption similarity
        generated_captions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        similarity_scores.extend([calculate_similarity(generated_caption, caption) for generated_caption, caption in zip(generated_captions, captions)])
    
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    print(f"Average Caption Similarity: {avg_similarity:.4f}")
    return avg_similarity


def calculate_similarity(text1, text2):
    tokens1 = tokenizer.tokenize(text1)
    tokens2 = tokenizer.tokenize(text2)
    vec1 = torch.tensor(tokenizer.convert_tokens_to_ids(tokens1))
    vec2 = torch.tensor(tokenizer.convert_tokens_to_ids(tokens2))
    return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0), dim=1).item()


# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Assuming your data file and images directory are correctly set
train_file_path = '/home/rinzler/Github/Image-Text-Matching/data/flickr8k.TrainImages.txt'
images_directory = 'data/images'

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create the DataFrame from your data
data = []
with open(train_file_path, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            data.append({
                'image_path': os.path.join(images_directory, parts[0]),
                'caption': parts[1],
                'label': 1 if parts[2] == 'match' else 0
            })

df = pd.DataFrame(data)

# Dataset and DataLoader with improved efficiency
dataset = ImageTextDataset(train_file_path, tokenizer, image_dir=images_directory)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, prefetch_factor=2)

# Model and move it to the device (GPU/CPU)
model = ImageTextMatchingModel().to(device)

# Optimizer and Loss Function
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

train_model(model, dataloader, optimizer, criterion, epochs=5, device=device)

model_save_path = 'trained_models/image_text_matching_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Evaluate model on similarity of generated captions and actual captions
avg_similarity = evaluate_model(model, dataloader, device=device)
