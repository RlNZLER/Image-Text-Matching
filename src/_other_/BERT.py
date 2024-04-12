import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from transformers import BertTokenizer
from PIL import Image
import pandas as pd
import os

class ImageTextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, image_dir='data/images', image_size=(224, 224)):
        """
        dataframe: A pandas DataFrame containing 'image_path', 'caption', and 'label'
        tokenizer: An instance of BertTokenizer
        image_dir: Directory where images are stored (this parameter might be unused if image paths in dataframe are absolute)
        image_size: A tuple indicating the size to which images are resized
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.image_transform = Compose([
            Resize(image_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image_path']
        caption = row['caption']
        label = row['label']
        
        image = Image.open(image_path).convert('RGB')  # Make sure images are in RGB
        image = self.image_transform(image)
        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        label = torch.tensor(label).long()
        
        return image, text, label



from transformers import BertModel
from torchvision.models import resnet50
import torch.nn as nn

class ImageTextMatchingModel(nn.Module):
    def __init__(self):
        super(ImageTextMatchingModel, self).__init__()
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.image_model = resnet50(pretrained=True)
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, 768) # Matching BERT's embedding size
        self.classifier = nn.Linear(768 * 2, 2) # Binary classification

    def forward(self, images, input_ids, attention_mask):
        text_features = self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        image_features = self.image_model(images)
        combined_features = torch.cat((text_features, image_features), dim=1)
        logits = self.classifier(combined_features)
        return logits
    

def train_model(model, dataloader, optimizer, criterion, epochs=5, device='cpu'):
    model.train()
    for epoch in range(epochs):
        for images, texts, labels in dataloader:
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
        
        
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import torch

# Assuming your data file and images directory are correctly set
train_file_path = '/home/rinzler/Github/Image-Text-Matching/data/flickr8k.TrainImages.txt'
images_directory = 'data/images'



# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Read your dataset (adjust as needed)
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

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create the DataFrame from your data
df = pd.DataFrame(data)

# Dataset and DataLoader
dataset = ImageTextDataset(df, tokenizer, image_dir=images_directory)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# Model and move it to the device (GPU/CPU)
model = ImageTextMatchingModel().to(device)

# Optimizer and Loss Function
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

train_model(model, dataloader, optimizer, criterion, epochs=5, device=device)

model_save_path = 'trained_models/image_text_matching_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")