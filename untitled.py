import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import json



class MemeDataset(Dataset):
  def __init__(self, data, tokenizer, image_transform):
    self.data = data
    self.tokenizer = tokenizer
    self.image_transform = image_transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]
    
    # Tokenize text
    text_inputs = self.tokenizer(
        item.get('text', ''), 
        padding='max_length', 
        truncation=True, 
        max_length=128, 
        return_tensors='pt'
    )
    
    # Load and transform image
    image_path = item.get('img', '')
    image = load_image_safely(image_path)
    image_tensor = self.image_transform(image)
    
    # Get label (default to 0 if not present)
    label = torch.tensor(item.get('label', 0), dtype=torch.long)
    
    return {
        'text_input_ids': text_inputs['input_ids'].squeeze(),
        'text_attention_mask': text_inputs['attention_mask'].squeeze(),
        'image': image_tensor,
        'label': label
    }


def load_image_safely(img_path):
    """
    Safely load an image, handling potential errors
    """
    path = "../.cache/kagglehub/datasets/parthplc/facebook-hateful-meme-dataset/versions/1/data/"
    try:
        return Image.open(path+img_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {path+img_path}: {e}")
        # Return a blank image if loading fails
        return Image.new('RGB', (224, 224), color='black')

def create_dataset(jsonl_path, tokenizer, image_transform):
    """
    Create dataset outside the class to avoid pickling issues
    """
    data = []
    # Read JSONL file
    with open(jsonl_path, 'r') as f:
        for line in f:       
          data.append(json.loads(line))
            
    return MemeDataset(data, tokenizer, image_transform)

class MultimodalMemeClassifier(nn.Module):
    def __init__(self, text_feature_dim=768, image_feature_dim=2048, 
                 projection_dim=512, num_classes=2):
        super(MultimodalMemeClassifier, self).__init__()
        
        # Text Encoder (DistilBERT)
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Freeze text encoder weights
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Image Encoder (ResNet50)
        resnet = resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        # Freeze image encoder weights
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        # Text Projection Layer
        self.text_projection = nn.Sequential(
            nn.Linear(text_feature_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Image Projection Layer
        self.image_projection = nn.Sequential(
            nn.Linear(image_feature_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Feature Interaction Layer
        self.feature_interaction = nn.Sequential(
            nn.Linear(projection_dim * projection_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, text_input_ids, text_attention_mask, image):
        # Extract text features
        text_outputs = self.text_encoder(
            input_ids=text_input_ids, 
            attention_mask=text_attention_mask
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Extract image features
        image_features = self.image_encoder(image).flatten(start_dim=1)
        
        # Project features
        text_proj = self.text_projection(text_features)
        image_proj = self.image_projection(image_features)
        
        # Create feature interaction matrix
        interaction_matrix = torch.bmm(
            text_proj.unsqueeze(2), 
            image_proj.unsqueeze(1)
        ).flatten(start_dim=1)
        
        # Process interaction features
        interaction_features = self.feature_interaction(interaction_matrix)
        
        # Classify
        logits = self.classifier(interaction_features)
        
        return logits

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
  model.to(device)

  for epoch in range(epochs):
    i = 0
    model.train()
    total_train_loss = 0
    
    for batch in train_loader:
      print(i)
      i+=1    
      optimizer.zero_grad()
      
      # Move data to device
      text_input_ids = batch['text_input_ids'].to(device)
      text_attention_mask = batch['text_attention_mask'].to(device)
      images = batch['image'].to(device)
      labels = batch['label'].to(device)
      
      # Forward pass
      outputs = model(text_input_ids, text_attention_mask, images)
      loss = criterion(outputs, labels)
      
      # Backward pass and optimize
      loss.backward()
      optimizer.step()
      
      total_train_loss += loss.item()
      
    # Validation
    model.eval()
    total_val_loss = 0
    correct_predictions = 0
    total_predictions = 0
      
    with torch.no_grad():
      for batch in val_loader:
          text_input_ids = batch['text_input_ids'].to(device)
          text_attention_mask = batch['text_attention_mask'].to(device)
          images = batch['image'].to(device)
          labels = batch['label'].to(device)
          
          outputs = model(text_input_ids, text_attention_mask, images)
          loss = criterion(outputs, labels)
          
          total_val_loss += loss.item()
          
          # Calculate accuracy
          _, predicted = torch.max(outputs.data, 1)
          total_predictions += labels.size(0)
          correct_predictions += (predicted == labels).sum().item()
    
    print(f'Epoch {epoch+1}/{epochs}:')
    print(f'Train Loss: {total_train_loss/len(train_loader):.4f}')
    print(f'Val Loss: {total_val_loss/len(val_loader):.4f}')
    print(f'Val Accuracy: {100 * correct_predictions/total_predictions:.2f}%')

def main():
  # Setup
  device = torch.device('mps' if torch.mps.is_available() else 'cpu')
  
  print(device)
  # Tokenizer and image transformations
  tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
  image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
      mean=[0.485, 0.456, 0.406], 
      std=[0.229, 0.224, 0.225]
    )
  ])
  
  # Paths to your datasets (update these)
  train_path = '../.cache/kagglehub/datasets/parthplc/facebook-hateful-meme-dataset/versions/1/data/train.jsonl'
  dev_path = '../.cache/kagglehub/datasets/parthplc/facebook-hateful-meme-dataset/versions/1/data/dev.jsonl'
  
  # Create datasets
  train_dataset = create_dataset(
    train_path, 
    tokenizer=tokenizer, 
    image_transform=image_transform
  )
  val_dataset = create_dataset(
    dev_path, 
    tokenizer=tokenizer, 
    image_transform=image_transform
  )
  
  # Create data loaders
  train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True
  )
  val_loader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True
  )
  
  # Initialize model
  model = MultimodalMemeClassifier()
  
  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad], 
    lr=1e-4, 
    weight_decay=1e-5
  )
  
  # Train
  train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device
  )

if __name__ == '__main__':
  # Set multiprocessing method to 'spawn' for better compatibility
  import multiprocessing
  multiprocessing.set_start_method('spawn')
  
  main()