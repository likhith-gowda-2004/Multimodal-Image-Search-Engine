import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import kagglehub
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torchvision import models, transforms
from torch.autograd import Variable

# Download the dataset
print("Downloading Flickr8k dataset...")
path = kagglehub.dataset_download("aladdinpersson/flickr8kimagescaptions")
print("Path to dataset files:", path)

# Define paths - Update this line
captions_file = os.path.join(path, "flickr8k", "captions.txt")
images_folder = os.path.join(path, "flickr8k", "Images")  # Images are likely in this subdirectory too

# Load captions data
def load_captions():
    # Different delimiter or format might be needed based on actual file
    try:
        # Try standard CSV format first
        captions_df = pd.read_csv(captions_file, delimiter=',')
    except:
        # Alternative format (tab-separated or different structure)
        # For Flickr8k.token.txt, format is often: image_id#number<tab>caption
        with open(captions_file, 'r') as f:
            lines = f.readlines()
        
        images = []
        captions = []
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                image_id = parts[0].split('#')[0]  # Remove the #number suffix
                caption = parts[1]
                images.append(image_id)
                captions.append(caption)
        
        captions_df = pd.DataFrame({'image': images, 'caption': captions})
    
    print(f"Loaded {len(captions_df)} caption entries")
    return captions_df

# Create image filename to captions mapping
def create_image_caption_mapping(captions_df):
    image_caption_map = {}
    for idx, row in captions_df.iterrows():
        image_filename = row['image']
        caption = row['caption']
        if image_filename not in image_caption_map:
            image_caption_map[image_filename] = []
        image_caption_map[image_filename].append(caption)
    return image_caption_map

# Feature extraction using a pre-trained CNN (ResNet)
class ImageFeatureExtractor:
    def __init__(self):
        # Load pre-trained ResNet model
        self.model = models.resnet18(pretrained=True)
        # Remove the final fully connected layer
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, img_path):
        try:
            # Load and transform the image
            img = Image.open(img_path).convert('RGB')
            img_t = self.transform(img)
            img_tensor = img_t.unsqueeze(0)
            
            # Extract features
            with torch.no_grad():
                features = self.model(img_tensor)
                
            # Reshape and convert to numpy array
            features = features.squeeze().numpy().flatten()
            return features
        except Exception as e:
            print(f"Error extracting features from {img_path}: {e}")
            return None

# Color histogram feature extraction
def extract_color_histogram(img_path, bins=32):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((256, 256))
        img_array = np.array(img)
        
        # Calculate histogram for each channel
        hist_r = np.histogram(img_array[:,:,0], bins=bins, range=(0, 256))[0]
        hist_g = np.histogram(img_array[:,:,1], bins=bins, range=(0, 256))[0]
        hist_b = np.histogram(img_array[:,:,2], bins=bins, range=(0, 256))[0]
        
        # Normalize histograms
        hist_r = hist_r / hist_r.sum()
        hist_g = hist_g / hist_g.sum()
        hist_b = hist_b / hist_b.sum()
        
        # Combine histograms
        hist = np.concatenate([hist_r, hist_g, hist_b])
        return hist
    except Exception as e:
        print(f"Error extracting color histogram from {img_path}: {e}")
        return None

# Text feature extraction
class TextFeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def fit(self, text_data):
        self.vectorizer.fit(text_data)
        
    def transform(self, text_data):
        return self.vectorizer.transform(text_data)

# Multimodal search engine
class MultimodalSearchEngine:
    def __init__(self, image_paths, image_caption_map, sample_limit=None):
        self.image_paths = list(image_paths)[:sample_limit] if sample_limit else list(image_paths)
        self.image_caption_map = image_caption_map
        self.image_feature_extractor = ImageFeatureExtractor()
        self.text_feature_extractor = TextFeatureExtractor()
        
        # Initialize empty features
        self.deep_features = {}
        self.color_features = {}
        self.caption_features = None
        
        print(f"Initializing search engine with {len(self.image_paths)} images")
    
    def preprocess(self):
        print("Extracting image features...")
        for i, img_name in enumerate(self.image_paths):
            if i % 100 == 0:
                print(f"Processing image {i}/{len(self.image_paths)}")
                
            img_path = os.path.join(images_folder, img_name)
            
            # Extract deep features
            deep_feat = self.image_feature_extractor.extract_features(img_path)
            if deep_feat is not None:
                self.deep_features[img_name] = deep_feat
                
            # Extract color histogram
            color_hist = extract_color_histogram(img_path)
            if color_hist is not None:
                self.color_features[img_name] = color_hist
        
        print("Extracting text features...")
        # Prepare captions for text feature extraction
        all_captions = []
        caption_map = {}
        
        for img_name in self.image_paths:
            if img_name in self.image_caption_map:
                # Combine all captions for this image
                combined_caption = " ".join(self.image_caption_map[img_name])
                all_captions.append(combined_caption)
                caption_map[img_name] = combined_caption
            else:
                all_captions.append("")
                caption_map[img_name] = ""
        
        # Extract text features
        self.text_feature_extractor.fit(all_captions)
        self.caption_features = self.text_feature_extractor.transform(all_captions)
        self.caption_map = caption_map
        
        print("Preprocessing complete.")
    
    def search(self, query_img_path=None, query_text=None, top_k=5, 
               visual_weight=0.5, text_weight=0.5):
        
        if query_img_path is None and query_text is None:
            print("Please provide either a query image or text.")
            return []
        
        scores = np.zeros(len(self.image_paths))
        
        # Image query
        if query_img_path is not None and visual_weight > 0:
            # Extract deep features
            query_deep_feat = self.image_feature_extractor.extract_features(query_img_path)
            # Extract color histogram
            query_color_hist = extract_color_histogram(query_img_path)
            
            # Calculate visual similarity
            for i, img_name in enumerate(self.image_paths):
                if img_name in self.deep_features and img_name in self.color_features:
                    # Deep feature similarity (70%)
                    deep_sim = cosine_similarity(
                        query_deep_feat.reshape(1, -1), 
                        self.deep_features[img_name].reshape(1, -1)
                    )[0][0]
                    
                    # Color histogram similarity (30%)
                    color_sim = cosine_similarity(
                        query_color_hist.reshape(1, -1), 
                        self.color_features[img_name].reshape(1, -1)
                    )[0][0]
                    
                    # Combined visual similarity
                    visual_sim = 0.7 * deep_sim + 0.3 * color_sim
                    scores[i] += visual_weight * visual_sim
        
        # Text query
        if query_text is not None and text_weight > 0:
            # Transform query text
            query_vec = self.text_feature_extractor.transform([query_text])
            
            # Calculate text similarity with all captions
            text_sim = cosine_similarity(query_vec, self.caption_features).flatten()
            scores += text_weight * text_sim
        
        # Get top results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            img_name = self.image_paths[idx]
            result = {
                'image_name': img_name,
                'image_path': os.path.join(images_folder, img_name),
                'score': scores[idx],
                'captions': self.image_caption_map.get(img_name, [])
            }
            results.append(result)
            
        return results
    
    def display_results(self, results):
        n_results = len(results)
        fig, axes = plt.subplots(n_results, 1, figsize=(10, 5*n_results))
        
        if n_results == 1:
            axes = [axes]
            
        for i, result in enumerate(results):
            img = Image.open(result['image_path'])
            axes[i].imshow(img)
            axes[i].set_title(f"Score: {result['score']:.4f}")
            
            caption_text = "\n".join(result['captions'][:2])
            if len(result['captions']) > 2:
                caption_text += f"\n... ({len(result['captions'])-2} more captions)"
                
            axes[i].set_xlabel(caption_text)
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()

# Main execution
def main():
    print("Loading captions data...")
    captions_df = load_captions()
    
    # Create image-caption mapping
    image_caption_map = create_image_caption_mapping(captions_df)
    
    # Get list of all images
    all_image_files = [f for f in os.listdir(images_folder) 
                      if f.endswith('.jpg') and os.path.isfile(os.path.join(images_folder, f))]
    
    print(f"Found {len(all_image_files)} images in the dataset")
    
    # Create and preprocess search engine (limit to 500 images to save resources)
    search_engine = MultimodalSearchEngine(all_image_files, image_caption_map, sample_limit=500)
    search_engine.preprocess()
    
    # Example: Perform a multimodal search
    query_img_path = os.path.join(images_folder, all_image_files[10])  # Use an image from the dataset
    query_text = "people playing on the beach"
    
    print("\nPerforming multimodal search...")
    results = search_engine.search(
        query_img_path=query_img_path,
        query_text=query_text,
        top_k=5,
        visual_weight=0.4,
        text_weight=0.6
    )
    
    # Display results
    search_engine.display_results(results)
    
    return search_engine

if __name__ == "__main__":
    search_engine = main()