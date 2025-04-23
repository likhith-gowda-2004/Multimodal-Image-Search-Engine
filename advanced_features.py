"""
Advanced features for the Multimodal Search Engine:
1. Relevance feedback
2. Semantic matching using sentence embeddings
3. Performance optimizations
"""

import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from multimodal_search_engine import MultimodalSearchEngine

class AdvancedMultimodalSearchEngine(MultimodalSearchEngine):
    def __init__(self, image_paths, image_caption_map, sample_limit=None):
        super().__init__(image_paths, image_caption_map, sample_limit)
        
        # Initialize semantic text encoder (Sentence BERT)
        self.semantic_encoder_init = False
        self.semantic_features = None
    
    def initialize_semantic_encoder(self):
        """Initialize semantic text encoder"""
        print("Initializing semantic text encoder...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.semantic_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.semantic_encoder_init = True
            print("Semantic encoder initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize semantic encoder: {e}")
            self.semantic_encoder_init = False
    
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling for sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode_text_semantically(self, texts):
        """Encode text using Sentence BERT"""
        if not self.semantic_encoder_init:
            self.initialize_semantic_encoder()
            if not self.semantic_encoder_init:
                return None
        
        # Tokenize input texts
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.semantic_model(**encoded_input)
        
        # Perform mean pooling
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        return embeddings.numpy()
    
    def extract_semantic_features(self):
        """Extract semantic features for all captions"""
        if not hasattr(self, 'caption_map'):
            print("Caption map not found. Run preprocess() first.")
            return
        
        print("Extracting semantic features from captions...")
        captions = [self.caption_map[img_name] for img_name in self.image_paths]
        
        # Extract in batches to avoid memory issues
        batch_size = 32
        self.semantic_features = []
        
        for i in range(0, len(captions), batch_size):
            batch = captions[i:i+batch_size]
            batch_features = self.encode_text_semantically(batch)
            if batch_features is not None:
                self.semantic_features.append(batch_features)
        
        if len(self.semantic_features) > 0:
            self.semantic_features = np.vstack(self.semantic_features)
            print(f"Extracted semantic features: {self.semantic_features.shape}")
        else:
            self.semantic_features = None
            print("Failed to extract semantic features.")
    
    def preprocess(self):
        """Override preprocess to include semantic features"""
        super().preprocess()
        self.extract_semantic_features()
    
    def search_with_semantic(self, query_img_path=None, query_text=None, top_k=5,
                  visual_weight=0.4, text_weight=0.3, semantic_weight=0.3):
        """Search with semantic text matching"""
        if query_img_path is None and query_text is None:
            print("Please provide either a query image or text.")
            return []
        
        # Get regular scores
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
        
        # Text query - TF-IDF
        if query_text is not None and text_weight > 0:
            # Transform query text
            query_vec = self.text_feature_extractor.transform([query_text])
            
            # Calculate text similarity with all captions
            text_sim = cosine_similarity(query_vec, self.caption_features).flatten()
            scores += text_weight * text_sim
        
        # Semantic text query
        if query_text is not None and semantic_weight > 0 and self.semantic_features is not None:
            # Encode query text
            query_semantic = self.encode_text_semantically([query_text])
            
            if query_semantic is not None:
                # Calculate semantic similarity
                semantic_sim = cosine_similarity(query_semantic, self.semantic_features).flatten()
                scores += semantic_weight * semantic_sim
        
        # Get top results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            img_name = self.image_paths[idx]
            result = {
                'image_name': img_name,
                'image_path': os.path.join(self.images_folder, img_name),
                'score': scores[idx],
                'captions': self.image_caption_map.get(img_name, [])
            }
            results.append(result)
            
        return results
    
    def relevance_feedback(self, query_img_path, query_text, results, 
                          relevant_indices, irrelevant_indices, alpha=0.1, beta=0.05):
        """Implement Rocchio algorithm for relevance feedback"""
        if len(relevant_indices) == 0 and len(irrelevant_indices) == 0:
            return self.search(query_img_path, query_text)
            
        # Get original query vectors
        if query_img_path is not None:
            query_deep_feat = self.image_feature_extractor.extract_features(query_img_path)
            query_color_hist = extract_color_histogram(query_img_path)
        else:
            query_deep_feat = None
            query_color_hist = None
            
        if query_text is not None:
            query_text_vec = self.text_feature_extractor.transform([query_text]).toarray()[0]
        else:
            query_text_vec = None
            
        # Extract features from relevant and irrelevant results
        relevant_deep_feats = []
        relevant_color_hists = []
        irrelevant_deep_feats = []
        irrelevant_color_hists = []
        
        for idx in relevant_indices:
            if idx < len(results):
                img_name = results[idx]['image_name']
                if img_name in self.deep_features:
                    relevant_deep_feats.append(self.deep_features[img_name])
                if img_name in self.color_features:
                    relevant_color_hists.append(self.color_features[img_name])
                    
        for idx in irrelevant_indices:
            if idx < len(results):
                img_name = results[idx]['image_name']
                if img_name in self.deep_features:
                    irrelevant_deep_feats.append(self.deep_features[img_name])
                if img_name in self.color_features:
                    irrelevant_color_hists.append(self.color_features[img_name])
        
        # Apply Rocchio algorithm to modify query vectors
        if query_deep_feat is not None and len(relevant_deep_feats) > 0:
            relevant_centroid = np.mean(relevant_deep_feats, axis=0)
            query_deep_feat = query_deep_feat + alpha * relevant_centroid
            
            if len(irrelevant_deep_feats) > 0:
                irrelevant_centroid = np.mean(irrelevant_deep_feats, axis=0)
                query_deep_feat = query_deep_feat - beta * irrelevant_centroid
                
        if query_color_hist is not None and len(relevant_color_hists) > 0:
            relevant_centroid = np.mean(relevant_color_hists, axis=0)
            query_color_hist = query_color_hist + alpha * relevant_centroid
            
            if len(irrelevant_color_hists) > 0:
                irrelevant_centroid = np.mean(irrelevant_color_hists, axis=0)
                query_color_hist = query_color_hist - beta * irrelevant_centroid
        
        # Now compute similarity with modified query vectors
        scores = np.zeros(len(self.image_paths))
        
        # Visual similarity with modified query
        if query_deep_feat is not None and query_color_hist is not None:
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
                    scores[i] += 0.5 * visual_sim
        
        # Text similarity
        if query_text is not None:
            query_vec = self.text_feature_extractor.transform([query_text])
            text_sim = cosine_similarity(query_vec, self.caption_features).flatten()
            scores += 0.5 * text_sim
        
        # Get top results
        top_indices = np.argsort(scores)[::-1][:10]
        
        results = []
        for idx in top_indices:
            img_name = self.image_paths[idx]
            result = {
                'image_name': img_name,
                'image_path': os.path.join(self.images_folder, img_name),
                'score': scores[idx],
                'captions': self.image_caption_map.get(img_name, [])
            }
            results.append(result)
            
        return results

# Import necessary functions from original module
from multimodal_search_engine import extract_color_histogram