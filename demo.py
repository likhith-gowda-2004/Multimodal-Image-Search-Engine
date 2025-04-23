"""
Demo script for the Multimodal Search Engine
This provides a simple command-line interface to test the search engine
"""

import os
import random
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import kagglehub
from multimodal_search_engine import (
    load_captions, 
    create_image_caption_mapping, 
    MultimodalSearchEngine
)

def parse_args():
    parser = argparse.ArgumentParser(description='Multimodal Search Engine Demo')
    parser.add_argument('--text', type=str, default=None, help='Text query')
    parser.add_argument('--image', type=str, default=None, help='Path to query image')
    parser.add_argument('--visual_weight', type=float, default=0.5, help='Weight for visual similarity')
    parser.add_argument('--text_weight', type=float, default=0.5, help='Weight for text similarity')
    parser.add_argument('--top_k', type=int, default=5, help='Number of results to return')
    parser.add_argument('--sample_limit', type=int, default=500, 
                        help='Limit number of images to process (for faster demo)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Download the dataset
    print("Downloading/accessing Flickr8k dataset...")
    path = kagglehub.dataset_download("aladdinpersson/flickr8kimagescaptions")
    
    # Define paths - Update these lines
    captions_file = os.path.join(path, "flickr8k", "captions.txt")
    images_folder = os.path.join(path, "flickr8k", "Images")
    
    print("Loading captions data...")
    captions_df = load_captions()
    
    # Create image-caption mapping
    image_caption_map = create_image_caption_mapping(captions_df)
    
    # Get list of all images
    all_image_files = [f for f in os.listdir(images_folder) 
                      if f.endswith('.jpg') and os.path.isfile(os.path.join(images_folder, f))]
    
    print(f"Found {len(all_image_files)} images in the dataset")
    
    # Create and preprocess search engine
    search_engine = MultimodalSearchEngine(all_image_files, image_caption_map, sample_limit=args.sample_limit)
    search_engine.preprocess()
    
    # Handle query image
    query_img_path = args.image
    if query_img_path is None:
        # If no image is provided, randomly select one from the dataset
        random_img = random.choice(all_image_files)
        query_img_path = os.path.join(images_folder, random_img)
        print(f"Randomly selected image: {random_img}")
    
    # Show query image
    query_img = Image.open(query_img_path)
    plt.figure(figsize=(5, 5))
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis('off')
    plt.show()
    
    # Handle query text
    query_text = args.text
    if query_text:
        print(f"Query text: '{query_text}'")
    else:
        print("No text query provided. Using only image similarity.")
    
    print("\nPerforming multimodal search...")
    results = search_engine.search(
        query_img_path=query_img_path,
        query_text=query_text,
        top_k=args.top_k,
        visual_weight=args.visual_weight,
        text_weight=args.text_weight
    )
    
    # Display results
    print(f"\nFound {len(results)} results:")
    search_engine.display_results(results)
    
    return search_engine

if __name__ == "__main__":
    main()