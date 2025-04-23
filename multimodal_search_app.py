import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image
import kagglehub
import tempfile
from multimodal_search_engine import (
    load_captions, 
    create_image_caption_mapping, 
    MultimodalSearchEngine
)

st.set_page_config(
    page_title="Multimodal Search Engine",
    layout="wide"
)

@st.cache_resource
def load_dataset():
    # Download the dataset (will use cached version if already downloaded)
    path = kagglehub.dataset_download("aladdinpersson/flickr8kimagescaptions")
    
    # Define paths - Update these lines
    captions_file = os.path.join(path, "flickr8k", "captions.txt")
    images_folder = os.path.join(path, "flickr8k", "Images")
    
    return path, captions_file, images_folder

@st.cache_resource
def initialize_search_engine(images_folder, image_caption_map, sample_limit=500):
    # Get list of all images
    all_image_files = [f for f in os.listdir(images_folder) 
                      if f.endswith('.jpg') and os.path.isfile(os.path.join(images_folder, f))]
    
    # Create and preprocess search engine
    search_engine = MultimodalSearchEngine(all_image_files, image_caption_map, sample_limit=sample_limit)
    search_engine.preprocess()
    
    return search_engine, all_image_files

def main():
    st.title("Multimodal Image Search Engine")
    st.write("Search images using both text and image inputs")
    
    # Load dataset
    with st.spinner("Loading dataset..."):
        path, captions_file, images_folder = load_dataset()
        captions_df = load_captions()
        image_caption_map = create_image_caption_mapping(captions_df)
    
    # Initialize search engine
    with st.spinner("Initializing search engine..."):
        search_engine, all_image_files = initialize_search_engine(images_folder, image_caption_map)
    
    # Sidebar controls
    st.sidebar.header("Search Parameters")
    
    # Weight sliders
    visual_weight = st.sidebar.slider("Visual Similarity Weight", 0.0, 1.0, 0.5, 0.1)
    text_weight = st.sidebar.slider("Text Similarity Weight", 0.0, 1.0, 0.5, 0.1)
    
    # Number of results
    top_k = st.sidebar.slider("Number of Results", 1, 10, 5)
    
    # Query inputs
    st.header("Query Input")
    
    # Text query
    query_text = st.text_input("Enter text query:", "")
    
    # Image query options
    img_query_method = st.radio(
        "Image Query Method:",
        ["Upload an image", "Select from dataset"],
        horizontal=True
    )
    
    # Image upload
    query_img_path = None
    uploaded_file = None
    
    if img_query_method == "Upload an image":
        uploaded_file = st.file_uploader("Upload a query image:", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                query_img_path = temp_file.name
            
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", width=300)
    else:
        # Random sample of images to choose from
        sample_images = np.random.choice(all_image_files, size=min(5, len(all_image_files)), replace=False)
        
        st.write("Select a query image from the dataset:")
        image_cols = st.columns(len(sample_images))
        
        selected_idx = None
        for i, img_name in enumerate(sample_images):
            img_path = os.path.join(images_folder, img_name)
            img = Image.open(img_path)
            image_cols[i].image(img, caption=f"Image {i+1}", width=150)
            if image_cols[i].button(f"Select {i+1}", key=f"select_{i}"):
                selected_idx = i
        
        if selected_idx is not None:
            query_img_path = os.path.join(images_folder, sample_images[selected_idx])
            st.write(f"Selected: Image {selected_idx+1}")
    
    # Search button
    if st.button("Search"):
        if query_text or query_img_path:
            with st.spinner("Searching..."):
                results = search_engine.search(
                    query_img_path=query_img_path,
                    query_text=query_text,
                    top_k=top_k,
                    visual_weight=visual_weight,
                    text_weight=text_weight
                )
                
                if len(results) > 0:
                    st.header("Search Results")
                    
                    for i, result in enumerate(results):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            img = Image.open(result['image_path'])
                            st.image(img, caption=f"Result {i+1}", width=300)
                        
                        with col2:
                            st.write(f"**Score:** {result['score']:.4f}")
                            st.write("**Captions:**")
                            for j, caption in enumerate(result['captions']):
                                st.write(f"- {caption}")
                        
                        st.divider()
                else:
                    st.warning("No results found.")
        else:
            st.warning("Please enter a text query or select/upload an image.")
    
    # Clean up temporary file
    if uploaded_file is not None and query_img_path is not None:
        if os.path.exists(query_img_path):
            os.unlink(query_img_path)

if __name__ == "__main__":
    main()