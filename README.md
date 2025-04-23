# Multimodal Search Engine

A sophisticated search engine for the Flickr8k dataset that combines both **visual** and **textual** features to enable powerful image retrieval capabilities.

---

## Features

- **Visual Search**: Retrieve visually similar images using ResNet18 deep features and color histograms.
- **Text Search**: Search images using natural language queries via caption embeddings.
- **Multimodal Search**: Combine both image and text queries with customizable weights for hybrid retrieval.
- **Interactive UI**: Streamlit-based user interface for interactive search experience.
- **Command-line Demo**: CLI for quick testing and experimentation.

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/multimodal-search-engine.git
cd multimodal-search-engine
```

Requirements
Python 3.8+

PyTorch

torchvision

pandas

numpy

scikit-learn

matplotlib

Pillow

streamlit

kagglehub

Dataset
This project uses the Flickr8k dataset, which contains 8,000 images with 5 captions each. The dataset is automatically downloaded from Kaggle when you run the application.

Note: You may need to authenticate with Kaggle when running the application for the first time. Follow the instructions provided by kagglehub.

Usage
Command-line Demo
Run a simple demo with the following command:

bash
Copy
Edit
python demo.py --text "children playing" --image "sample.jpg" --visual_weight 0.7 --text_weight 0.3 --top_k 5 --sample_limit 500
Options:

--text: Text query (optional)

--image: Path to query image (optional, randomly selects one if not provided)

--visual_weight: Weight for visual similarity (default: 0.5)

--text_weight: Weight for text similarity (default: 0.5)

--top_k: Number of results to return (default: 5)

--sample_limit: Limit number of images to process for faster demo (default: 500)

Streamlit Application
Launch the interactive web interface:

bash
Copy
Edit
streamlit run multimodal_search_app.py
The app allows you to:

Upload query images

Enter text queries

Adjust weights between visual and text features

Set the number of results to display

How It Works
The search engine combines multiple feature extraction methods:

Deep Visual Features: Extracted using a pre-trained ResNet18 model to capture high-level visual semantics.

Color Histograms: Used to analyze and compare color distributions across images.

Text Features: TF-IDF vectorization applied to the image captions to enable textual similarity matching.

The similarity scores from visual and text features are combined using a weighted average defined by user input.

Project Structure
multimodal_search_engine.py: Core engine implementation

multimodal_search_app.py: Streamlit web interface

demo.py: Command-line demo script

explore_dataset.py: Utility to examine dataset structure

requirements.txt: Project dependencies

Examples
Find images of "children playing at the beach"

Upload a dog image and find visually similar dog images

Combine an image of a mountain with the text "sunset" to find mountain sunset images

License
