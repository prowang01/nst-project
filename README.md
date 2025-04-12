# 🎨 Fast Neural Style Transfer - Project Report

This project is a practical implementation of **Neural Style Transfer (NST)** using a pre-trained model from TensorFlow Hub. The idea is simple: take a **photo** (content) and a **work of art** (style), and blend them together to create something unique.

---

## Objective

The goal of this project is to explore **Generative AI** by recreating a simplified version of Neural Style Transfer. We use a **Fast NST model** (feed-forward), apply it to custom images, and analyze the results in depth.

The final deliverable is a **single `.html` file**, including code, visual tests, and commentary.

---

## Inspiration

This project is based on the public GitHub repo by [Deepesh D M](https://github.com/deepeshdm/Neural-Style-Transfer), who built a complete NST demo with API and Streamlit.  
We simplified his approach to focus on a clean, readable and reproducible pipeline.

---

## What We Did

- Used the pre-trained model [`magenta/arbitrary-image-stylization-v1-256`](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2)
- Tested our own portraits (neutral background, front-facing)
- Used a variety of artistic styles: Hockney, Ghibli, Van Gogh, Ukiyo-e, Anime...
- Analyzed what works and what doesn't (texture, contrast, clarity)
- Optimized the pipeline for Colab & low RAM (resizing, batch processing)
- Documented all tests and results in a single `.html` report

---

## Project Structure

This project is organized in a modular way:

- `api/` contains the core logic for style transfer, encapsulated in `transfer_style.py`, and made importable with `__init__.py`.
- `assets/` stores all input images used for content and style (e.g. portraits, artworks).
- `model/` is where the pre-trained TensorFlow Hub model is downloaded and extracted.
- `run_local.py` allows you to test the NST pipeline locally with any chosen images.
- `requirements.txt` lists the Python dependencies for local installation.
- `README.md` contains the full project description, instructions, and usage guide.

This structure makes the project clean, testable, and easy to extend (e.g. for future web deployment).

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/prowang01/nst-project.git
cd nst-project
```

### 2. (Optional) Set up a virtual environment

It’s recommended to use a virtual environment for clean dependency management:

```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the pre-trained model

We use a TensorFlow Hub model for fast style transfer. Download and extract it into the model/ directory:

```bash
wget "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2?tf-hub-format=compressed" -O model.tar.gz
mkdir model
tar -xvzf model.tar.gz -C model
```

### 5. Prepare your images

Place your content and style images in the assets/ folder. You can use .jpg or .png files.

### 6. Run the project

Edit the image paths in run_local.py if needed, then run:
```bash
python run_local.py
```

This will generate a stylized image by applying the chosen artistic style to your content image and display it using matplotlib.



## Web App with Streamlit (Optional)

We’ve included a `Streamlit` interface for an interactive experience.  
You can upload your own images and apply style transfer directly in your browser!

### Setup

Make sure you’ve installed the required packages (including Streamlit):

```bash
pip install -r requirements.txt

```

### Run the web app locally
```bash
streamlit run app.py
```

Your default browser will open automatically at http://localhost:8501, showing the interface.

What you can do : 

- Upload a content image (.jpg)

- Upload a style image (.jpg)

- Click to generate your stylized result

- View and download the result

Images are resized automatically for performance and saved temporarily in the assets/ folder.
