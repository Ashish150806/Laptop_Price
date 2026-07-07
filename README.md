# 💻 Laptop Price Predictor

A machine learning web app that estimates the price of a laptop from its specifications — brand, type, processor, RAM, display, weight, and more.

Built with **Streamlit** for the interface and a **scikit-learn Random Forest** regression model under the hood.

<p align="left">
  <a href="https://laptopprice-ashish-divyansh-se-045-058.streamlit.app/"><img alt="Live Demo" src="https://img.shields.io/badge/Live%20Demo-Open%20App-FF4B4B?logo=streamlit&logoColor=white"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

## 🚀 Live Demo

**Try it here → [laptopprice-ashish-divyansh-se-045-058.streamlit.app](https://laptopprice-ashish-divyansh-se-045-058.streamlit.app/)**

---

## Overview

Enter a laptop's specifications and the app returns an estimated price (in ₹ INR). The model was trained on a dataset of ~1,300 real laptops, with feature engineering that turns raw spec strings into useful numeric features (e.g. converting screen resolution + size into **pixels-per-inch**, and grouping CPUs/GPUs/OS into brand categories).

## Features

- 🎯 Instant price estimate from 10 laptop specifications
- 🧮 Feature engineering: PPI from resolution & screen size, touchscreen/IPS flags, CPU/GPU/OS brand grouping
- 🌲 Random Forest regressor trained on log-transformed prices for stable predictions
- 🎨 Clean, custom-styled Streamlit UI
- ♻️ Reproducible training pipeline — regenerate the model with a single script

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python |
| Web app | Streamlit |
| ML / Data | scikit-learn, pandas, numpy |
| Visualization | matplotlib, seaborn |
| Model | `RandomForestRegressor` inside a scikit-learn `Pipeline` |

## Project Structure

```
Laptop_Price/
└── laptop_price/
    ├── app.py                    # Streamlit web app
    ├── data/
    │   └── laptop_data.csv        # Raw dataset (~1,300 laptops)
    ├── models/
    │   ├── pipe.pkl               # Trained model pipeline
    │   └── df.pkl                 # Processed dataframe (feeds UI dropdowns)
    ├── notebooks/
    │   └── prediction_model.ipynb # EDA + model development
    ├── src/
    │   ├── preprocess.py          # Reusable preprocessing utilities
    │   └── train_model.py         # Trains model and exports .pkl files
    ├── requirements.txt           # Python dependencies
    ├── Procfile                   # Heroku/Render deployment
    └── setup.sh                   # Streamlit deployment config
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Ashish150806/Laptop_Price.git
cd Laptop_Price/laptop_price
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## How It Works

The app loads a pre-trained pipeline (`models/pipe.pkl`) and a processed dataframe (`models/df.pkl`). When you submit the form:

1. Categorical inputs (brand, type, CPU, GPU, OS) and numeric inputs (RAM, weight) are collected.
2. Screen resolution and size are combined into a **PPI** (pixels-per-inch) value.
3. Touchscreen and IPS choices are encoded as binary flags.
4. The pipeline one-hot encodes categorical columns and feeds everything to the Random Forest.
5. The model predicts `log(price)`, which is exponentiated back into a rupee figure.

### Input features

`Company` · `TypeName` · `RAM` · `Weight` · `Touchscreen` · `IPS` · `PPI` · `CPU brand` · `GPU brand` · `OS`

## Retraining the Model

If you update `data/laptop_data.csv`, regenerate the pickle files (run from the `laptop_price/` directory):

```bash
python src/train_model.py
```

This retrains the pipeline, prints the R² score and MAE on a held-out test set, and overwrites `models/pipe.pkl` and `models/df.pkl`.

## Deployment

### Streamlit Community Cloud (recommended)
1. Push the repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect the repo and set `laptop_price/app.py` as the main file.

### Render
1. Push to GitHub and create a new **Web Service** on [render.com](https://render.com).
2. **Root directory:** `laptop_price`
3. **Build command:** `pip install -r requirements.txt`
4. **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## License

Released under the MIT License. Feel free to use, modify, and share.
