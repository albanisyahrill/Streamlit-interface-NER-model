# Named Entity Recognition (NER) Interface with Streamlit

This repository contains a Streamlit-based web application that provides an interface for performing inference using a Named Entity Recognition (NER) model. The application allows users to input text and receive annotated output highlighting the named entities in the text.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)

## Introduction

Named Entity Recognition (NER) is a sub-task of information extraction that seeks to locate and classify named entities mentioned in unstructured text into predefined categories such as person names, organizations, locations, etc.

This project provides a simple interface to interact with an NER model using Streamlit, a Python library for creating web applications. 

## Features

- **User-Friendly Interface**: Easily input text and view NER results.
- **Model Flexibility**: Replaceable model backend for different NER models.

## Installation

To run this application, you need to have Python 3.11.9 version installed on your machine. Follow the steps below to set up and run the application.

1. **Clone the repository**:
    ```sh
    git clone https://github.com/albanisyahrill/Streamlit-interface-NER-model.git
    ```

2. **Create a virtual environment** (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate
    ```
3. **Install the required packages**
   - Streamlit
   - Numpy
   - TensorFlow == 2.15.0
   
4. **Run the Streamlit application**:
    ```sh
    streamlit run app.py
    ```

## Usage

Once the application is running, you can access it in your web browser at `http://localhost:8501`. You will see a text input area where you can paste or type the text you want to analyze. After entering the text, the application will display only recognized entities with their labels.

## Example

1. **Input Text**: Enter your text in the input box.
2. **View Results**: Only recognized entities will be highlighted with their labels.
