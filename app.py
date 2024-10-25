import joblib
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score
from torch.utils.data import DataLoader

from flask import Flask, request, jsonify
import numpy as np
from osgeo import gdal
import os

from Models.CNNSpectralAttention import CNN_With_Spectral_Attention
from FileHandler.HandleDataset import HyperspectralDataset

app = Flask(__name__)

# Check if a GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load the model
def load_model(path, model_class):
    model = model_class().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to use the model for predictions
def predict(model, data_loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Make predictions
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


# Load the saved Random Forest model
def load_random_forest_model(model_file_path):
    if not os.path.exists(model_file_path):
        raise FileNotFoundError(f"Model not found at {model_file_path}")
    model = joblib.load(model_file_path)
    return model

# Function to read the CSV file and extract features and labels
def process_csv(file_path):
    data = pd.read_csv(file_path)
    # Assuming the last column is the label and the rest are features
    features = data.iloc[:, :-1].values  # All columns except the last one
    labels = data.iloc[:, -1].values  # Last column as labels
    return features, labels


# Function to calculate NDVI for a single image
def calculate_ndvi(image_path, red_band_index=30, nir_band_index=80):
    dataset = gdal.Open(image_path)
    if dataset is None:
        return None

    # Read red and NIR bands
    red_band = dataset.GetRasterBand(red_band_index).ReadAsArray().astype(np.float32)
    nir_band = dataset.GetRasterBand(nir_band_index).ReadAsArray().astype(np.float32)

    # NDVI calculation
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    ndvi = np.nan_to_num(ndvi)  # Handle NaN values

    return ndvi

# Function to classify NDVI values
def classify_ndvi(ndvi):
    avg_ndvi = np.mean(ndvi)
    if avg_ndvi > 0.6:
        return "Healthy"
    elif avg_ndvi > 0.3:
        return "Rust"
    else:
        return "Other"

# Function to process each folder and return classifications
def process_folder(folder_path):
    classified_counts = {"Healthy": 0, "Rust": 0, "Other": 0}
    num_images = 0

    for file in os.listdir(folder_path):
        if file.endswith('.tif'):
            image_path = os.path.join(folder_path, file)
            ndvi = calculate_ndvi(image_path)
            if ndvi is not None:
                classification = classify_ndvi(ndvi)
                classified_counts[classification] += 1
                num_images += 1

    return num_images, classified_counts


# Function to analyze dataset for band count statistics
def analyze_dataset(root_dir):
    subfolder_stats = {}

    # Loop through each subfolder and calculate band count statistics
    for subdir, dirs, files in os.walk(root_dir):
        if subdir == root_dir:
            continue

        subfolder_name = os.path.basename(subdir)
        total_images = 0
        images_with_125_bands = 0
        images_with_126_bands = 0

        # Process each file in the current subfolder
        for file in files:
            if file.endswith('.tif'):
                file_path = os.path.join(subdir, file)
                try:
                    dataset = gdal.Open(file_path)
                    if dataset is not None:
                        num_bands = dataset.RasterCount  # Get the number of bands

                        total_images += 1
                        if num_bands == 125:
                            images_with_125_bands += 1
                        elif num_bands == 126:
                            images_with_126_bands += 1
                    else:
                        print(f"Unable to open file {file_path}.")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

        # Store the results for the current subfolder
        subfolder_stats[subfolder_name] = {
            "Total Images": total_images,
            "Images with 125 Bands": images_with_125_bands,
            "Images with 126 Bands": images_with_126_bands
        }

    return subfolder_stats

# Function to calculate hyperspectral image properties
def get_image_properties(file_path):
    try:
        dataset = gdal.Open(file_path)

        if dataset is None:
            return {"error": f"Unable to open file {file_path}"}

        # Get image dimensions
        width = dataset.RasterXSize
        height = dataset.RasterYSize

        # Get the number of bands
        num_bands = dataset.RasterCount

        # Get the datatype of the first band (assuming all bands have the same type)
        band = dataset.GetRasterBand(1)
        data_type = gdal.GetDataTypeName(band.DataType)

        # Get geotransform and projection information (if available)
        geotransform = dataset.GetGeoTransform()  # Affine transform coefficients
        projection = dataset.GetProjection()

        # Store the properties in a dictionary
        properties = {
            "file_path": file_path,
            "width": width,
            "height": height,
            "number_of_bands": num_bands,
            "data_type": data_type,
            "geotransform": geotransform,
            "projection": projection
        }

        return properties

    except Exception as e:
        return {"error": str(e)}

# NDVI calculation function
def calculate_ndvi(image_path):
    dataset = gdal.Open(image_path)
    red_band_index = 30  # Example index for Red band
    nir_band_index = 80  # Example index for NIR band

    # Read Red and NIR bands as arrays
    red_band = dataset.GetRasterBand(red_band_index).ReadAsArray().astype(np.float32)
    nir_band = dataset.GetRasterBand(nir_band_index).ReadAsArray().astype(np.float32)

    # Calculate NDVI
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    ndvi = np.nan_to_num(ndvi)  # Handle NaN and Inf

    # Step 1: Classify NDVI as vegetation or non-vegetation
    vegetation_mask = ndvi > 0.3

    # Step 2: Classify as healthy, rust-infected, or non-vegetation
    classification = np.zeros_like(ndvi, dtype=np.uint8)
    classification[ndvi > 0.6] = 1  # Healthy vegetation
    classification[(ndvi > 0.3) & (ndvi <= 0.6)] = 2  # Rust-infected vegetation
    classification[ndvi <= 0.3] = 3  # Non-vegetation

    return classification

# Inspect NDVI function
def inspect_ndvi(image_path):
    dataset = gdal.Open(image_path)
    red_band_index = 30  # Example index for Red band
    nir_band_index = 80  # Example index for NIR band

    # Read Red and NIR bands as arrays
    red_band = dataset.GetRasterBand(red_band_index).ReadAsArray().astype(np.float32)
    nir_band = dataset.GetRasterBand(nir_band_index).ReadAsArray().astype(np.float32)

    # Calculate NDVI
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    ndvi = np.nan_to_num(ndvi)  # Handle NaN and Inf

    return ndvi[:5, :5]  # Return a portion of the NDVI for inspection

# API endpoint for NDVI classification
@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Perform NDVI-based classification
    classification = calculate_ndvi(file_path)

    result = classification[:5, :5].tolist()
    return jsonify({'classification': result})

# API endpoint for NDVI inspection
@app.route('/inspectNDVI', methods=['POST'])
def inspect_ndvi_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Perform NDVI inspection
    ndvi_result = inspect_ndvi(file_path)
    return jsonify({'ndvi': ndvi_result.tolist()})

# New API endpoint for Hyperspectral Properties
@app.route('/hyperspectral-properties', methods=['POST'])
def hyperspectral_properties():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Get properties of the uploaded image
    properties = get_image_properties(file_path)

    return jsonify(properties)

# New API endpoint for dataset analytics
@app.route('/dataset-analytics', methods=['POST'])
def dataset_analytics():
    data = request.get_json()
    if 'root_dir' not in data:
        return jsonify({'error': 'root_dir not provided'}), 400

    root_dir = data['root_dir']

    if not os.path.exists(root_dir):
        return jsonify({'error': f'Directory {root_dir} does not exist'}), 400

    # Perform dataset analysis
    stats = analyze_dataset(root_dir)

    return jsonify(stats)


@app.route('/ndvi-analytics', methods=['POST'])
def ndvi_analytics():
    data = request.get_json()
    if 'root_dir' not in data:
        return jsonify({'error': 'root_dir not provided'}), 400

    root_dir = data['root_dir']

    # Define the subfolders to be analyzed
    subfolders = ['0_Health', '1_Rust', '2_Other']

    results = {}

    # Loop through each subfolder and analyze its contents
    for subfolder in subfolders:
        folder_path = os.path.join(root_dir, subfolder)
        if not os.path.exists(folder_path):
            results[subfolder] = {
                'error': f'Folder {subfolder} does not exist'
            }
            continue

        # Process the folder and get the classification results
        num_images, counts = process_folder(folder_path)

        # Store the result for this subfolder
        results[subfolder] = {
            'Number of Images': num_images,
            'Classified as Healthy': counts['Healthy'],
            'Classified as Rust': counts['Rust'],
            'Classified as Other': counts['Other']
        }

    return jsonify(results)

# New route to perform inference and return report
@app.route('/cnn-spectral-attention', methods=['POST'])
def cnn_spectral_attention():
    try:
        # Get model and folder paths from the request
        model_load_path = request.json.get('model_load_path')
        test_folder = request.json.get('test_folder')

        if not os.path.exists(model_load_path):
            return jsonify({'error': 'Model path not found'}), 400

        if not os.path.exists(test_folder):
            return jsonify({'error': 'Test folder not found'}), 400

        # Load the trained model
        model = load_model(model_load_path, CNN_With_Spectral_Attention)

        # Prepare the test dataset and DataLoader
        test_dataset = HyperspectralDataset(test_folder)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Use the model to make predictions
        predictions, true_labels = predict(model, test_loader)

        # Calculate accuracy
        accuracy = accuracy_score(true_labels, predictions)

        # Generate classification report and confusion matrix
        classification_rep = classification_report(true_labels, predictions, output_dict=True)
        confusion_mat = confusion_matrix(true_labels, predictions).tolist()

        # Return the results
        return jsonify({
            'accuracy': accuracy,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_mat
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Random Forest inference function with CSV upload support
@app.route('/random-forest-model', methods=['POST'])
def random_forest_model():
    try:
        # Check if a file is part of the request
        if 'csv_file' not in request.files:
            return jsonify({'error': 'No CSV file provided'}), 400

        file = request.files['csv_file']

        # Save the uploaded CSV file
        csv_file_path = os.path.join('uploads', file.filename)
        file.save(csv_file_path)

        # Process the CSV file to get the validation data
        validation_data, validation_labels = process_csv(csv_file_path)

        # Get the model path from the request JSON
        model_file_path = request.form.get('model_file_path')
        if not model_file_path:
            return jsonify({'error': 'Model file path is required'}), 400

        # Load the Random Forest model
        model = load_random_forest_model(model_file_path)

        # Make predictions on the validation set
        predicted_labels = model.predict(validation_data)

        # Calculate evaluation metrics
        accuracy = accuracy_score(validation_labels, predicted_labels)
        precision = precision_score(validation_labels, predicted_labels, average='weighted')
        recall = recall_score(validation_labels, predicted_labels, average='weighted')
        f1 = f1_score(validation_labels, predicted_labels, average='weighted')

        # Generate confusion matrix and classification report
        conf_matrix = confusion_matrix(validation_labels, predicted_labels).tolist()
        class_report = classification_report(validation_labels, predicted_labels, target_names=['Health', 'Rust', 'Other'], output_dict=True)

        # Return the results as JSON
        return jsonify({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
