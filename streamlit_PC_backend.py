# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:00:01 2024

@author: limyu
"""

import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from ultralytics import YOLO
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
import io
import datetime
import tempfile  # Import tempfile to create temporary files

# Define the class names and colors
class_names = ['Good_Grating','Ring_Resonator','Fiber','Taper','Overetched_Grating','MMI','Bond_Pad','OPA_Outlet','Detached_Grating', 'PD_with_Pads', 'Bonded_Pad']
class_colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'lime', 'pink', 'white']

# Define the service account file as a dictionary
SERVICE_ACCOUNT_DICT = {
    "type": st.secrets["type"],
    "project_id": st.secrets["project_id"],
    "private_key_id": st.secrets["private_key_id"],
    "private_key": st.secrets["private_key"],
    "client_email": st.secrets["client_email"],
    "client_id": st.secrets["client_id"],
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://accounts.google.com/o/oauth2/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": st.secrets["client_x509_cert_url"],
    "universe_domain": "googleapis.com"
}

# Google Drive folder IDs
IMAGE_FOLDER_ID = '1lyak-BK1_m8k6OMbiAREPqK5Xjb7NoJa'  # Update with your images folder ID
LABELS_FOLDER_ID = '1s2D7CXzyi48ym4gQH8Ylc9sgpJk0GTIR'  # Update with your labels folder ID

# Function to authenticate and create a Google Drive service
def create_drive_service():
    SCOPES = ['https://www.googleapis.com/auth/drive']
    credentials = service_account.Credentials.from_service_account_info(SERVICE_ACCOUNT_DICT, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)
    return service

# Function to download a file from Google Drive and return it as a BytesIO object
def download_file_from_drive(service, file_id):
    request = service.files().get_media(fileId=file_id)
    downloaded_file = io.BytesIO()
    downloader = MediaIoBaseDownload(downloaded_file, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")
    downloaded_file.seek(0)
    return downloaded_file

# Load the YOLO model
@st.cache_resource
def load_model():
    service = create_drive_service()
    weights_file_id = '1--UOa52iTPXqvuhMLv7W3hAD9iZ11NC2'  # Replace with your weights file ID
    weights_file = download_file_from_drive(service, weights_file_id)
    
    # Write the BytesIO contents to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
        tmp_file.write(weights_file.read())
        tmp_file_path = tmp_file.name
    
    return YOLO(tmp_file_path)

model = load_model()

# Function to save the file to Google Drive
def save_file_to_drive(service, folder_id, file_bytes, filename, mimetype):
    file_metadata = {
        'name': filename,
        'parents': [folder_id]
    }
    media = MediaIoBaseUpload(file_bytes, mimetype=mimetype)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')

# Function to visualize predictions and return YOLO formatted labels
def visualize_predictions(image, results):
    image = image.convert("L")
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap="gray")
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    image_width, image_height = image.size
    detected_classes = []
    yolo_labels = []

    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0]
            class_id = int(box.cls[0])
            conf = box.conf[0]
            class_name = class_names[class_id]
            detected_classes.append(class_name)
            x_center = ((x_min + x_max) / 2) / image_width
            y_center = ((y_min + y_max) / 2) / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height
            yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
            color = class_colors[class_id]
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

    st.pyplot(fig)
    class_counts = Counter(detected_classes)
    st.write("Detected Device:")
    for class_name, count in class_counts.items():
        st.write(f"{class_name}: {count}")

    return fig, yolo_labels

# Main function
def main():
    st.sidebar.title("Contact Information")
    st.sidebar.write("**Admin:** Lim Yu Dian")
    st.sidebar.write("**Email:** limyudian@gmail.com")
    st.title("PeekChip")
    uploaded_file = st.file_uploader("Choose an image (optical microscope images of SiPh chips preferred)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        results = model(image)
        fig, yolo_labels = visualize_predictions(image, results)
        service = create_drive_service()
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"predicted_image_{current_time}.png"
        
        # Save the figure to a BytesIO object
        plot_bytes = io.BytesIO()
        fig.savefig(plot_bytes, format='png')
        plot_bytes.seek(0)
        
        # Ensure the BytesIO object is not empty
        if plot_bytes.getbuffer().nbytes > 0:
            print(f"Saving image to Google Drive: {filename}")
            image_file_id = save_file_to_drive(service, IMAGE_FOLDER_ID, plot_bytes, filename, 'image/png')
            print(f"Image saved with ID: {image_file_id}")
        else:
            print("Failed to save image: plot_bytes is empty.")
        
        labels_filename = f"predicted_image_{current_time}.txt"
        labels_bytes = io.BytesIO('\n'.join(yolo_labels).encode())
        labels_file_id = save_file_to_drive(service, LABELS_FOLDER_ID, labels_bytes, labels_filename, 'text/plain')
        print(f"Labels saved with ID: {labels_file_id}")

if __name__ == "__main__":
    main()
