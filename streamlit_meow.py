import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from ultralytics import YOLO
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseUpload
import io
import datetime

# URL to the weights file on GitHub
weights_url = "https://github.com/yd145763/EPTC2024GenerativeDNN/raw/main/best.pt"

# Load the YOLO model
@st.cache_resource
def load_model():
    return YOLO(weights_url)

model = load_model()

# Define the class names
class_names =  ['Good_Grating','Ring_Resonator','Fiber','Taper','Overetched_Grating','MMI','Bond_Pad','Electrode','OPA_Outlet','Detached_Grating']
class_colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'lime', 'pink']

# Function to authenticate and create a Google Drive service
def create_drive_service():
    SCOPES = ['https://www.googleapis.com/auth/drive']
    SERVICE_ACCOUNT_FILE = "https://github.com/yd145763/peekchip/raw/main/orbital-expanse-310515-963cd8243098.json"  # Update with your service account file

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)
    return service

# Function to save the file to Google Drive
def save_file_to_drive(service, folder_id, file_bytes, filename, mimetype):
    # Create a file metadata object
    file_metadata = {
        'name': filename,
        'parents': [folder_id]
    }
    media = MediaIoBaseUpload(file_bytes, mimetype=mimetype)

    # Upload the file
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

# Function to visualize predictions and return YOLO formatted labels
def visualize_predictions(image, results):
    # Convert the image to grayscale
    image = image.convert("L")
    
    # Create figure and axes without borders, margins, and axes
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap="gray")
    
    # Remove borders and axes
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    image_width, image_height = image.size
    
    # Store detected class names in a list
    detected_classes = []

    yolo_labels = []

    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0]
            class_id = int(box.cls[0])
            conf = box.conf[0]
            class_name = class_names[class_id]
            detected_classes.append(class_name)
            
            # YOLO format: class_id x_center y_center width height (normalized)
            x_center = ((x_min + x_max) / 2) / image_width
            y_center = ((y_min + y_max) / 2) / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height
            yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
            
            # Choose the color for the current class
            color = class_colors[class_id]
            
            # Create a Rectangle patch
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor=color, facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            # Optionally, you can add class_id and confidence score text
            # ax.text(x_min, y_min, f'{class_names[class_id]}: {conf:.2f}', color='white', fontsize=8, backgroundcolor=color)

    # Display the plot
    st.pyplot(fig)
    
    # Count the occurrences of each class name
    class_counts = Counter(detected_classes)
    
    # Print the class name and the number of times each class name is detected
    st.write("Detected Class Name Counts:")
    for class_name, count in class_counts.items():
        st.write(f"{class_name}: {count}")

    return fig, yolo_labels  # Return the figure and YOLO labels for saving

# Main function
def main():
    st.title("Image Prediction with YOLO")

    # File uploader component
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Make predictions using the YOLO model
        results = model(image)

        # Visualize the predictions and get YOLO labels
        fig, yolo_labels = visualize_predictions(image, results)

        # Save the plot to Google Drive
        service = create_drive_service()
        IMAGE_FOLDER_ID = '1s24y-sOFQh-AHcxNVihpOSXXG1bgze9F'  # Update with your images folder ID
        LABELS_FOLDER_ID = '1s2D7CXzyi48ym4gQH8Ylc9sgpJk0GTIR'  # Update with your labels folder ID
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"predicted_image_{current_time}.png"

        # Save the image plot to Google Drive
        plot_bytes = io.BytesIO()
        fig.savefig(plot_bytes, format='png')
        plot_bytes.seek(0)
        save_file_to_drive(service, IMAGE_FOLDER_ID, plot_bytes, filename, 'image/png')

        # Save the YOLO labels to Google Drive
        labels_filename = f"predicted_image_{current_time}.txt"
        labels_bytes = io.BytesIO('\n'.join(yolo_labels).encode())
        save_file_to_drive(service, LABELS_FOLDER_ID, labels_bytes, labels_filename, 'text/plain')

if __name__ == "__main__":
    main()
