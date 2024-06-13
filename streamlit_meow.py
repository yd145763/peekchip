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

# Define the service account file as a dictionary
SERVICE_ACCOUNT_DICT = {
    "type": "service_account",
    "project_id": "orbital-expanse-310515",
    "private_key_id": "963cd824309861910dc6a2109796f62c5e52709d",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDajBai8iKfQK6A\nuMsKjCdMwOvBMBZVzVDmKGeGLaTb0BLwhZovBjXciftms7dc+tgu3RN9mMG34Jnt\nApILfwzsv9vilZHbm6Cfn8yg6ydGIZz+kjromIdMj5d7rHTcaXSd5YaTEb1O2u+3\nMAZbeeXQNE16f4RTljLbmqPECNHRA+zWeYhPcv/hydkP+TwMvivit8pLhhmxYP7W\ni7bNmzz06PGeMcY9xrfoIwoeuz4tkbcMJGOIs/HMsuwwCIoXZQLaW6rCuhUA1IVl\nAzzbYTkHpB51rpZjbzw6UHJfPicB85hbBpGAjrHisaQuTwF1cQQGwgvp0Wm9oVBr\nNteo8EU/AgMBAAECggEATaP4BMJ1x1LQy70asPASpCNjfdnqDWhoBaQ0BwifKVI1\n8EneeTdBGkzQye8txLP+6kMzAesrYvpBZOCFZt0nh9IvOUN+smCLAzpflYmFBda6\nMfxcPja11l6q585gI8+5FMEueoASW3nPMKq4j3XyTXHPVqYHqjRRdA/vfxzNAW41\nz4Xp/mPE2gcDWVOL3MJ8iHCC7zlYCXA5gvvzSeOxqggfL3xsAQ9IcoYpq6wHRUcM\n9p8aPvItQFuZkpGSDRxGpbN7D42LwGnnZ9IhG5YLWelWvhCoxPPhZ3UCtj0jLbSk\njqGRauknfjCiV63YQYJccAWue3HlsJW0qxMONk7rkQKBgQD3pfnSykp1h8GQRRii\nKQdLy9V+JffHf/keBjf8jOIlBa2kujSopYKJu19MdnzIU0DTsOZbnJEX2ZqWADOM\nvaVAMBTIpk+b9Oa0QcDgeDxTJm5oMPExITRINZ05NX36lKcMvtP41C1FxMd3Qmbe\nmyZu4sFJU9fwmOM/7TaO8iLmMQKBgQDh6t+mKtFBpZ8hoPKWZo46SnOrNc8+9X6b\nLud90u8vR9y+QHVuxLZGHSqci+J8D9Oz+bUAoF4oha/opiQHVRgr5CZRSlp7caxM\nJWcm+CbiIXsyE9xmGqgAYQNEkAPQAzIN+VDwBsMGHrGrevzV7jxVSUaoXxemCEmc\nCuHqyZpWbwKBgQC/wMNs5Pe8g25sQCQvw6cDmIPploqB7eQ9zEDji067rebjKs8F\nWop2DSBgO0qpblU+1LW5b8Sk+/gHd4ZyzpE63z/okWYI8xsDrfojlNXY6GQKxwKq\nsPQjiWgA2Rt/wESMyOGmxNt9Lz3naQHwEaAzsd5J4eLNSASGBi39hx1gsQKBgQCr\n4StY2/iqzlw8lJwcC1ioRp6sIttTVimF3XCoDQSG31C1uordxHG89FHIGrEsnEoA\nArpQCLK6d8O26Dg/D+G8+u6gKEag5oIkyMwrHN9QahK1reCgGiW5bBW1YzBTYSsC\n8K1AgJR+InogMgkMph1m8cH9NaPjYSjHvIvXOpzepwKBgQDpU1zEJI/zrPs2uj6l\n9ni0fyLoT1TmU7L8kmsWghGxKlrSdD6nEPDWEOW9WE/izzS1fsKgx/ykCk7O7PyT\nXeYSZ6mCvRlrM3i3dhaqPf8pC2K6ZtyET/oZJO3htJpj8qWhts7bXMCH0o+9ooqM\nsOuVdRUHQebQhEk1RSTZPK8RrA==\n-----END PRIVATE KEY-----\n",
    "client_email": "peekchip@orbital-expanse-310515.iam.gserviceaccount.com",
    "client_id": "110283024412892628851",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://accounts.google.com/o/oauth2/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/peekchip%40orbital-expanse-310515.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
}



# Function to authenticate and create a Google Drive service
def create_drive_service():
    SCOPES = ['https://www.googleapis.com/auth/drive']

    credentials = service_account.Credentials.from_service_account_info(
        SERVICE_ACCOUNT_DICT, scopes=SCOPES)
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
