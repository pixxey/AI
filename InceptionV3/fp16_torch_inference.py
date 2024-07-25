import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load the model using torch.hub
def load_pretrained_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model = model.to(device).half()  # Convert model to fp16
    model.eval()
    return model

# Function to preprocess the image
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file '{image_path}'")
        return None
    image = preprocess(image).unsqueeze(0).half()  # Add batch dimension and convert to fp16
    return image.to(device)

# Function to load ImageNet class names
def load_class_names(filepath):
    with open(filepath) as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

# Function to make predictions
def predict_image(model, image_path, class_names):
    image = preprocess_image(image_path)
    if image is None:
        return None
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds[0]]
    return predicted_class

# Main function
if __name__ == "__main__":
    model = load_pretrained_model()
    class_names = load_class_names('imagenet_classes.txt')  # Path to the ImageNet classes file

    image_path = 'cat.jpg'  # Path to the image you want to predict
    predicted_class = predict_image(model, image_path, class_names)
    if predicted_class is not None:
        print(f'The predicted class is: {predicted_class}')

