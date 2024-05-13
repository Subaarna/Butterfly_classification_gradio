import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf


model = tf.keras.models.load_model("model/transfer_model.keras")

# Define function to preprocess the image
def preprocess_image(image):
    # Resize the image to the model's input shape 
    image = image.resize((180, 180))
    # Convert the image to a NumPy array
    image = np.array(image)
    # Expand dimensions to match the model's input shape 
    image = np.expand_dims(image, axis=0)
    return image

class_names_sorted = [
    "ADONIS", "AFRICAN GIANT SWALLOWTAIL", "AMERICAN SNOOT", "AN 88", "APPOLLO", 
    "ATALA", "BANDED ORANGE HELICONIAN", "BANDED PEACOCK", "BECKERS WHITE", 
    "BLACK HAIRSTREAK", "BLUE MORPHO", "BLUE SPOTTED CROW", "BROWN SIPROETA", 
    "CABBAGE WHITE", "CAIRNS BIRDWING", "CHECQUERED SKIPPER", "CHESTNUT", 
    "CLEOPATRA", "CLODIUS PARNASSIAN", "CLOUDED SULPHUR", "COMMON BANDED AWL", 
    "COMMON WOOD-NYMPH", "COPPER TAIL", "CRECENT", "CRIMSON PATCH", 
    "DANAID EGGFLY", "EASTERN COMA", "EASTERN DAPPLE WHITE", "EASTERN PINE ELFIN", 
    "ELBOWED PIERROT", "GOLD BANDED", "GREAT EGGFLY", "GREAT JAY", 
    "GREEN CELLED CATTLEHEART", "GREEN HAIRSTREAK", "INDRA SWALLOW", 
    "Iphiclus sister", "JULIA", "LARGE MARBLE", "MALACHITE", 
    "MANGROVE SKIPPER", "MESTRA", "METALMARK", "MILBERTS TORTOISESHELL", 
    "MONARCH", "MOURNING CLOAK", "ORANGE OAKLEAF", "ORANGE TIP", 
    "ORCHARD SWALLOW", "PAINTED LADY", "PAPER KITE", "PEACOCK", 
    "PINE WHITE", "PIPEVINE SWALLOW", "POPINJAY", "PURPLE HAIRSTREAK", 
    "PURPLISH COPPER", "QUESTION MARK", "RED ADMIRAL", "RED CRACKER", 
    "RED POSTMAN", "RED SPOTTED PURPLE", "SCARCE SWALLOW", "SILVER SPOT SKIPPER", 
    "SLEEPY ORANGE", "SOOTYWING", "SOUTHERN DOGFACE", "STRAITED QUEEN", 
    "TROPICAL LEAFWING", "TWO BARRED FLASHER", "ULYSES", "VICEROY", 
    "WOOD SATYR", "YELLOW SWALLOW TAIL", "ZEBRA LONG WING"
]

# Define prediction function
def predict_class(image):
    # Preprocess the input image
    preprocessed_image = preprocess_image(image)
    # Perform prediction
    prediction = model.predict(preprocessed_image)
    # Get the predicted class label (index of maximum probability)
    predicted_class = np.argmax(prediction, axis=1)[0]
    # Get the corresponding class name
    predicted_class_name = class_names_sorted[predicted_class]
    return predicted_class_name

# Define Gradio interface with preuploaded images and labels
image_input = gr.Image(type="pil", label="Upload an image of a butterfly")
label_output = gr.Label(num_top_classes=1, label="Predicted Butterfly Class")

# Preuploaded images with labels
# Preuploaded images with labels
examples = [
    [Image.open("./1.jpg"), "Adonis"],
    [Image.open("./2.jpg"), "CHECQUERED SKIPPER"],
    [Image.open("./3.jpg"), "ORANGE TIP"],
    # Add more preuploaded images with labels as needed
]

interface = gr.Interface(fn=predict_class, inputs=image_input, outputs=label_output, examples=examples)

# Launch the Gradio interface
interface.launch(debug=True)
