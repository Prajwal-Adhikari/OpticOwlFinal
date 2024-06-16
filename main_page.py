import numpy as np
import streamlit as st
from keras.models import load_model
from PIL import Image

# Set the page config
st.set_page_config(layout='wide')

# Set the title of the app
st.markdown("<h1 style='text-align: center;'>Eye Disease Detection App</h1>", unsafe_allow_html=True)

st.markdown('''
<div style="text-align: center">
Please follow the following instructions:
<ul>
<li>1. Upload the Fundus image.</li>
<li>2. Wait for the file to be uploaded.</li>
<li>3. Select the model to use.</li>
<li>4. Click the Predict button to predict the disease.</li>
</ul>
</div>
''', unsafe_allow_html=True)

# Add a description
st.markdown('<div style="text-align: center">Upload an image and select the model to use, then click on the Predict button to classify the image.</div>', unsafe_allow_html=True)

# Model selection
model_option = st.selectbox("Select Model", ["VGG19", "ResNet50", "Xception"])

image = st.file_uploader('Choose an image file', type=['jpeg', 'jpg', 'png'])

if image:
    im = Image.open(image)
    im_resized = im.resize((224, 224))
    test = np.expand_dims(im_resized, axis=0)

    # Model paths
    model_paths = {
        "VGG19": './vgg19.h5',
        "ResNet50": './resnet50.h5',
        "Xception": './xception.h5'
    }

    model = load_model(model_paths[model_option])

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', width=300)

    labels = ["Normal", "Cataract", "Diabetes", "Glaucoma", "Hypertension", "Myopia", "Age Issues", "Other"]

    # Create a button
    if st.button('Predict'):
        # Perform inference
        predictions = model.predict(test)
        
        # Get the predicted class and probabilities
        predicted_class = np.argmax(predictions)
        probabilities = predictions[0]
        percentage = probabilities[predicted_class]

        # Display the predicted class and probabilities for all classes
        st.markdown(f"<div style='text-align: left; color: blue;'><h2>Eye Disease : {labels[predicted_class]}</h2></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: left; color: green;'><h3>Probability: {percentage*100:.2f}%</h3></div>", unsafe_allow_html=True)

        