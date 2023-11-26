# streamlit_app.py
import streamlit as st
from models import PerceptronSentiment, BackpropagationSentiment, DNNSentiment, RNNSentiment, LSTMSentiment

# Import your tumor detection model class
from tumor_detection_model import TumorDetectionModel  

def main():
    st.title("Task Selection App")

    # User input for task selection
    selected_task = st.radio("Select a Task", ["Sentiment Classification", "Tumor Detection"])

    if selected_task == "Sentiment Classification":
        sentiment_classification_interface()
    elif selected_task == "Tumor Detection":
        tumor_detection_interface()

def sentiment_classification_interface():
    st.title("Sentiment Classification App")

    # User input for model selection
    model_selection = st.selectbox("Select a Sentiment Classification Model", ["Perceptron", "Backpropagation", "DNN", "RNN", "LSTM"])

    # Train model button
    train_model_button = st.button("Train Model")

    # Training progress bar
    progress_bar_training = st.progress(0.0)

    # Get the selected model class
    model_class = get_model_class(model_selection)

    # Create an instance of the selected model
    model_instance = model_class()

    # Train the model if the "Train Model" button is clicked
    if train_model_button:
        train_sentiment_model(model_instance, progress_bar_training)

    # User input for text to classify
    input_text = st.text_area("Enter the text for sentiment classification:")

    # Debugging information
    st.text(f"Selected Model: {model_selection}")
    st.text(f"Selected Dataset: IMDB Movie Review")
    st.text(f"Input Text: {input_text}")
    

    # Perform sentiment classification when Enter is pressed
    with st.form(key="sentiment_form"):
        st.form_submit_button("Classify Sentiment")

        # Call the classify_sentiment function after training
        classify_sentiment(model_instance, input_text)

def classify_sentiment(model_instance, input_text):
    # Placeholder function for sentiment classification
    if input_text:
        result = model_instance.predict(input_text)
        st.success(f"The sentiment is: {result}")
    else:
        st.warning("Please enter some text for sentiment classification.")

def train_sentiment_model(model_instance, progress_bar_training):
    # Placeholder function for training the sentiment model
    for percent_complete in range(100):
        progress_bar_training.progress(percent_complete + 1)
        st.empty()  # To clear the spinner
        # Add your actual training logic here (if applicable)

def tumor_detection_interface():
    st.title("Tumor Detection App")

    # User input for tumor detection (e.g., upload image)
    uploaded_file = st.file_uploader("Upload an image for tumor detection", type=["jpg", "jpeg", "png"])

    # Debugging information
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.text("Tumor detection results will appear here.")

        # Create an instance of the tumor detection model
        tumor_model = TumorDetectionModel()

        # Perform tumor detection when a file is uploaded
        if st.button("Detect Tumor"):
            result = tumor_model.detect_tumor(uploaded_file)
            st.success(f"Tumor detection result: {result}")
    else:
        st.warning("Please upload an image for tumor detection.")

def get_model_class(model_name):
    model_mapping = {
        "Perceptron": PerceptronSentiment,
        "Backpropagation": BackpropagationSentiment,
        "DNN": DNNSentiment,
        "RNN": RNNSentiment,
        "LSTM": LSTMSentiment,
    }
    return model_mapping.get(model_name, PerceptronSentiment)

if __name__ == "__main__":
    main()
