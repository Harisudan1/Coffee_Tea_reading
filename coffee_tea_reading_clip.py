import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from groq import Groq
import os
import fitz  # PyMuPDF for PDF extraction

# Initialize Groq client with your API key
client = Groq(api_key="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")

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# PDF extraction function
def extract_pdf_text(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text")  # Extract text from each page
    return text

# Extract reading documents
coffee_pdf = "E:/Alesa AI/Coffee reading/coffee reading.pdf"
tea_pdf = "E:/Alesa AI/Coffee reading/tea reading.pdf"

coffee_reading = extract_pdf_text(coffee_pdf)
tea_reading = extract_pdf_text(tea_pdf)

# Set up Streamlit UI styling
st.markdown(
    """
    <style>
    body {
        background-color: #f4f0db; /* Old parchment-like background */
        font-family: 'Georgia', serif; /* Old-style font */
        color: #3c2f2f; /* Dark brown text */
    }
    h1, h2, h3 {
        color: #3c2f2f; /* Dark brown headings */
    }
    .stButton>button {
        background-color: #d4cfc4; /* Aged paper button */
        border: 1px solid #3c2f2f;
        color: #3c2f2f;
        font-family: 'Georgia', serif;
    }
    .stButton>button:hover {
        background-color: #3c2f2f;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.title("🔮 Mystical Coffee & Tea Cup Reading")
st.write("Upload an image of your coffee or tea cup, and let the AI interpret its hidden patterns to predict your future!")

# Image uploader
uploaded_image = st.file_uploader("Upload an image of your cup", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Your Coffee/Tea Cup", use_column_width=True)

    # CLIP analysis to determine whether the cup is coffee or tea
    with st.spinner("Analyzing the image for patterns..."):
        # Preprocess image for CLIP
        inputs = clip_processor(images=image, return_tensors="pt", padding=True).to(device)
        
        # Use CLIP to extract image embeddings
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)

        # Use a simple approach: text prompts for coffee and tea
        coffee_prompt = "A cup of coffee with patterns, swirls, and readings related to love, career, and health."
        tea_prompt = "A tea cup with leaves, swirls, and mystical patterns related to love, career, and health."
        
        # Process text prompts for comparison
        coffee_inputs = clip_processor(text=[coffee_prompt], return_tensors="pt", padding=True).to(device)
        tea_inputs = clip_processor(text=[tea_prompt], return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            coffee_features = clip_model.get_text_features(**coffee_inputs)
            tea_features = clip_model.get_text_features(**tea_inputs)

        # Calculate cosine similarity for both coffee and tea
        coffee_similarity = torch.nn.functional.cosine_similarity(image_features, coffee_features).item()
        tea_similarity = torch.nn.functional.cosine_similarity(image_features, tea_features).item()

    # Determine whether it's a coffee or tea image based on similarity
    if coffee_similarity > tea_similarity:
        relevant_reading = coffee_reading
        st.write("This seems like a coffee reading!")
    else:
        relevant_reading = tea_reading
        st.write("This seems like a tea reading!")

    # Use Groq to generate a mystical reading based on the relevant document section
    with st.spinner("Generating your mystical reading..."):
        description_prompt = f"""
        Based on the following patterns described:
        {relevant_reading}

        Provide an engaging and mystical description of the patterns and their meanings in the context of love, career, and health. 
        Be sure to integrate the specific symbols and offer practical, actionable advice for the user.
        """
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # You can adjust this model as needed
            messages=[
                {"role": "system", "content": "You are a mystical fortune teller."},
                {"role": "user", "content": description_prompt},
            ],
            temperature=0.7,  # Set temperature to control randomness
            max_tokens=1024,  # Set the max tokens for completion
        )
        interpretation = response.choices[0].message.content

    # Display the mystical interpretation
    st.subheader("🔍 Pattern Interpretation")
    st.write(interpretation)

    # Generate a prediction with specific calls to action
    prediction_prompt = f"""
    Based on the mystical interpretation:
    {interpretation}

    Provide a fun and engaging prediction about the user's future in themes like love, career, and health. 
    Offer personalized, practical advice for each theme (love, career, health).
    """
    
    with st.spinner("Reading your future..."):
        prediction_response = client.chat.completions.create(
            model="llama3-8b-8192",  # Again, you can use any suitable model
            messages=[
                {"role": "system", "content": "You are a mystical fortune teller."},
                {"role": "user", "content": prediction_prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        prediction = prediction_response.choices[0].message.content

    # Display prediction
    st.subheader("🔮 Your Future Prediction")
    st.write(prediction)

st.write("**Disclaimer:** This app is for entertainment purposes only!")
