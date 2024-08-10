import streamlit as st 
import pandas as pd
import numpy as np
import requests
from st_social_media_links import SocialMediaIcons
from LogoYolo.inference import predict
import cv2
from PIL import Image

def glowingSocial():
    
    #github
    
    iconlink = 'https://static-00.iconduck.com/assets.00/kaggle-icon-2048x2048-fxhlmjy3.png'
    proflink = 'https://www.kaggle.com/sushovansaha9'
    
    glowingLogo(href=proflink,iconlink=iconlink,bpos=80,rpos=180,size=50,R=100)
    
    # kaggle
    
    iconlink = 'https://qph.cf2.quoracdn.net/main-qimg-729a22aba98d1235fdce4883accaf81e'
    proflink = 'https://github.com/ambideXtrous9'
    
    glowingLogo(href=proflink,iconlink=iconlink,bpos=80,rpos=280,size=50,G=100)
    
    #linkedin
    
    iconlink = 'https://images.rawpixel.com/image_png_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdjk4Mi1kMy0xMC5wbmc.png'
    proflink = 'https://www.linkedin.com/in/sushovan-saha-29a00a113'
    
    glowingLogo(href=proflink,iconlink=iconlink,bpos=80,rpos=380,size=50,B=100)
    
    #leetcode
    
    iconlink = 'https://upload.wikimedia.org/wikipedia/commons/8/8e/LeetCode_Logo_1.png'
    proflink = 'https://leetcode.com/u/ambideXtrous9/'
    
    glowingLogo(href=proflink,iconlink=iconlink,bpos=80,rpos=80,size=50)
    
    iconlink = 'https://cdn4.iconfinder.com/data/icons/social-media-2210/24/Medium-512.png'
    proflink = 'https://medium.com/@sushovansaha95'
    
    glowingLogo(href=proflink,iconlink=iconlink,bpos=80,rpos=480,size=50)
    

def Social(sidebarPos = False,heading = None):
    
    if heading != None:
        st.title(f":rainbow[{heading}]")
        
    social_media_links = [
            "https://www.linkedin.com/in/sushovan-saha-29a00a113",
            "https://github.com/ambideXtrous9",
            "https://medium.com/@sushovansaha95"]

    social_media_icons = SocialMediaIcons(social_media_links) 

    social_media_icons.render(sidebar=sidebarPos, justify_content="center")

def HomePage():
    
    st.title(":blue[My Portfolio] :sunglasses:")
    
    
    gif_path = 'thor.gif'
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(gif_path)
    
    # Display "About Me" text in the right column
    with col2:
        st.subheader("About Me")
        st.write("""
        👋 Hi there! I'm **Sushovan Saha**, a Machine Learning (ML) enthusiast specializing in **Natural Language Processing (NLP)** and **Computer Vision (CV)**. 
        I did my M.Tech in Data Science from **IIT Guwahati**. I am also a **Kaggle Notebook Expert**.
    
        🌟 I'm passionate about exploring the possibilities of ML to solve real-world problems and improve people's lives. 
        I love working on challenging projects that require me to stretch my abilities and learn new things.
    
        📚 In my free time, I like to contribute in **Kaggle**, Write ML blogs in **Medium**, read ML related blogs and updates. 
        I'm always looking for ways to stay up-to-date with the latest developments in the field.
        """)
        
    glowingSocial()
    
    


def GitHubStats():
    st.title(":rainbow[GitHub Stats]")
    username = "ambideXtrous9"  # Replace with your GitHub username
    response = requests.get(f"https://api.github.com/users/{username}")

    if response.status_code == 200:
        user_data = response.json()
        st.write(f"**Username:** {user_data['login']}")
        st.write(f"**Name:** {user_data.get('name', 'N/A')}")
        st.write(f"**Public Repos:** {user_data['public_repos']}")
        st.write(f"**Followers:** {user_data['followers']}")
        st.write(f"**Following:** {user_data['following']}")
        st.write(f"**Profile URL:** {user_data['html_url']}")
    else:
        st.error("Failed to fetch GitHub stats. Please check the username or try again later.")
        
        
def glowingLogo(href,iconlink,bpos,rpos,size,R=64,G=224,B=208):
    
    st.markdown(f'''
        <style>
        @keyframes glow {{
            0% {{ box-shadow: 0 0 10px 5px rgba({R}, {G}, {B}, 0.8); }}
            50% {{ box-shadow: 0 0 20px 10px rgba({R}, {G}, {B}, 1); }}
            100% {{ box-shadow: 0 0 10px 5px rgba({R}, {G}, {B}, 0.8); }}
        }}

        .glowing-image {{
            width: {size}px;
            height: {size}px;
            border-radius: 50%;
            animation: glow 2s infinite;
        }}
        </style>

        <div style="position: fixed; bottom: {bpos}px; right: {rpos}px;">
            <a href='{href}' target='_blank'>
                <img src='{iconlink}' class='glowing-image'/>
            </a>
        </div>
    ''', unsafe_allow_html=True)

        
        
def YoloforLogo():
    
    st.write("""
            ### 🚀 **YOLOv8.1: The Latest in Object Detection**
            - 🆕 **YOLOv8.1 is out!**: The newest update in the YOLO series, maintaining its position as the state-of-the-art model for:
            - 🎯 **Object Detection**
            - 🌀 **Instance Segmentation**
            - 🏷️ **Classification**

            ### ⚠️ **Main Challenge: Custom Dataset Preparation**
            - 🔍 **Dataset Selection**: Using **Flickr27** as our image dataset.
            - 📸 **Flickr27 Overview**: Contains 27 different brand logos, perfect for training YOLO on custom data.

            - 💼 **Custom Dataset Prep**: The most crucial step in training YOLO models.

            - 🛠️ **Get Ready to Train**: With YOLOv8.1 and Flickr27, you'll be well-equipped to handle custom object detection tasks!
            """)
                
    iconlink = 'https://miro.medium.com/v2/resize:fit:4800/format:webp/1*x4eteo0X9VqrLEHAYyCX8Q.jpeg'
    proflink = 'https://medium.com/@sushovansaha95/yolov8-1-on-custom-dataset-logo-detection-8915286999ef'
    
    glowingLogo(href=proflink,iconlink=iconlink,bpos=80,rpos=80,size=80)


    uploaded_file = st.file_uploader("Upload an Image containing Logo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        # Convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        
        # Perform prediction
        prediction_image = predict(opencv_image)
        
        # Create two columns for side-by-side images
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", width = 200,use_column_width='auto')
            
            
        with col2:
            st.image(prediction_image, caption="Predicted Image", width=200,use_column_width='auto')
            
            
def NewsQA():
    
    st.write("""
            ### 🧠 **Step 1: Initial QA with Gemma 2b (No Fine-Tuning)**
            - 👉 **Load Gemma 2b instruct model**: Start with the pre-trained model.
            - 📝 **Create suitable prompts**: Develop prompts tailored for your QA task.
            - ✅ **Run QA**: Test the model's performance on Indian News QA without any fine-tuning.

            ### 🎯 **Step 2: Fine-Tune Gemma with Indian News Dataset**
            - 📚 **Gather Indian News Dataset**: Collect and prepare the relevant dataset.
            - 🔧 **Fine-tune Gemma**: Adjust the model using the dataset to enhance its performance for QA tasks.
            - 🚀 **Evaluate Results**: Compare the model's performance before and after fine-tuning.

            ### 🔍 **Step 3: Use RAG for Context Fetching**
            - 🛠️ **Integrate RAG**: Utilize RAG for context retrieval to improve QA accuracy.
            - **Steps Involved:**
            - ✂️ **Text Chunking**: Split the text data into smaller, token-based chunks.
            - 🧬 **Generate Embeddings**: Use `SentenceTransformer` to create embeddings for each chunk.
            - 🗂️ **Store in ChromaDB**: Save the embeddings in ChromaDB for efficient retrieval.
            - 🔎 **Context Fetching**: Retrieve the relevant context from ChromaDB for a given query.
            
            - 🎯 **Final QA**: Run the QA with the fine-tuned Gemma model and the retrieved context.
            """)
    
    iconlink = 'https://pbs.twimg.com/media/GG3sJg7X0AEaNIq.jpg'
    proflink = 'https://medium.com/@sushovansaha95/finetuning-gemma2b-instruct-for-qa-with-rag-6d879226157b'
    
    glowingLogo(href=proflink,iconlink=iconlink,bpos=80,rpos=80,size=80)
    

    iconlink = 'https://aidrivensearch.com/wp-content/uploads/2023/02/T5-framework-Logo-280px.png'
    proflink = 'https://github.com/ambideXtrous9/MTP-News-Article-based-Question-Answering-System'
    
    glowingLogo(href=proflink,iconlink=iconlink,bpos=80,rpos=200,size=80)
    
    

