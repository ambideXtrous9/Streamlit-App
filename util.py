import streamlit as st 
import requests
from st_social_media_links import SocialMediaIcons
from LogoYolo.inference import predict
from PIL import Image
from HarryAgent.chatbot import ChatBot
from icons import glowingSocial,glowingYolo
from login import login_page


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
    # Display "About Me" text in the right column
    st.markdown("""
        <div align="center">
            <a href="https://in.linkedin.com/in/sushovan-saha-29a00a113" target="blank"><img align="center" src="https://www.baretreemedia.com/wp-content/uploads/2018/05/01_YESS_Thor_v4.gif" width="400" height="400" /></a>
        </div>
        
        # <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/People/Man%20Technologist.png" alt="Man Technologist" width="40" height="40" /> About Me  

        <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Hand%20gestures/Waving%20Hand%20Medium-Light%20Skin%20Tone.png" alt="Waving Hand Medium-Light Skin Tone" width="30" height="30" /> Hi there! 
        I'm **Sushovan Saha** — a passionate **Machine Learning (ML)** practitioner with deep interests in **Machine Learning (ML)**, **Deep Learning**, **Natural Language Processing (NLP)**, **Computer Vision (CV)**, and the transformative capabilities of **Large Language Models (LLMs)** and **Gen AI**.  
        Currently working as a founding **AI Engineer** at a Stealth Startup.
            
        <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Telegram-Animated-Emojis/main/Objects/Graduation%20Cap.webp" alt="Graduation Cap" width="30" height="30" /> I hold an **M.Tech in Data Science from IIT Guwahati** and I’m currently a **Kaggle Notebook Expert** with a strong inclination toward solving real-world challenges using intelligent systems.  

        ## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Rocket.png" alt="Rocket" width="30" height="30" /> My Skills:  
        - **LLMs & Generative AI**: Proficient in **RAG, Agentic RAG, LangChain, LangGraph, Crew AI**, and building **Agentic Workflows**  
        - **Applied ML & Deep Learning**: End-to-end project experience across NLP and CV domains  
        - **Model Fine-Tuning**: Experience in fine-tuning models on custom datasets in **HuggingFace, Unsloth, Pytorch and Pytorch Lightning**
        - **MLOps & Scalable ML Systems**: Focus on production-grade pipelines, model deployment, and monitoring **MLFlow**
        - **API & Deployment**: Experience in deploying models as APIs using **FastAPI** and deploying on cloud platforms like **AWS and GCP** 
        - **CI/CD**: Experience in setting up CI/CD pipelines using **GitHub Actions**
        - **Data Science**: Experience in data analysis, statistical analysis, data visualization, and data preprocessing using **Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn**

        ---

        [![My Skills](https://skillicons.dev/icons?i=cpp,python,pytorch,vscode,git,github,docker,gcp,aws,githubactions&perline=10)](https://skillicons.dev)  

        <div style="display: flex; flex-wrap: wrap; gap: 10px;" align="left">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/anaconda/anaconda-original.svg" width="50" height="50"/>
            <img src="https://colab.research.google.com/img/colab_favicon_256px.png" width="50" height="50"/>
            <img src="https://cdn3.iconfinder.com/data/icons/logos-and-brands-adobe/512/189_Kaggle-512.png" width="50" height="50"/>
            <img src="https://cdn-avatars.huggingface.co/v1/production/uploads/62ecdc18b72a69615d6bd857/E4lkPz1TZNLzIFr_dR273.png" width="50" height="50"/>
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Keras_logo.svg/1200px-Keras_logo.svg.png" width="50" height="50"/>
            <img src="https://github.com/ambideXtrous9/ambideXtrous9/blob/main/icons/huggingface.png?raw=true" width="50" height="50"/>
            <img src="https://newrelic.com/sites/default/files/styles/medium/public/quickstarts/images/icons/langchain--logo.png" width="50" height="50"/>
            <img src="https://miro.medium.com/v2/resize:fit:1196/0*GuAKET2lI82IcBrW.png" width="50" height="50"/>
            <img src="https://miro.medium.com/v2/resize:fit:1400/0*-7HC-GJCxjn-Dm7i.png" width="50" height="50"/>
            <img src="https://github.com/ambideXtrous9/ambideXtrous9/blob/main/icons/lightning.png?raw=true" width="50" height="50"/>
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linux/linux-original.svg" width="50" height="50"/>
        </div>

        ---
        
        - Contributing to the ML community on [Kaggle](https://www.kaggle.com/sushovansaha9)  
        - Writing practical ML blogs on [Medium](https://medium.com/@sushovansaha95)  
        - Exploring new AI tools, frameworks, and academic papers to stay at the cutting edge  

        ⚡ I'm always eager to collaborate on innovative projects, exchange ideas, and learn from the community. Let’s build something amazing with AI! 🤝  

        ---

        
    """,
    unsafe_allow_html=True,
    )
        
    glowingSocial()

    
    
    


def GitHubStats():
    st.title(":rainbow[GitHub Stats]")
    username = "ambideXtrous9"  # Replace with your GitHub username
    response = requests.get(f"https://api.github.com/users/{username}", timeout=10)  # 10 second timeout

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
                

    uploaded_file = st.file_uploader("Upload an Image containing Brand Logo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        uploaded_image = Image.open(uploaded_file)

        # Perform prediction
        prediction_image = predict(uploaded_image)
        
        # Create two columns for side-by-side images
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", width = 200,use_container_width='auto')
            
            
        with col2:
            st.image(prediction_image, caption="Predicted Image", width=200,use_container_width='auto')
            
    glowingYolo()
            
    
            
            
def NewsQA():
    
    ChatBot()
    #glowingLLM()
    