import streamlit as st
from HarryAgent.HpAgent import GraphBuild
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.runnables import RunnableConfig
import uuid
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse, get_client


# langfuse = Langfuse(
#   secret_key="sk-lf-a61bf1a3-5388-4960-8de9-67c5b69348fa",
#   public_key="pk-lf-2bc3ac3d-74d7-424e-880c-eee1ddd7f42c",
#   host="http://localhost:3000"
# )


Langfuse(
    public_key=st.secrets.get("LANGFUSE_PUBLIC_KEY"),
    secret_key=st.secrets.get("LANGFUSE_SECRET_KEY"),
    host=st.secrets.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
)
langfuse = get_client()

# Instantiate handler (no args)
langfuse_handler = CallbackHandler()


conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)

app = GraphBuild(checkpointer)

with open("graph.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())




def ChatBot():
    if not st.session_state.get('logged_in'):
        st.warning("Please log in to access this feature.")
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get the assistant's response using the predict function
        # Checkpointer requires one or more of the following 'configurable' keys: thread_id
        thread_id = str(uuid.uuid4())
        thread_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        output = app.invoke(input={"topic": prompt, "review": "Write an awesome article on the topic."}, config={"thread_id":thread_id,"callbacks": [langfuse_handler]})
        
        if output.get("classification") and output.get("classification")["classification"] == "generic":
            st.chat_message("assistant").markdown(output["classification"]["reply"])
            return
            
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(output["draft"])


        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": output["draft"]})
