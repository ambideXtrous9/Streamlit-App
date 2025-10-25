from langchain_core.output_parsers.pydantic import PydanticOutputParser
from pydantic import ValidationError
from pydantic import BaseModel, Field
from typing import Literal
from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv
import os 

load_dotenv()

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

model_name = "openai/gpt-oss-20b"
temperature = 0


llm = ChatGroq(
    model_name=model_name,
    temperature=temperature)  

class Classify(BaseModel):
    classification: Literal["harry", "generic", "exit"] = Field(
        description="Classify the topic type: 'generic' or 'harry' or 'exit'"
    )
    reply: str = Field(
        description="If 'generic', a natural language reply to the query. If 'harry', respond with 'Harry Potter'. If 'exit', respond with 'exit'."
    )

parser = PydanticOutputParser(pydantic_object=Classify)

format_instructions = parser.get_format_instructions()



def AgentClassifyNode(topic):

    max_retries = 3
    attempt = 0
    
    prompt=(
        """
        You are an expert in Harry Potter Universe.
        Your output MUST conform exactly to this JSON schema (no extra fields):
        {format_instructions}
        Add a key `classification` with value either \"generic\" or \"harry\" or \"exit\":\n
        - If the query is a greeting or unrelated to Harry Potter topics, set `classification` to \"generic\".\n
        - If the query is related to Harry Potter or Mix of Indian Mythology and Harry Potter topics set `classification` to \"harry\".\n\n
        - If the query is to exit the conversation, set `classification` to \"exit\".\n\n
        Also add a key `reply`:\n
        - If `classification` is 'generic', `reply` must be a natural language response to the query referring to the chat history for context. Also encourage user to ask Harry Potter related questions.\n
        - If `classification` is 'harry', `reply` must be the string \"Harry Potter\".\n\n
        - If `classification` is 'exit', `reply` must be the string \"exit\".\n\n
        IMPORTANT: The output must be *only* the JSON object—no extra text or reasoning.\n
        """
    )
    
    sys_prompt = {
        "role": "system",
        "content": prompt.format(format_instructions=format_instructions)
    }

    user_msg = {"role": "user", "content": f"Classify this topic : {topic}"}

    while attempt < max_retries:
        
        attempt += 1
        
        # Run agent and capture full assistant output (stream or no-stream)
        response = llm.invoke([sys_prompt, user_msg])

        try :
            # Parse the final JSON into Pydantic model
            article: Classify = parser.parse(response.content)
            return article.model_dump()

        except ValidationError as e:
            print(f"[Attempt {attempt}] Parsing failed:", e)
            # Optionally modify the prompt to highlight the error:
            user_msg = {"role": "user", "content": f"Classify this topic : {topic}\n\nNote: Your previous output did not match the required JSON schema. Please fix it exactly."}
            continue

    # If all attempts fail, raise or return empty/default
    raise RuntimeError(f"Failed to get valid ArticleDraft JSON after {max_retries} attempts.")




def classify_node(state):
    
    print("Node : classify_node")
    
    res = AgentClassifyNode(state['topic'])

    print("Classification Result : ", res)
    
    if isinstance(res, dict):
        state['classification'] = res
    
    return state