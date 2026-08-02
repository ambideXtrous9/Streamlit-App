import streamlit as st
from HarryAgent.HpAgent import GraphBuild
import os
import uuid
import asyncio
import concurrent.futures
from langgraph.checkpoint.memory import MemorySaver
from langfuse.langchain import CallbackHandler

# Langfuse handler: graceful if not configured
try:
    langfuse_handler = CallbackHandler()
except Exception:
    langfuse_handler = None

checkpointer = MemorySaver()
app = GraphBuild(checkpointer)

try:
    with open("graph.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
except Exception as e:
    print(f"Graph image write note: {e}")


def ChatBot():
    if not st.session_state.get('logged_in'):
        st.warning("Please log in to access this feature.")
        return

    # Display chat messages from history on app rerun
    session_key = "hp_agent_messages"
    if session_key not in st.session_state:
        st.session_state[session_key] = []

    for message in st.session_state[session_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    # Suggestion buttons when history is empty
    selected_prompt = None
    if not st.session_state[session_key]:
        st.write("💡 **Suggested Topics & Comparisons:**")
        cols = st.columns(2)
        suggestions = [
            ("🧙‍♂️ Ron vs Lakshmana", "Compare Ron Weasley vs Lakshmana in narrative, duty, and loyalty context"),
            ("🔮 Deathly Hallows & Samsara", "Relate Deathly Hallows to Hindu concepts of Samsara, Karma, and Immortality"),
            ("⚡ Rama's Exile & Horcrux Quest", "Explore parallels between Lord Rama's exile and Harry Potter's quest for Horcruxes"),
            ("📜 Brahmastra vs Avada Kedavra", "Explore the symbolism and ethical constraints of Brahmastra vs Avada Kedavra")
        ]
        for idx, (label, prompt_text) in enumerate(suggestions):
            with cols[idx % 2]:
                if st.button(label, key=f"hp_sug_{idx}", use_container_width=True):
                    selected_prompt = prompt_text

    # React to user input or suggestion button click
    user_input = st.chat_input("Ask a question or topic (e.g. Harry Potter, mythology, or general questions)...")
    prompt = selected_prompt or user_input

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state[session_key].append({"role": "user", "content": prompt})

        thread_id = str(uuid.uuid4())
        callbacks = [langfuse_handler] if langfuse_handler else []

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            text_placeholder = st.empty()

            status_placeholder.caption("🚀 Starting Harry & Mythology Multi-Agent Workflow...")

            async def _stream_hp_agent():
                import time
                start_time = time.time()
                current_label = "🚀 Starting Harry & Mythology Multi-Agent Workflow..."
                status_placeholder.caption(f"{current_label} (0.0s)")

                config = {
                    "configurable": {"thread_id": thread_id},
                    "callbacks": callbacks,
                    "run_name": "hp_agent"
                }

                final_draft = ""
                generic_reply = ""
                is_generic = False
                live_stream_text = ""

                async for event in app.astream_events(
                    input={"topic": prompt, "review": "Write an awesome article on the topic."},
                    config=config,
                    version="v2"
                ):
                    event_type = event.get("event")
                    metadata = event.get("metadata", {})
                    node = metadata.get("langgraph_node")

                    # 1. Step-by-Step Live Agent Progress Updates with Timer
                    if event_type == "on_chain_start" and node:
                        if node == "classify":
                            current_label = "🔍 Router Agent: Analyzing & classifying query..."
                        elif node == "researcher":
                            current_label = "📚 Researcher Agent: Searching Qdrant Cloud vector database..."
                        elif node == "mythologist":
                            current_label = "🕉️ Mythology Agent: Analyzing Indian Mythology & HP parallels..."
                        elif node == "writer":
                            current_label = "✍️ Writer Agent: Authoring comprehensive article draft..."
                        elif node == "critic":
                            current_label = "🧑‍⚖️ Critic Agent: Reviewing accuracy & critique feedback..."

                        elapsed = time.time() - start_time
                        status_placeholder.caption(f"{current_label} ({elapsed:.1f}s)")

                    # 2. Capture Node Output State on Node Completion
                    if event_type == "on_chain_end" and node:
                        output_data = event.get("data", {}).get("output", {})
                        if isinstance(output_data, dict):
                            if "classification" in output_data:
                                cls_info = output_data["classification"]
                                if isinstance(cls_info, dict) and cls_info.get("classification") == "generic":
                                    is_generic = True
                                    generic_reply = cls_info.get("reply", "")
                            if "draft" in output_data and output_data["draft"]:
                                final_draft = output_data["draft"]

                    # 3. Real-time Live Token Streaming with Timer
                    if event_type == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        content = getattr(chunk, "content", "") if chunk else ""
                        if content:
                            elapsed = time.time() - start_time
                            status_placeholder.caption(f"{current_label} ({elapsed:.1f}s)")
                            
                            # Stream tokens during article composition
                            if node in ["writer", "mythologist", "researcher"] or not node:
                                live_stream_text += content
                                text_placeholder.markdown(live_stream_text, unsafe_allow_html=True)
                                await asyncio.sleep(0.005)

                status_placeholder.empty()

                if is_generic and generic_reply:
                    return generic_reply
                elif final_draft:
                    return final_draft
                elif live_stream_text:
                    return live_stream_text
                else:
                    return "No response generated."

            # Run stream directly on Streamlit thread with full ScriptRunContext
            final_content = asyncio.run(_stream_hp_agent())
            text_placeholder.markdown(final_content, unsafe_allow_html=True)
            st.session_state[session_key].append({"role": "assistant", "content": final_content})
