import warnings
import os
import sys
import logging
import io
import re

# ── Suppress noisy third-party warnings BEFORE any imports ────────────
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*__path__.*")
warnings.filterwarnings("ignore", message=".*LOAD REPORT.*")
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
warnings.filterwarnings("ignore", message=".*Safe alternative available.*")
warnings.filterwarnings("ignore", message=".*Loading SentenceTransformer.*")
warnings.filterwarnings("ignore", message=".*No modules.json.*")
warnings.filterwarnings("ignore", message=".*Loading pretrained weights.*")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Silence loggers at source
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("timm").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)


# ── Filtered stdout: show progress logs, hide noise ──────────────────
class LogFilter(io.TextIOBase):
    """Pass through meaningful logs, suppress noisy import spam."""
    _BLOCK_PATTERNS = [
        "Accessing `__path__`",
        "LOAD REPORT",
        "UNEXPECTED",
        "Safe alternative available",
        "Loading SentenceTransformer",
        "No modules.json found",
        "Loading pretrained weights",
        "Loading weights:",
        "Using device:",
        "Langfuse client is authenticated",
        "Authentication error",
        "WARNING!",
        "seed was transferred",
        "Please confirm that seed",
        "validated_self",
        "missing ScriptRunContext",
        "Authentication failed",
        "Langfuse: Not configured",
        "Langfuse: Authentication failed",
        "UserWarning",
    ]

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, s):
        for pattern in self._BLOCK_PATTERNS:
            if pattern in s:
                return len(s)  # swallow it
        return self._wrapped.write(s)

    def flush(self):
        self._wrapped.flush()

    def writelines(self, lines):
        for line in lines:
            self.write(line)


_old_stdout, _old_stderr = sys.stdout, sys.stderr
sys.stdout = LogFilter(_old_stdout)
sys.stderr = LogFilter(_old_stderr)

def _log(msg):
    """Print directly to real stdout bypassing filter."""
    _old_stdout.write(f"{msg}\n")
    _old_stdout.flush()


_log("🚀 Starting ambideXtrous AI Portfolio...")
_log("⏳ Loading core modules (Streamlit, LangChain, LangGraph)...")

import streamlit as st
import pandas as pd
import numpy as np
import requests
import torch

_log("✅ Core modules loaded")
_log("⏳ Loading Langfuse tracing...")

from langfuse import Langfuse, get_client

# Debug: show what secrets are actually loaded
_pk = st.secrets.get("LANGFUSE_PUBLIC_KEY")
_sk = st.secrets.get("LANGFUSE_SECRET_KEY")
_log(f"🔑 Langfuse PUBLIC_KEY found: {'✅ yes' if _pk else '❌ no (value: ' + str(_pk)[:20] + ')'}")
_log(f"🔑 Langfuse SECRET_KEY found: {'✅ yes' if _sk else '❌ no'}")

if _pk and _sk:
    try:
        Langfuse(
            public_key=_pk,
            secret_key=_sk,
            host="https://us.cloud.langfuse.com"
        )
        langfuse = get_client()
        if langfuse.auth_check():
            _log("✅ Langfuse authenticated successfully")
        else:
            _log("⚠️ Langfuse auth check failed — keys may be expired or incorrect")
    except Exception as e:
        _log(f"⚠️ Langfuse connection failed: {type(e).__name__}. Continuing without tracing.")
        langfuse = None
else:
    _log("⚠️ Langfuse keys missing from .streamlit/secrets.toml. Tracing disabled.")
    langfuse = None

from sidebar import SideBar
from navigate import navigator

_log("✅ App modules loaded")
_log("✅ Module loading complete")

# Restore raw stdout for normal operation
sys.stdout, sys.stderr = _old_stdout, _old_stderr

os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# UI configurations
st.set_page_config(page_title="ambideXtrous",
                   page_icon=":bridge_at_night:",
                   layout="centered")

# ── Langfuse: graceful init (won't crash in Docker without secrets) ───
try:
    Langfuse(
        public_key=st.secrets.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=st.secrets.get("LANGFUSE_SECRET_KEY"),
        host="https://us.cloud.langfuse.com"
    )
    langfuse = get_client()
    if langfuse.auth_check():
        print("Langfuse client is authenticated and ready!")
    else:
        print("Langfuse: Authentication failed. Langfuse disabled.")
except Exception as e:
    print(f"Langfuse: Not configured or unreachable ({e}). Continuing without tracing.")
    langfuse = None

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# or simply:
# torch.classes.__path__ = []

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    

    # If a username is present in session_state, assume logged in for persistence across refreshes
    # Check URL query parameters for persistence
    if "logged_in_user" in st.query_params and st.query_params["logged_in_user"]:
        st.session_state['username'] = st.query_params["logged_in_user"]
        st.session_state['logged_in'] = True
    elif st.session_state['username']:
        st.session_state['logged_in'] = True

    SideBar()
    navigator()

if __name__ == '__main__':
    main()


