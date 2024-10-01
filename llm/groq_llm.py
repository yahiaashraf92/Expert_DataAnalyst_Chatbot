import os, getpass

from langchain_groq import ChatGroq

""" def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("GROQ_API_KEY") """



def groq_llm_initializer():
    #return ChatGroq(api_key="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", model="llama3-70b-8192")
    return ChatGroq(api_key="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", model="gemma2-9b-it")