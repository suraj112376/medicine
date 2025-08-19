import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
load_dotenv()

# 🎯 Title
st.title("💊 GenAI Medicine Recommendation App")

st.write("This app suggests medicines based on your symptoms and age using the same model from first.ipynb.")

# ✅ User Inputs
symptoms = st.text_input("Enter your symptoms (e.g., fever, cough, headache):")
age = st.number_input("Enter your age:", min_value=1, max_value=120, step=1)

# ⚡ LLM Setup (copied from your notebook)
llm = ChatGroq(
    model="llama3-8b-8192",  # <-- replace with the exact model you used in notebook
    temperature=0.7
)

# ⚡ Prompt (copied from your notebook)
prompt = PromptTemplate(
    input_variables=["symptoms", "age"],
    template="The patient is {age} years old with the following symptoms: {symptoms}. Suggest suitable medicines."
)

# ⚡ Chain (same as notebook)
chain = prompt|llm


# 🚀 Button
if st.button("Get Medicine Recommendation"):
    if symptoms.strip() == "":
        st.warning("⚠️ Please enter your symptoms before proceeding.")
    else:
        with st.spinner("Generating recommendation..."):
            result = chain.invoke({"symptoms": symptoms, "age": age}) 
            st.success("✅ Recommendation Generated")
            st.write(result)

# Footer
st.caption("Powered by your LLM setup from first.ipynb 🚀")
