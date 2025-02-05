import streamlit as st
import openai
import json
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Access the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
model="gpt-4o-mini"
# Check if the API key is available
if not openai_api_key:
    # If the key is not found, prompt the user to input the API key manually
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if not openai_api_key:
        st.warning("Please provide your OpenAI API key to proceed.")
    else:
        # Set the API key to OpenAI
        client = openai.OpenAI(api_key=openai_api_key)
        st.success("API key successfully set!")
else:
    client = openai.OpenAI(api_key=openai_api_key)


# Streamlit UI
st.subheader("Generate AI-driven prompts and responses with a hallucination score!")

# File upload or text input
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
text_content = ""
if uploaded_file:
    text_content = uploaded_file.read().decode("utf-8")
else:
    text_content = st.text_area("Or paste text here:")

# User Inputs
prompt_type = st.selectbox("Select Prompt Type", ["Analytical", "Descriptive", "Problem-Solving", "Opinion-Based", "Informational"])
num_prompts = st.slider("Number of Prompts", 1, 50, 10)
hallucination_score = st.slider("Hallucination Score (%)", 0, 100, 10)

# Function to generate prompts
def generate_n_prompts(context, prompt_type, n):
    """
    Generates 'n' prompts based on the given context and prompt type.
    """
    system_prompt = (
        "[Instructions] Generate well-structured prompts based on the given text. You are an advanced AI system designed to generate high-quality, enterprise-grade prompts. These prompts will be used by our clients within an enterprise chat application to ensure accurate, context-aware, and effective interactions"
        "Avoid starting every question with 'What' or 'How', except for informational prompts where it is allowed. "
        "Ensure variety and clarity in the generated prompts.\n\n"
        "[Examples]\n"
        "{ \"prompt_1\": \"Analyze the impact of economic policies on inflation trends.\",\n"
        "  \"prompt_2\": \"Compare the differences between classical and quantum computing approaches.\" }\n\n"
        "[OutputFormat]\n"
        "{ \"prompt_1\": \"\", \"prompt_2\": \"\", ... }"
    )

    user_prompt = f"{context}, Generate {n} prompts of type {prompt_type} while following the provided instructions and examples."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        generated_prompts = response.choices[0].message.content
        prompts_dict = json.loads(generated_prompts)
        # print(f"Output Prompts: {prompts_dict}")
    except json.JSONDecodeError:
        print("Error: The output is not valid JSON. Returning raw response.")
        return {"error": "Invalid JSON", "raw_output": generated_prompts}

    return prompts_dict


# Function to generate responses with hallucination score
def generate_response(prompt, context, hallucination_score):
    """
    Generates a response for the given prompt based on the context while considering hallucination.
    """
    system_prompt = (
        f"[Instructions] Generate a response based on the given context. "
        f"The response should align with the provided text and must have a hallucination level of {hallucination_score}%. "
        "If the hallucination level is high, introduce some fabricated details while keeping the response coherent.\n\n"
        "[Example]\n"
        "{ \"response\": \"Based on the context, X is true. However, some sources claim Y, which may not be fully supported by the provided text.\" }\n\n"
        "[OutputFormat]\n"
        "{ \"response\": \"\" }"
    )

    user_prompt = f"Context: {context}\n\nGenerate a response for the following prompt:\n{prompt}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        response_text = response.choices[0].message.content
        response_json = json.loads(response_text)

    except json.JSONDecodeError:
        print("Error: The output is not valid JSON. Returning raw response.")
        return {"error": "Invalid JSON", "raw_output": response_text}

    return response_json.get("response", "")

def save_to_json(data):
    """Saves the responses to a JSON file and provides it for download."""
    json_output = json.dumps(data, indent=4)
    st.download_button("Download JSON", json_output, "generated_responses.json", "application/json")
    st.write("Responses saved to generated_responses.json")

def save_to_csv(data):
    """Saves the responses to a CSV file and provides it for download."""
    csv_output = "Prompt,Response,Hallucination Score\n"
    for row in data:
        csv_output += f'"{row["prompt"]}","{row["response"]}",{row["hallucination_score"]}\n'
    st.download_button("Download CSV", csv_output, "generated_responses.csv", "text/csv")
    st.write("Responses saved to generated_responses.csv")
    
# Generate prompts & responses on button click
if st.button("Generate Prompts & Responses"):
    if not text_content.strip():
        st.warning("Please provide text input.")
    else:
        st.write(" Generates 'n' prompts based on the given context and prompt type.\n Generating prompts...")
        prompts = generate_n_prompts(text_content, prompt_type, num_prompts)
        st.json(prompts)
        if "error" not in prompts:
                responses_data = []

                st.write("Generates a response for the given prompt based on the context while considering hallucination.\nGenerating responses for the prompts...")

                for key, prompt in prompts.items():
                    response_text = generate_response(prompt, text_content, hallucination_score)
                    responses_data.append({
                        "prompt": prompt,
                        "response": response_text,
                        "hallucination_score": hallucination_score
                    })

                # Save to JSON and CSV
                st.json(responses_data)
                save_to_json(responses_data)
                save_to_csv(responses_data)
        else:
            st.error("Failed to generate valid prompts.")


st.write("👨‍💻  Streamlit & OpenAI API")

