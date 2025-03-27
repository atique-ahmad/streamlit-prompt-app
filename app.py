import streamlit as st
import tiktoken
import openai
import json
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Access the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
# model="gpt-4o-mini"

# Streamlit UI
st.subheader("Generate AI-driven prompts and responses with a hallucination score from given Context/text-file!")
model = st.selectbox("Select Model", [ "gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "o3-mini"])

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

# File upload or text input
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
text_content = ""
if uploaded_file:
    text_content = uploaded_file.read().decode("utf-8")
else:
    text_content = st.text_area("Or paste text here:")

# Function to calculate token length
def get_token_count(text):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)


# Display token count for text content
if text_content.strip():
    token_count = get_token_count(text_content)
    st.success(f"Token length of the provided context: {token_count}")
# User Inputs
prompt_type = st.selectbox("Select Prompt Type", ["Analytical", "Descriptive", "Problem-Solving", "Opinion-Based", "Informational"])
num_prompts = st.slider("Number of Prompts", 1, 50, 10)
hallucination_score = st.slider("Hallucination Score (%)", 0, 100, 10)


# Function to calculate hallucination score based on context and response
def generate_hallucination_score(response, context):
    system_prompt = (
        f"Here are the instructions using the formula to calculate the Hallucination score:\n"
        f"n = Count the total number of statements in the response (a statement is defined as a full stop sentence).\n"
        f"Number of Incorrect Statements = Identify the number of incorrect statements based on the given context or out-of-context statements. An incorrect statement means it is not relevant to the context, or its meaning is the opposite of the context.\n"
        f"Formula Hallucination = (Number of Incorrect Statements / n) * 100\n\n"
        f"[Example]\n"
        f"{{\n"
        f"    \"correct_statements\": [\"This is a correct statement.\", \"This is another correct statement.\"],\n"
        f"    \"incorrect_statements\": [\"Here is an incorrect statement.\", \"Another wrong statement.\"],\n"
        f"    \"correct\": 2,\n"
        f"    \"incorrect\": 2,\n"
        f"    \"total\": 4,\n"
        f"    \"hallucination\": 50\n"
        f"}}\n\n"
        
        f"[Output Format]\n"
        f"{{\n"
        f"    \"correct_statements\": [\"\", \"\", \"\"],\n"
        f"    \"incorrect_statements\": [\"\", \"\", \"\"],\n"
        f"    \"correct\": ,\n"
        f"    \"incorrect\": ,\n"
        f"    \"total\": ,\n"
        f"    \"hallucination\": \n"
        f"}}\n\n"
        
        f"Please follow this output format exactly, with no extra context.\n"
        f"Ensure that both the incorrect_statements and correct_statements are full stop sentences as per the definition of a statement."
    )

    user_prompt = (
        f"Context: {context}\n\n"
        f"Response: {response}\n\n"
        f"Please verify how many statements in the response are correct and how many are incorrect based on the context.\n"
        f"Provide your answer in the following format:\n"
        f"[Output Format]\n"
        f"{{\n"
        f"    \"incorrect_statements\": [\"\", \"\", \"\"],\n"
        f"    \"correct_statements\": [\"\", \"\", \"\"],\n"
        f"    \"incorrect\": ,\n"
        f"    \"correct\": ,\n"
        f"    \"total\": ,\n"
        f"    \"hallucination\": \n"
        f"}}\n\n"
        f"Please follow the output format exactly, with no extra context."
    )

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
        return response_json
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to generate prompts
def generate_n_prompts(context, prompt_type, n):
    """
    Generates 'n' prompts based on the given context and prompt type.
    """
    system_prompt = (
        "[Instructions] Generate a mix of simple and well-structured prompts based on the provided text. "
        "You are an advanced AI system designed to create high-quality, enterprise-grade prompts. "
        "These prompts will be used within an enterprise chat application to enhance context-aware and effective interactions.\n\n"
        
        "Ensure variety in complexity—some prompts should be simple and direct."
        "Avoid overusing 'What' or 'How' at the beginning unless necessary for clarity. "
        "Keep prompts concise, engaging, and relevant.\n\n"
        f"Ensure they are relevant, well-structured, and match the specified type ('{prompt_type}'). "

        "[Examples]\n"
        "{ \"prompt_1\": \"Explain the role of oxygen in respiration.\",\n"
        "  \"prompt_2\": \"Analyze the impact of economic policies on inflation trends.\",\n"
        "  \"prompt_3\": \"Describe the key differences between classical and quantum computing.\" }\n\n"
        
        "[OutputFormat]\n"
        "{ \"prompt_1\": \"\", \"prompt_2\": \"\", \"prompt_3\": \"\", ... }"
    )

    user_prompt = f"Based on the given context: '{context}', generate {n} prompts of type '{prompt_type}', ""ensuring adherence to the provided instructions and examples."

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
        f"[Instructions] Generate a concise, contextually accurate, and coherent response based on the provided context. "
        f"Ensure that the hallucination score does not exceed {hallucination_score}%. "
        "This means the proportion of incorrect or out-of-context statements should remain within the specified limit.\n\n"
        
        "Hallucination Score Calculation:\n"
        "- A score of **0%** means all statements are correct and fully based on the given context.\n"
        "- A score of **50%** means that half of the statements in the response are incorrect or out of context.\n"
        "- A score of **100%** means every statement in the response is incorrect or unrelated to the context.\n\n"
        
        "If necessary, minor details can be added to maintain logical flow, but they must be relevant and factually accurate.\n\n"
        
        "[Example]\n"
        "{ \"response\": \"X is true based on the context. Some suggest Y, but it is not fully supported.\" }\n\n"
        
        "[Output Format]\n"
        "{ \"response\": \"\" }"
    )

    user_prompt = (
        f"Context: {context}\n\n"
        "Generate a concise and accurate response for:\n"
        f"{prompt}\n\n"
        f"Ensure the response strictly adheres to the given context while maintaining factual accuracy. "
        f"The hallucination score must not exceed {hallucination_score}%, meaning the proportion of incorrect statements should remain within this limit."
    )


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

# Process data to remove 'error' and move 'raw_output' to 'response' as JSON
def clean_json_responses(data):
    for item in data:
        if "response" in item and "raw_output" in item["response"]:
            try:
                # Parse the raw_output string as JSON
                raw_output = item["response"]["raw_output"]
                #  Remove the beginning part
                cleaned_output = raw_output.split('{ "response": "', 1)[-1]
                # Remove the ending ' }'
                cleaned_output = cleaned_output.rsplit('" }', 1)[0]
                item["response"] = cleaned_output  
            except Exception as e:
                print(f"Error processing item: {e}")
    return data

def save_to_json(data):
    """Saves the responses to a JSON file and provides it for download."""
    json_output = json.dumps(data, indent=4)
    st.download_button("Download JSON", json_output, "generated_responses.json", "application/json")
    st.write("Responses saved to generated_responses.json")

def saveprocessed_data_json(data):
    json_output = json.dumps(data, indent=4)
    st.download_button("Download JSON", json_output, "processed_data_generated.json", "application/json")
    st.write("Responses saved to processed_data_generated.json")

def save_to_csv(data):
    """Saves the responses to a CSV file and provides it for download."""
    csv_output = "Prompt,Response,Hallucination Score\n"
    for row in data:
        csv_output += f'"{row["prompt"]}","{row["response"]}",{row["hallucination_score"]}\n'
    st.download_button("Download CSV", csv_output, "generated_responses.csv", "text/csv")
    st.write("Responses saved to generated_responses.csv")
    # csv_output = "Prompt\n"
    # for row in data:
    #     csv_output += f'"{row["prompt"]}"\n'
    # st.download_button("Download CSV", csv_output, "generated_prompts.csv", "text/csv")
    # st.write("Responses saved to generated_prompts.csv")
    
# Generate prompts & responses on button click
if st.button("Generate Prompts & Responses"):
    if not text_content.strip():
        st.warning("Please provide text input.")
    else:
        st.write("Generating 'n' prompts based on the given context and prompt type...")
        prompts = generate_n_prompts(text_content, prompt_type, num_prompts)
        st.json(prompts)
        if "error" not in prompts:
                responses_data = []

                st.write("Generating a responses based on the context while considering hallucination...")
                for key, prompt in prompts.items():
                    response_text = generate_response(prompt, text_content, hallucination_score)
                    responses_data.append({
                        "prompt": prompt,
                        "response": response_text,
                        "hallucination_score": hallucination_score
                    })

                # Save to JSON and CSV
                st.json(responses_data)
                cleaned_data = clean_json_responses(responses_data)
                save_to_csv(responses_data) # just prompt save here
                # st.write("Cleaning JSON data...")
                # st.json(cleaned_data)
                # save_to_json(cleaned_data)
                # save_to_csv(cleaned_data)
                # # calculate hallucination score based on context and response
                # st.write("Calculate hallucination score based on context and response...")
                # processed_data = []
                # for entry in cleaned_data:
                #     prompt = entry['prompt']
                #     response = entry['response']
                #     hallucination_data = generate_hallucination_score(response, text_content)
                #     if hallucination_data:
                #         processed_data.append({
                #             "prompt": prompt,
                #             "response": response,
                #             "context": text_content,
                #             "correct_statements": hallucination_data["correct_statements"],
                #             "incorrect_statements": hallucination_data["incorrect_statements"],
                #             "correct": hallucination_data["correct"],
                #             "incorrect": hallucination_data["incorrect"],
                #             "total": hallucination_data["total"],
                #             "hallucination": hallucination_data["hallucination"]
                #         })
                #     else:
                #         print(f"Error processing prompt: {prompt}")

                # st.json(processed_data)
                # saveprocessed_data_json(processed_data)
               
        else:
            st.error("Failed to generate valid prompts.")


# st.write("Streamlit & OpenAI API")

