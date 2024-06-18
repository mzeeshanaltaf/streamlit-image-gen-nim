import streamlit as st
import requests
import base64

# --- PAGE SETUP ---
# Initialize streamlit app
page_title = "Image Generator using NVIDIA NIM"
page_icon = "🖼️"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")

model_api = {'stable-diffusion-3-medium': st.secrets['stable-diffusion-3-medium'],
             'stable-diffusion-xl': st.secrets['stable-diffusion-xl'],
             'sdxl-lightning': st.secrets['sdxl-lightning'],
             'sdxl-turbo': st.secrets['sdxl-turbo']}

st.title(f"🖼️ {page_title}")
st.write("***A Streamlit Image Generator powered by NVIDIA Inference Microservice (NIM)***")

st.subheader('Diffusion Model Selection')
image_model = st.selectbox('Select the Image Generation Model', model_api.keys())
# URL and API Header for NVIDIA NIM
invoke_url = f"https://ai.api.nvidia.com/v1/genai/stabilityai/{image_model}"
headers = {
    "Authorization": f"Bearer {model_api[image_model]}",
    "Accept": "application/json",
}

st.subheader("Enter the Image Description")
prompt = st.text_input('Enter the Image Description', placeholder='Enter the Image Description',
                       label_visibility="collapsed")
generate = st.button('Generate Image', type='primary', disabled=not prompt)
if generate:
    st.subheader('Generated Image:')
    with st.spinner('Processing...'):
        if image_model == 'stable-diffusion-xl':
            payload = {
                "text_prompts": [
                    {
                        "text": prompt,
                        "weight": 1
                    },
                    {
                        "text": "",
                        "weight": -1
                    }
                ],
                "cfg_scale": 5,
                "sampler": "K_DPM_2_ANCESTRAL",
                "seed": 0,
                "steps": 25
            }
            response = requests.post(invoke_url, headers=headers, json=payload)
            response.raise_for_status()
            response_body = response.json()
            image_data = base64.b64decode(response_body['artifacts'][0]['base64'])
            st.image(image_data)
        elif image_model == 'sdxl-turbo':
            payload = {
                "text_prompts": [
                    {
                        "text": prompt,
                        "weight": 1
                    }
                ],
                "seed": 0,
                "sampler": "K_EULER_ANCESTRAL",
                "steps": 2
            }
            response = requests.post(invoke_url, headers=headers, json=payload)
            response.raise_for_status()
            response_body = response.json()
            image_data = base64.b64decode(response_body['artifacts'][0]['base64'])
            st.image(image_data)
        elif image_model == 'sdxl-lightning':
            payload = {
                "text_prompts": [{"text": prompt}],
                "seed": 0,
                "steps": 4
            }
            response = requests.post(invoke_url, headers=headers, json=payload)
            response.raise_for_status()
            response_body = response.json()
            image_data = base64.b64decode(response_body['artifacts'][0]['base64'])
            st.image(image_data)
        elif image_model == 'stable-diffusion-3-medium':
            payload = {
                "prompt": prompt,
                "cfg_scale": 5,
                "aspect_ratio": "16:9",
                "seed": 0,
                "steps": 50,
                "negative_prompt": ""
            }
            response = requests.post(invoke_url, headers=headers, json=payload)
            response.raise_for_status()
            response_body = response.json()
            image_data = base64.b64decode(response_body['artifacts'][0]['base64'])
            st.image(image_data)

