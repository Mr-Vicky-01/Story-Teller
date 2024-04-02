from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain import HuggingFaceHub, LLMChain, PromptTemplate
import gradio as gr
import numpy as np
import requests
import os

# Load image captioning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption_from_image(image_path):
    # Process the image and generate caption
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_story_from_caption(caption):
    # Generate story based on caption
    llm = HuggingFaceHub(huggingfacehub_api_token=os.getenv('HUGGING_FACE'),
                            repo_id="tiiuae/falcon-7b-instruct",
                            verbose=False,
                            model_kwargs={"temperature": 0.2, "max_new_tokens": 4000})
    template = """You are a story teller.
    You get a scenario as an input text, and generate a short story out of it.
    Context: {scenario}
    Story:"""
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    # Let's create our LLM chain now
    chain = LLMChain(prompt=prompt, llm=llm)
    story = chain.run(caption)
    start_index = story.find("Story:") + len("Story:")
    # Extract the text after "Story:"
    story = story[start_index:].strip()
    return story

def text_to_speech(text):
    headers = {"Authorization": f"Bearer {os.getenv('HUGGING_FACE')}"}
    payload = {"inputs": text}
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        with open("output.mp3", "wb") as f:
            f.write(response.content)
        return "output.mp3"

def generate_story_from_image(image_input):
    input_image = Image.fromarray(image_input)
    input_image.save("input_image.jpg")
    image_path = 'input_image.jpg'
    caption = generate_caption_from_image(image_path)
    story = generate_story_from_caption(caption)
    audio = text_to_speech(story)
    return audio


# Define the input and output components
inputs = gr.Image(label="Image")
outputs = gr.Audio(label="Story Audio")

# Create the Gradio interface
gr.Interface(fn=generate_story_from_image, inputs=inputs, outputs=outputs, title="Story Teller").launch(debug=True)