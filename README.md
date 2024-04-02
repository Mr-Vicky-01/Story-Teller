# Story Teller

Story Teller is a project that generates short stories from images using image captioning and language generation techniques. It utilizes machine learning models from the Hugging Face Transformers library and LangChain API to process images, generate captions, and produce stories based on those captions. Additionally, it employs a text-to-speech model to convert the generated stories into audio format.

## How it Works

1. **Image Captioning**: The project uses a pre-trained image captioning model (Salesforce/blip-image-captioning-base) to generate captions for input images.
2. **Story Generation**: The generated captions serve as prompts for story generation. The LangChain API, specifically the Falcon-7B Instruct model, is used to generate stories based on the provided prompts.
3. **Text-to-Speech Conversion**: The final generated stories are converted into audio format using the ESPnet model (kan-bayashi_ljspeech_vits) for text-to-speech synthesis.

## Usage

To use the Story Teller:

1. Visit the project's website: [Story Teller](https://huggingface.co/spaces/Mr-Vicky-01/Story_teller)
2. Upload an image for which you want to generate a story.
3. Wait for the system to process the image, generate a caption, and produce a story.
4. Listen to the generated story in audio format.

## Installation

To deploy the project locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/Mr-Vicky-01/Story-Teller.git
cd Story-Teller
pip install -r requirements.txt
python app.py
```

## Dependencies

- PIL: Image processing library
- transformers: Hugging Face Transformers library for machine learning models
- langchain: LangChain API for language generation
- gradio: Gradio library for building interactive UIs
- numpy: Numerical computing library
- requests: HTTP library for making requests to APIs

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to Salesforce for providing the image captioning model.
- Thanks to LangChain for providing the Falcon-7B Instruct model.
- Thanks to Hugging Face for hosting the models used in this project.
- Thanks to ESPnet for providing the text-to-speech model.

## Demo
