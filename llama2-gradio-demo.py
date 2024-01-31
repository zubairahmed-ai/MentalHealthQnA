import gradio as gr
from transformers import *
import torch

from ludwig.api import LudwigModel
import pandas as pd
model = LudwigModel.load('/home/ubuntu/lambda_labs/llama2-finetune/results/api_experiment_run_20/model')

theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Open Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
)

def answer_question(question):
  
    test_examples = pd.DataFrame([      
        {
                "instruction": "You are an expert in mental health issues, write a response according to the input below, if a question is not related to mental health, reply with 'I dont't know', if you don't know the answer to something, do not reply. ",                
                "input": question,
        }      
    ])

    predictions = model.predict(test_examples)[0]
        
    response = str(predictions['output_response'][0][0]).strip()
  
    return response

examples = [
    "Is recovery possible for individuals with mental illness?",
    "what is the difference between mental health professionals?",
    "What steps should I take if I encounter someone displaying symptoms of a mental disorder?",
    "what causes mental illness?",
    "what are some of thee warning signs of mental illness?"
]


def process_example(args):
    for x in generate(args):
        pass
    return x

css = ".generating {visibility: hidden}"

with gr.Blocks(theme=theme, analytics_enabled=False, css=css) as demo:
    with gr.Column():
        gr.Markdown(
            """ ## Mental Health Q&A
            
            Type in the box below and click the button to generate answers to your most pressing mental health related questions!
            
      """
        )
        gr.HTML("<p>This is a Q&A for mental health problems, demonstrated to show Llama2-30B model fine-tuned on custom data, created by <a href='https://rewisdom.ai' alt='Rewisdom.AI'>Rewisdom.AI</a> / Zubair Ahmed</p><p>For technical details about the Llama2-30B model and fine-tuning, please check this <a href='https://www.linkedin.com/posts/zubairahmed-ai_promptengineering-opensource-ai-activity-7104740948512813058-Ix9n?utm_source=share&utm_medium=member_desktop' alt='LinkedIn post on detailing of Llama2-30B fine-tuning'>LinkedIn post</a></p><br/><p><strong>Disclaimer</strong>: This chatbot does not provide legal advice or substitute for professional mental health assistance; use with caution.</p>")

        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(placeholder="Enter your mental health related question here", label="Question", elem_id="q-input")

                with gr.Box():
                    gr.Markdown("**Answer**")
                    output = gr.Markdown(elem_id="q-output")
                submit = gr.Button("Generate", variant="primary")
                gr.Examples(
                    examples=examples,
                    inputs=[question],
                    cache_examples=False,
                    fn=process_example,
                    outputs=[output],
                )


    submit.click(answer_question, inputs=[question], outputs=[output])
    question.submit(answer_question, inputs=[question], outputs=[output])

demo.queue(concurrency_count=1).launch(share=True, debug=True)