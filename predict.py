from ludwig.api import LudwigModel
import pandas as pd
model = LudwigModel.load('/home/ubuntu/lambda_labs/llama2-finetune/results/api_experiment_run_20/model')
while True:
    user_input = input("Type something (or 'exit' to quit): ")
    
    if user_input.lower() == 'exit':
        break  # Exit the loop if the user types 'exit'

    test_examples = pd.DataFrame([      
        {
                "instruction": "You are an expert in mental health issues, write a response according to the input below, do not create a response, if you don't know the answer to something, do not reply.",                
                "input": user_input,
        }      
    ])

    predictions = model.predict(test_examples)[0]

    for input_with_prediction in zip(test_examples['instruction'], test_examples['input'], predictions['output_response']):
        print(f"Instruction: {input_with_prediction[0]}")
        print(f"Input: {input_with_prediction[1]}")
        print(f"Generated Output: {str(input_with_prediction[2][0]).strip()}")
        print("\n\n")