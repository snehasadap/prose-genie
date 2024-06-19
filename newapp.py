import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


def preprocess_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)


def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset


def create_data_collator(tokenizer):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    return data_collator


def generate_text(model, tokenizer, prompt, max_length=400, temperature=0.7, top_p=0.9, top_k=50):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=True,  
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

@app.route('/')
def index():
    html_code = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Prose Genie</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=IM+Fell+English:ital@0;1&family=Special+Elite&display=swap" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=PT+Serif&display=swap" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Silkscreen:wght@400;700&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: Arial, sans-serif;
                height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
                background-color: #262626;
                color: #FFFFFF;
            }
            .main-title {
                text-align: center;
                margin-bottom: 20px;
            }
            .main-title h1 {
                font-size: 3rem;
                font-family: 'IM Fell English', serif;
                font-weight: 400;
                font-style: normal;
                margin: 0;
                color: #FFFFFF;
            }
            .im-fell-english-regular {
                font-family: 'IM Fell English', serif; 
                font-weight: 400;
                font-style: normal;
                color: #000000;
            }
            .pt-serif-regular {
                font-family: "PT Serif", serif;
                font-weight: 400;
                font-style: normal;
            }
            .silkscreen-regular {
                font-family: "Silkscreen", sans-serif;
                font-weight: 400;
                font-style: normal;
            }
            .site-description {
                font-weight: 400;
                font-style: italic;
                font-size: 1.2rem;
                color: #A5A9A6;
                text-align: center;
                margin-top: 10px;
            }
            .form-container {
                width: 100%;
                max-width: 600px;
                margin: 0 auto;
                background-color: #EAD1E6; 
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.1);
                margin-top: 20px;
            }
            label {
                font-weight: bold;
                color: #555555;
            }
            .prompt-container {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 10px;
            }
            textarea {
                width: calc(100% - 38px);
                height: 50px;
                padding: 10px;
                border: 3px solid #9F95B4;
                border-radius: 3px;
                resize: none;
                margin-bottom: 5px;
                margin-right: 10px;
                outline: 3px solid #9F95B4;
            }
            .generated-text {
                font-family: 'IM Fell English', serif;
                color: #9966CC; 
                margin-bottom: 20px;
                font-size: 1.2rem;
                line-height: 1.6;
            }
            input[type="submit"] {
                background-color: #93E9BE;
                color: #37664F;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: block;
                font-size: 16px;
                margin: 20px auto 30px auto;
                cursor: pointer;
                border-radius: 3px;
                font-family: 'IM Fell English', serif;
                font-weight: 400;
                font-style: normal;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
                width: auto;
                max-width: 200px;
            }
            input[type="submit"]:hover {
                background-color: #84D1AB;
            }
            .char-count {
                font-size: 40px;
                color: #FFFFFF;
                font-family: 'IM Fell English', serif;
                position: absolute; 
                right: 30px; 
                bottom: -25px;
            }
            .button-74 {
                background-color: #fbeee0;
                border: 2px solid #422800;
                border-radius: 30px;
                box-shadow: #422800 4px 4px 0 0;
                color: #422800;
                cursor: pointer;
                display: inline-block;
                font-weight: 600;
                font-size: 18px;
                margin: 10px auto 0 auto;
                padding: 0 18px;
                line-height: 50px;
                text-align: center;
                text-decoration: none;
                user-select: none;
                -webkit-user-select: none;
                touch-action: manipulation;
            }
            .button-74:hover {
                background-color: #fff;
            }
            .button-74:active {
                box-shadow: #422800 2px 2px 0 0;
                transform: translate(2px, 2px);
            }
            @media (min-width: 768px) {
                .button-74 {
                    min-width: 120px;
                    padding: 0 25px;
                }
            }
            .loading-message-container {
                display: flex; 
                align-items: center; 
            }
            .loading-message {
                font-family: 'Silkscreen'
                font-size: 1rem;
                color: #59485C; 
            }
            .generated-title {
                font-family: 'IM Fell English', serif; 
                margin-top: 20px; 
                font-size: 1.5rem; 
                color: #000000; 
                text-align: center; 
                display: none;
            }
            p {
                margin-top: 20px;
                line-height: 1.6;
            }
            a {
                color: #1e90ff;
                text-decoration: none;
                margin-top: 20px;
                display: inline-block;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    
    <body>
        <div class="main-title">
            <h1 class="im-fell-english-regular">Prose Genie</h1>
            <p class="pt-serif-regular site-description">An AI-powered tool that effortlessly generates human-like text. Use the prosed prompt as your writing base, allowing you to focus more on refining your content.</p>
        </div>
        <div class="form-container">
            <form id="prompt-form">
                <textarea id="prompt" name="prompt" rows="4" cols="45" maxlength="100" placeholder="Enter your prompt to get started"></textarea><br><br>
                <div class="loading-message-container">
                    <button type="submit" class="button-74 im-fell-english-regular">Create Prose</button>
                    <div id="loading-message" style="display: none; font-family: 'Silkscreen', sans-serif; color: #59485C;">Generating text...</div>
                </div>
                <span class="char-count">100</span>
            </form>
            <h2 class="generated-title">Prosed Prompt</h2>
            <div class="im-fell-english-regular" id="generated-text"></div>
        </div>
        <script>
            const form = document.getElementById('prompt-form');
            const promptInput = document.getElementById('prompt');
            const generatedTextDiv = document.getElementById('generated-text');
            const charCountSpan = document.querySelector('.char-count');
            const generateButton = document.getElementById('generate-button');

            promptInput.addEventListener('input', () => {
                const remainingChars = promptInput.maxLength - promptInput.value.length;
                charCountSpan.textContent = remainingChars;
            });

            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                await generateText(promptInput.value);
            });

            generateButton.addEventListener('click', async () => {
                await generateText(promptInput.value);
            });

            async function generateText(prompt) {
                const loadingMessage = document.getElementById('loading-message');
                loadingMessage.style.display = 'block'; // Show loading message
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ prompt })
                });
                const data = await response.json();
                loadingMessage.style.display = 'none';
                let generatedTextTitle = document.getElementById('generated-text-title');
                if (!generatedTextTitle) {
                    generatedTextTitle = document.createElement('h2');
                    generatedTextTitle.id = 'generated-text-title';
                    generatedTextTitle.textContent = "Prosed Prompt:";
                    generatedTextTitle.style.fontFamily = 'IM Fell English, serif';
                    generatedTextTitle.style.color = '#000000'; // Set title color to black
                    generatedTextDiv.parentNode.insertBefore(generatedTextTitle, generatedTextDiv);
                }
                
                
                generatedTextDiv.textContent = data.generated_text;
                generatedTextDiv.style.color = "#000000"; // Set text color to black
            }

        </script>
    </body>
    </html>
    """
    return html_code
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    generated_text = generate_text(fine_tuned_model, fine_tuned_tokenizer, prompt)
    return jsonify({'generated_text': generated_text})

if __name__ == "__main__":
    model_name = "gpt2" 
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    file_path = "essays.txt"


    preprocess_text(file_path)


    train_dataset = load_dataset(file_path, tokenizer)


    data_collator = create_data_collator(tokenizer)


    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=200,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        prediction_loss_only=True,
        logging_dir='./logs',
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )


    trainer.train()


    model.save_pretrained("./gpt2-finetuned")
    tokenizer.save_pretrained("./gpt2-finetuned")


    fine_tuned_model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
    fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")


    app.run(debug=True)

