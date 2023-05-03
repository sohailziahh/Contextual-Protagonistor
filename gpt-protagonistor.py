import openai

openai.api_key = "sk-..."


def extract_protagonist(text):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt="Who is the main protagonist in the following piece of text: " + text,
            temperature=0.2,
            max_tokens=len(text),
            top_p=1.0,
        )
        paraphrased_text = response["choices"][0]['text']
        return paraphrased_text
    except:
        print("Something fucked up with openai!")
        return "failed"


