from transformers import pipeline
import pandas as pd


def load_pipe(task="question-answering"):
    question_answerer = pipeline(task)

    return question_answerer


def main_character(row, question_text):
    # print(row)
    result = question_answerer(question=question_text, context=row)
    return result["answer"]


def predict_using_qa(input_path, output_path):
    data = pd.read_csv(input_path).dropna()
    q1 = "Who is the main character in the text?"
    data["prediction_with_cleaning"] = data["news_merged"].apply(
        main_character, question_text=q1
    )
    data["prediction_no_cleaning"] = data["news_merged"].apply(
        main_character, question_text=q1
    )

    data.to_csv(output_path, index=False)


if __name__ == "__main__":
    question_answerer = load_pipe(task="question-answering")
    predict_using_qa("./data/cleaned_news_all.csv", "./data/predictions.csv")
from transformers import pipeline
import pandas as pd


def load_pipe(task="question-answering"):
    question_answerer = pipeline(task)

    return question_answerer


def main_character(row, question_text):
    # print(row)
    result = question_answerer(question=question_text, context=row)
    return result["answer"]


def predict_using_qa(input_path, output_path):
    data = pd.read_csv(input_path).dropna()
    q1 = "Who is the main character in the text?"
    data["prediction_with_cleaning"] = data["news_merged"].apply(
        main_character, question_text=q1
    )
    data["prediction_no_cleaning"] = data["news_merged"].apply(
        main_character, question_text=q1
    )

    data.to_csv(output_path, index=False)


if __name__ == "__main__":
    question_answerer = load_pipe(task="question-answering")
    predict_using_qa("./data/cleaned_news_all.csv", "./data/predictions.csv")
