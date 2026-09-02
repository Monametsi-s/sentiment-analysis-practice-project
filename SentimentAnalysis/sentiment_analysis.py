import json

import requests


def sentiment_analyzer(text_to_analyse):
    """Analyze the sentiment of the provided text."""

    # URL for the Watson sentiment analysis service.
    url = (
        "https://sn-watson-sentiment-bert.labs.skills.network/"
        "v1/watson.runtime.nlp.v1/NlpService/SentimentPredict"
    )

    # Prepare the request payload with the text to analyze.
    payload = {"raw_document": {"text": text_to_analyse}}

    # Set the model ID required by the sentiment analysis service.
    headers = {
        "grpc-metadata-mm-model-id": (
            "sentiment_aggregated-bert-workflow_lang_multi_stock"
        )
    }

    # Send the POST request and get the response.
    response = requests.post(url, json=payload, headers=headers)

    # Convert the JSON response into a Python dictionary.
    formatted_response = response.json()

    # Extract the sentiment score and label from the response.
    score = formatted_response["documentSentiment"]["score"]
    label = formatted_response["documentSentiment"]["label"]

    # Return the sentiment information as a dictionary.
    return {"score": score, "label": label}


# Test the sentiment analyzer with an example sentence.
# print(sentiment_analyzer("I love this new technology"))