import google.generativeai as genai
import os 
os.environ["GOOGLE_API_KEY"] = "AIzaSyA2xj6zQDRQ6Nd08SncwBkIDC40G6YDTVk"
genai.configure(api_key="AIzaSyA2xj6zQDRQ6Nd08SncwBkIDC40G6YDTVk")
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import requests

# URL of the file to be downloaded
file_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRJTKOiktNd3uYtRqEE83tgHFRsfc5fotZio5oceJkeBg&s'
# Send a GET request to the URL
response = requests.get(file_url)

# Check if the request was successful
if response.status_code == 200:
    # Open a local file in write-binary mode
    with open('downloaded_file.txt', 'wb') as file:
        # Write the content of the response to the file
        file.write(response.content)



llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")
# example
image_prompt=''' you are an assistant tasked with summarizing images for retrieval. \
    these summaries will be embedded and used to retrieve the raw image . \
        given a conise summary of the image that is well optimized for retrieval.'''

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": image_prompt,
        },  # You can optionally provide text parts
        {"type": "image_url", "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRJTKOiktNd3uYtRqEE83tgHFRsfc5fotZio5oceJkeBg&s"},
    ]
)
print(llm.invoke([message]).content)