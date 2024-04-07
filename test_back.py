import requests
data={
    "pdf_id":"45",
    "user_id":"max",
    "query":"what is TLS",
    "toxic_check":True
}
print(requests.post("https://fb16-14-139-241-214.ngrok-free.app/qna/",json=data))
