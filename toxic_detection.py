from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline


tokenizer = AutoTokenizer.from_pretrained("laiyer/unbiased-toxic-roberta-onnx")
model = ORTModelForSequenceClassification.from_pretrained("laiyer/unbiased-toxic-roberta-onnx",file_name="model.onnx")
classifier = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
)

classifier_output = classifier("i love black people")
print(classifier_output[])