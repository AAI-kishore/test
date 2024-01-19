from transformers import RagTokenizer, RagRetriever, RagModel, pipeline

# Load pre-trained models and tokenizers
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained("facebook/rag-token-base")
generator = RagModel.from_pretrained("facebook/rag-token-base")

# Define a function to get responses
def get_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    retriever_outputs = retriever(**inputs)
    generator_inputs = generator(**retriever_outputs)
    responses = tokenizer.batch_decode(generator_inputs["output"], skip_special_tokens=True)
    return responses[0] if responses else "Sorry, I didn't understand."

# Example usage
user_input = "Tell me a joke"
response = get_response(user_input)
print(response)
