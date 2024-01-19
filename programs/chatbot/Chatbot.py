import openai

# Set your OpenAI API key
openai.api_key = 'YOUR_API_KEY'

# Define user input
user_input = "Tell me a joke"

# Make API request to OpenAI RAG model
response = openai.Completion.create(
  model="text-davinci-003",  # Replace with the actual model name
  prompt=user_input,
  temperature=0.7,
  max_tokens=150
)

# Extract and print the generated response
generated_response = response['choices'][0]['text']
print(generated_response)