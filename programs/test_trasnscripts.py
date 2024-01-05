import whisper

model = whisper.load_model("base")

result = model.transcribe("C:/Users/KishoreKumar/OneDrive - NGENUX SOLUTIONS PRIVATE LIMITED/Desktop/test/programs/123.mp4")

print(result["text"])