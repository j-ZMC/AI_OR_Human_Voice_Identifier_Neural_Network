inputs = processor(
    audio_array,
    sampling_rate=16000,
    return_tensors="pt",
    return_attention_mask=True,
)

inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.inference_mode():
    logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)
    label_id = probabilities.argmax(dim=-1).item()
    confidence = probabilities[0, label_id].item()

class_name = model.config.id2label[str(label_id)]
print(class_name, confidence)