from pathlib import Path
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

model_path = Path("models/wav2vec2_finetuned_model_run")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    model_path,
    local_files_only=True,
).to(device)

processor = Wav2Vec2Processor.from_pretrained(
    model_path,
    local_files_only=True,
)

model.eval()
print(device, model.config.id2label)