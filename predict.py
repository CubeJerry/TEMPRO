import sys
import os
import torch
import esm
import keras
import numpy as np
import pandas as pd

print("Loading ESM-2 3B model...")
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # CPU, FP32 (as in notebook)

REPR_LAYER = 36  # CRITICAL: must match training

def read_fasta(fp):
    name, seq = None, []
    for line in fp:
        line = line.rstrip()
        if line.startswith(">"):
            if name:
                yield (name.lstrip(">"), ''.join(seq))
            name, seq = line, []
        else:
            seq.append(line)
    if name:
        yield (name.lstrip(">"), ''.join(seq))

fasta_filename = sys.argv[1]
output_csv = sys.argv[2]

data = []
with open(fasta_filename) as fp:
    for name, seq in read_fasta(fp):
        data.append((name, seq))

sequence_representations_list = []
chunk_size = 25  # matches notebook

for i in range(0, len(data), chunk_size):
    chunk = data[i:i + chunk_size]
    print(f"Processing {i + len(chunk)} / {len(data)}")

    batch_labels, batch_strs, batch_tokens = batch_converter(chunk)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(
            batch_tokens,
            repr_layers=[REPR_LAYER],
            return_contacts=False
        )
        token_representations = results["representations"][REPR_LAYER]

    seq_reps = []
    for j, L in enumerate(batch_lens):
        L = int(L)
        seq_reps.append(
            token_representations[j, 1:L-1].mean(0)
        )

    sequence_representations_list.append(seq_reps)

# flatten exactly like notebook
flat_list = [item for sublist in sequence_representations_list for item in sublist]
X = torch.stack(flat_list).cpu().numpy().astype(np.float32)

keras_model_path = (
    f"/vast/scratch/users/{os.environ['USER']}/TEMPRO/user/"
    "saved_ANNmodels_1500epoch/ESM_3B.keras"
)

print("Loading TEMPRO ANN...")
keras_model = keras.models.load_model(keras_model_path)

# HARD SAFETY CHECK
if X.shape[1] != keras_model.input_shape[-1]:
    raise RuntimeError(
        f"Embedding dimension mismatch: "
        f"{X.shape[1]} vs {keras_model.input_shape[-1]}"
    )

print("Predicting Tm (°C)...")
predictions = keras_model.predict(X, verbose=0).flatten()

results_df = pd.DataFrame({
    "Sequence_ID": [name for name, _ in data],
    "TEMPRO_Tm_C": predictions
})

results_df.to_csv(output_csv, index=False)
print(f"Done. Saved to {output_csv}")
