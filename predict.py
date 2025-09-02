import sys
import os
import torch
import esm
import keras
import numpy as np
import pandas as pd

# does this even work?
print("Loading ESM-2 model...")
model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
# model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
# model, alphabet = esm.pretrained.esm2_t36_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # FP32

def read_fasta(fp):
    name, seq = None, []
    for line in fp:
        line = line.rstrip()
        if line.startswith(">"):
            if name:
                yield (name, ''.join(seq))
            name, seq = line, []
        else:
            seq.append(line)
    if name:
        yield (name, ''.join(seq))

fasta_filename = sys.argv[1]
output_csv = sys.argv[2]

data = []
with open(fasta_filename) as fp:
    for name, seq in read_fasta(fp):
        data.append((name, seq))

sequence_representations_list = []
chunk_size = 16
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    print(f"Processing sequences {i+1}-{i+len(chunk)} of {len(data)}")
    
    batch_labels, batch_strs, batch_tokens = batch_converter(chunk)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[36], return_contacts=False)
        token_representations = results["representations"][36]

    sequence_representations = []
    for j, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[j, 1:tokens_len-1].mean(0))

    sequence_representations_list.extend(sequence_representations)

X = torch.stack(sequence_representations_list, dim=0).cpu().detach().numpy()

keras_model_path = f"/vast/scratch/users/{os.environ['USER']}/TEMPRO/user/saved_ANNmodels_1500epoch/ESM_15B.keras"
# keras_model_path = f"/vast/scratch/users/{os.environ['USER']}/TEMPRO/user/saved_ANNmodels_1500epoch/ESM_3B.keras"
# keras_model_path = f"/vast/scratch/users/{os.environ['USER']}/TEMPRO/user/saved_ANNmodels_1500epoch/ESM_650M.keras"
print("Loading pretrained Keras model for Tm prediction...")
keras_model = keras.models.load_model(keras_model_path)

print("Predicting Tm...")
predictions = keras_model.predict(X)

results_df = pd.DataFrame({
    "Sequence_ID": [name for name, _ in data],
    "Predicted_Tm": predictions.flatten()
})

results_df.to_csv(output_csv, index=False)
print(f"Done! Results saved to {output_csv}")
