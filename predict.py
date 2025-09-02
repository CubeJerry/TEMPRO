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

# Ensure model is on the right device and pick the final representation layer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.half().to(device)
# model.to(device)
repr_layer = model.num_layers - 1

sequence_representations_list = []
chunk_size = 16  # keep or adjust

for start in range(0, len(data), chunk_size):
    chunk = data[start:start+chunk_size]
    print(f"Processing sequences {start+1}-{start+len(chunk)} of {len(data)}")

    batch_labels, batch_strs, batch_tokens = batch_converter(chunk)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # move tokens to device for model forward
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
        token_representations = results["representations"][repr_layer]  # shape: [B, L, C]

    # For each sequence in the batch, average residue embeddings (skip BOS/EOS)
    for seq_idx, L in enumerate(batch_lens):
        L = int(L.item())
        if L <= 2:
            # empty or too short sequence -> zero vector on CPU
            vec = torch.zeros(token_representations.size(-1), dtype=token_representations.dtype)
        else:
            # mean over residues (tokens 1 .. L-2), move to CPU
            vec = token_representations[seq_idx, 1:L-1].mean(0).cpu()
        sequence_representations_list.append(vec)

    print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB | "
      f"Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB | "
      f"Max allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB | "
      f"Max reserved: {torch.cuda.max_memory_reserved()/1024**3:.2f} GB",
      flush=True)
# Stack embeddings into a single numpy array on CPU
X = torch.stack(sequence_representations_list, dim=0).cpu().numpy().astype(np.float32)


keras_model_path = f"/vast/scratch/users/{os.environ['USER']}/TEMPRO/user/saved_ANNmodels_1500epoch/ESM_15B.keras"
# keras_model_path = f"/vast/scratch/users/{os.environ['USER']}/TEMPRO/user/saved_ANNmodels_1500epoch/ESM_3B.keras"
# keras_model_path = f"/vast/scratch/users/{os.environ['USER']}/TEMPRO/user/saved_ANNmodels_1500epoch/ESM_650M.keras"
print("Loading pretrained Keras model for Tm prediction...")
keras_model = keras.models.load_model(keras_model_path)

print("Predicting Tm...")
predictions = keras_model.predict(X)

results_df = pd.DataFrame({
    "Sequence_ID": [name for name, _ in data],
    "TEMPRO_Tm": predictions.flatten()
})

results_df.to_csv(output_csv, index=False)
print(f"Done! Results saved to {output_csv}")
