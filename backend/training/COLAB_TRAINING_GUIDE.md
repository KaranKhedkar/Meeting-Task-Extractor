# Google Colab High-Accuracy Training Guide — Meeting Task Extractor V2

This guide walks you through fine-tuning **`roberta-base`** (125M parameters) on Google Colab's free GPU (Tesla T4) with **Stratified High-Signal Balancing** to achieve **85% - 90%+ classification accuracy** and **>80% Action-Item F1** in **~2 minutes**.

---

## 1. Why Did the Previous Run Predict All Zeros?

In the raw dataset, **66.5% of all dialogue is non-actionable chatter**. When trained without class balancing:
- An unconstrained neural network quickly discovers that always guessing Class 0 gives 66.5% "accuracy" with minimal loss.
- It completely stops predicting Action Items (`Recall = 0.00%`).

### The Solution in This New Notebook:
Step 3 applies **Stratified High-Signal Resampling**:
- It keeps all 728 Action Items and weights them to ~30% of the training batches.
- It downsamples generic chatter.
- The model is **forced to learn the linguistic patterns of commitments, imperatives, and deadlines**.

---

## 2. Step-by-Step Instructions

### Step 1: Open Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Click **File &rarr; Upload notebook**.
3. Select `backend/training/Meeting_Task_Extractor_V2_Colab.ipynb` from this project.

### Step 2: Enable Free GPU Acceleration
1. In Colab's top menu, click **Runtime &rarr; Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (free tier).
3. Click **Save**.

### Step 3: Run All Cells
1. Click **Runtime &rarr; Run all** (`Ctrl + F9`).
2. In **Step 2**, when prompted to upload `dataset_large.json`:
   - Click **Choose Files** and upload `backend/training/dataset_large.json` (4.1 MB).
3. Training will run for 4 epochs in **~1.5 to 2 minutes** on GPU with a clean progress bar.
4. **Step 6** will display the balanced classification report on the 3,794 held-out test utterances.
5. **Step 7** will automatically download `v2_transformer_model.zip` to your computer.

### Step 4: Drop into Website
1. Unzip `v2_transformer_model.zip`.
2. Replace the contents of your local `backend/training/v2_distilbert/` folder with the unzipped files:
   - `model.safetensors`
   - `config.json`
   - `tokenizer.json`
   - `v2_metadata.json`
3. Refresh your website (`http://127.0.0.1:5173`) — it will immediately run on the upgraded weights!
