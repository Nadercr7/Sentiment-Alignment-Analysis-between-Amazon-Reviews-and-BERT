
# ⭐ Sentiment Alignment Analysis between Amazon Reviews and BERT

This project analyzes the **alignment between user star ratings** and **actual sentiment expressed in review text** using a pre-trained **BERT sentiment classifier**. It helps identify discrepancies between how users score a product and what they actually write.

---

## 📌 Project Goals

- Use a multilingual BERT model to classify the sentiment of Amazon reviews.
- Compare BERT-based sentiment with the star rating (1 to 5).
- Detect and visualize alignment vs disagreement between star ratings and review tone.
- Extract meaningful examples of mismatches (e.g., positive star rating but negative text).

---

## 🧠 Model Used

- **Model**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **Source**: Hugging Face Transformers
- **Function**: Classifies input text into 1 to 5 stars, allowing inference of sentiment levels.

---

## 🛠️ Dependencies

Install the following Python libraries before running:

```bash
pip install pandas numpy matplotlib seaborn nltk torch transformers tqdm
````

---

## 📁 Input File

Make sure your dataset is a CSV file with the following columns:

* `Id`: Unique identifier
* `Score`: Star rating (1 to 5)
* `Text`: Review text

📌 Example file: `Amazon Customer Reviews.csv`

---

## 🚀 How It Works

1. **Load and clean the data**: Remove empty rows and normalize types.
2. **Use BERT to analyze text sentiment**.
3. **Compare BERT sentiment with the star score**:

   * Positive star rating vs positive text → ✅ Agreement
   * Positive star rating vs negative text → ❌ Disagreement
   * Neutral or ambiguous → 🤔 Ambiguous
4. **Visualize the results** using Seaborn.
5. **Save results** to a new CSV file.

---

## 📊 Output Sample

| Score | Text                                   | pos  | neu | neg  | compound | Alignment\_BERT                 |
| ----- | -------------------------------------- | ---- | --- | ---- | -------- | ------------------------------- |
| 5     | "It was awful and broken"              | 0.01 | 0.2 | 0.78 | -0.385   | Disagreement (PosScore-NegText) |
| 1     | "Excellent quality and fast delivery!" | 0.75 | 0.1 | 0.1  | +0.325   | Disagreement (NegScore-PosText) |

---

## 📈 Visualizations

A horizontal bar chart is generated showing alignment categories:

* Agreement (Positive-Positive)
* Agreement (Negative-Negative)
* Disagreement (PosScore-NegText)
* Disagreement (NegScore-PosText)
* Neutral/Ambiguous

---

## 💾 Output Files

* `Reviews_with_BERT_Alignment.csv`: Contains the original data + BERT sentiment + alignment category.

---

## ⏱️ Performance

Time is tracked from start to end. If a GPU is available, it's automatically used to accelerate processing.

---

## 📌 Future Work

* Fine-tune the BERT model on domain-specific data.
* Perform deeper analysis of neutral cases.
* Integrate with dashboards for live monitoring.

---

## 🤝 Contributing

Pull requests and feedback are welcome! Feel free to fork and improve.


