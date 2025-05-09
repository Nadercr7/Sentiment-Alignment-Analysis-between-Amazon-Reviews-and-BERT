

## 🟢 **الجزء الأول: استيراد المكتبات (Imports)**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from tqdm import tqdm
import time
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
````

### ✅ شرح:

* `pandas` و `numpy`: دول أهم مكتبتين للتعامل مع البيانات (جداول وأرقام).
* `matplotlib.pyplot` و `seaborn`: دول بنستخدمهم علشان نرسم جرافيك وشارتات توضح النتائج.
* `nltk`: دي مكتبة لمعالجة اللغة الطبيعية، بنستخدمها علشان نحضر النصوص.
* `tqdm`: بتظهر شريط تقدم (progress bar) لما يكون في عمليات بطيئة شوية.
* `time`: علشان نحسب الوقت اللي الكود بياخده.
* `torch`: دي مكتبة PyTorch بتشتغل في الـ deep learning.
* `transformers`: مكتبة من Hugging Face فيها موديلات جاهزة زي BERT.

---

## ⚙️ **الجزء التاني: الإعدادات الأولية**

```python
nltk.download('punkt', quiet=True)
tqdm.pandas(desc="Processing...")
plt.style.use('ggplot')
pd.set_option('display.max_colwidth', 300)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
start_time = time.time()
```

### ✅ شرح:

* بنحمّل حاجة من NLTK اسمها `punkt` علشان تساعدنا في تقطيع الجُمل والكلام.
* بنخلّي tqdm تشتغل مع pandas علشان تبينلنا شريط التقدم.
* بنضبط شكل الرسومات.
* بنخلي البانداس تعرض النص كامل مش تقطعه.
* بنحدد هنشتغل بكارت الشاشة (GPU) ولا البروسيسور (CPU).
* وبنبدأ نحسب الوقت من اللحظة دي.

---

## 📂 **الجزء التالت: تحميل البيانات**

```python
file_path = 'Amazon Customer Reviews.csv'
try:
    df = pd.read_csv(file_path, index_col='Id', usecols=['Id', 'Score', 'Text'])
    print(f"Loaded {file_path} successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    raise
except ValueError as e:
    print(f"ValueError: {e}. Attempting to read all columns...")
    df = pd.read_csv(file_path, index_col='Id')
    if not all(col in df.columns for col in ['Score', 'Text']):
        raise ValueError("Required columns 'Score' or 'Text' missing.")
```

### ✅ شرح:

* بنحاول نفتح ملف المراجعات (تعليقات الناس على منتجات أمازون).
* لو الملف مش موجود، بيظهر رسالة خطأ.
* لو الأعمدة اللي احنا محتاجينها ناقصة (Score و Text)، بيوقف الكود ويقولك فيه مشكلة.

---

## 🧹 **الجزء الرابع: تنظيف البيانات**

```python
initial_rows = len(df)
df['Text'] = df['Text'].fillna('')
df['Score'] = (
    pd.to_numeric(df['Score'], errors='coerce')
      .fillna(0)
      .astype(int)
)
df = df[df['Text'].str.strip() != '']
rows_after = len(df)
print(f"Removed {initial_rows - rows_after} rows; Current shape: {df.shape}")
```

### ✅ شرح:

* بنشيل أي نص فاضي ونعوّضه بكلام فاضي "".
* بنحول الـ "Score" لأرقام ونعوّض القيم اللي مش مفهومه بـ 0.
* بنشيل الصفوف اللي فيها كلام فاضي (مفيش تعليق).
* وبنحسب قد إيه صفوف اتشالت.

---

## 🤖 **الجزء الخامس: تحميل موديل BERT**

```python
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
print("BERT model and tokenizer loaded.")
```

### ✅ شرح:

* بنحدد اسم الموديل اللي هنستخدمه (BERT متدرب على تعليقات).
* بنحمل الـ tokenizer (بيحول الكلام لأرقام).
* وبنحمل الموديل نفسه اللي بيعمل التصنيف (positive, negative...).

---

## 🧠 **الجزء السادس: تحليل المشاعر بالنص**

```python
def get_bert_sentiment(text, max_length=512):
    ...
```

### ✅ شرح:

* دي دالة بتحلل أي نص وتديك:

  * نسبة المشاعر السلبية (`neg`)
  * نسبة المشاعر المحايدة (`neu`)
  * نسبة المشاعر الإيجابية (`pos`)
  * القيمة المركبة (`compound`) اللي بتلخص المشاعر كلها.

---

## 📊 **الجزء السابع: تطبيق التحليل على كل تعليق**

```python
bert_scores = df['Text'].progress_apply(get_bert_sentiment)
bert_df = pd.json_normalize(bert_scores)
df = pd.concat([df, bert_df], axis=1)
```

### ✅ شرح:

* بنطبق دالة المشاعر على كل تعليق.
* بنحوّل النتايج لأعمدة منفصلة.
* وبنضمهم للجدول الأصلي.

---

## 📏 **الجزء الثامن: تحديد التوافق بين تقييم النجوم والمشاعر**

```python
def check_alignment(row):
    ...
df['Alignment_BERT'] = df.progress_apply(check_alignment, axis=1)
```

### ✅ شرح:

* بنقارن بين رأي الشخص في النجوم (Score) وتحليل النص (BERT).
* لو الاتنين إيجابي → Agreement (Pos-Pos)
* لو الاتنين سلبي → Agreement (Neg-Neg)
* لو في تضاد → Disagreement
* لو محايد → Neutral

---

## 📈 **الجزء التاسع: عرض النتائج**

```python
counts = df['Alignment_BERT'].value_counts()
sns.countplot(data=df, y='Alignment_BERT', order=counts.index)
plt.show()
```

### ✅ شرح:

* بنعد كل نوع توافق حصل.
* وبنرسم جرافيك يوضحهم.

---

## 🔍 **الجزء العاشر: عرض أمثلة على الخلاف**

```python
neg_examples = df[df['Alignment_BERT'] == 'Disagreement (PosScore-NegText)']
pos_examples = df[df['Alignment_BERT'] == 'Disagreement (NegScore-PosText)']
```

### ✅ شرح:

* بنطبع أمثلة لناس ادّوا تقييم عالي بس كلامهم سلبي (والعكس).

---

## 💾 **الجزء الأخير: حفظ النتائج وقياس الوقت**

```python
df[['Score', 'Text', 'neg', 'neu', 'pos', 'compound', 'Alignment_BERT']].to_csv(output_file)
end_time = time.time()
print(f"--- Script End ---\nElapsed time: {end_time - start_time:.2f} seconds")
```

### ✅ شرح:

* بنحفظ البيانات كلها في ملف جديد.
* وبنحسب الوقت اللي الكود أخده من الأول للآخر.
