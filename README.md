# Fine-tuning Gemma for Idiom Understanding with Fine-tuned LLMs 🌍

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/keras-3.0%2B-red)](https://keras.io/)

A specialized fine-tuning of Google's Gemma 2B model to understand and generate culturally-appropriate idioms across 71 languages. 

This project is my "Google's Unlock Global Communcation" submission which enables cross-cultural/ language communication by matching figurative meanings with appropriate idioms while preserving cultural context.

## 🎯 Project Highlights

- **Multilingual Support**: Covers 71 languages from diverse cultural backgrounds
- **Cultural Preservation**: Maintains cultural context and nuances in translations
- **Efficient Fine-tuning**: Uses LoRA to reduce training parameters by 99.9%
- **High Accuracy**: Achieves 79.01% accuracy in idiom matching

## 📊 Dataset

Our custom dataset includes:
- 10 idioms per language
- 72 languages
- 720 total examples

Each entry contains:
```json
{
  "idiom": "Original idiom",
  "literal_meaning": "Word-for-word translation",
  "figurative_meaning": "Actual meaning",
  "example": "Usage example",
  "language": "Source language"
}
```

## 🛠️ Technical Implementation

### Model Architecture
- Base Model: `Gemma 2B en`
- Fine-tuning Method: Low-Rank Adaptation (LoRA)
  - Rank: 4
  - Original Parameters: 2.6B (9.74 GB)
  - Trainable Parameters: 2.9M (11.17 MB)

### Training Configuration
```python
# Model settings
sequence_length = 256 # to manage resources 
batch_size = 1
epochs = 2 # the accuracy is increased by 12%

# Optimizer
optimizer = AdamW(
    learning_rate=5e-5,
    weight_decay=0.01
)
```

## 📈 Performance

Training Metrics:
```
Epoch 1: Loss: 0.5887, Accuracy: 63.51%
Epoch 2: Loss: 0.3230, Accuracy: 79.01%
```

## 🚀 Quick Start

1. **Installation**
```bash
pip install -q -U keras-nlp
pip install -q -U keras>=3
```

2. **Environment Setup**
```python
import os
os.environ['KERAS_BACKEND'] = 'jax'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"
```

3. **Load Model and Generate**
```python
test_meaning = "to be stuck in a difficult situation"
prompt = (
    "Instruction:\n"
    "Find a suitable idiom for this situation: {}\n\n"
    "Response:\n"
).format(test_meaning)

sampler = keras_nlp.samplers.TopKSampler(k=7, seed=2)
print(model.generate(prompt, max_length=512))
```

## 📋 Sample Output

```
Instruction:
Find a suitable idiom for this situation: to be stuck in a difficult situation

Response:
Idiom: Bıçakta kalamak
Literal Meaning: To be stuck in a knife
Example Use: You might find yourself stuck in a knife if you don't find a solution quickly
Cultural Context: This idiom comes from the Turkish culture
```

## 🌟 Features

- ✅ Cross-cultural idiom matching with contextual understanding.
- ✅ Preservation of cultural context
- ✅ Support for 72 languages
- ✅ Efficient fine-tuning with LoRA

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google's Gemma team for the base model
- Kaggle for the competition platform
