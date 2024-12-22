# Gemma Language Model

## Model Overview

- **Model Name**: Gemma Causal Language Model
- **Preprocessor**: `gemma_causal_lm_preprocessor`
- **Tokenizer**: GemmaTokenizer
  - Vocabulary Size: 256,000 tokens

## Model Architecture

- **Backbone**: GemmaBackbone
- **Total Parameters**: 2,614,341,888 (9.74 GB)
  - All parameters are trainable
  - No non-trainable parameters

When LORA was applied, with rank = 4, theese trainable parameters turned into these 11.45mb. Much more easier to work with.
### Layer Details
- Input Layers:
  - Padding Mask
  - Token IDs
- Backbone Layer Output Shape: (None, None, 2304)
- Token Embedding Layer Output Shape: (None, None, 256000)

## Training Considerations
- Full model parameter training
- Large model size requires significant computational resources
- Recommended for advanced machine learning environments

## References
- [Hugging Face Gemma Documentation](https://huggingface.co/google/gemma-2b)
