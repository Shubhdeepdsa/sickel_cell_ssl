# All Model Results

Macro precision, recall, and F1 are reported on the test crop split for consistency across all trained models.

| Stage | Model | Input | Setup | Label % | Epochs | Batch | LR | Best Val Bal Acc | Test Acc | Test Bal Acc | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Stage 4 | Logistic Regression | RGB | Handcrafted Baseline | 100% | - | - | - | 0.808 | 0.850 | 0.848 | 0.849 | 0.848 | 0.849 | 0.899 |
| Stage 5 | ResNet-18 | RGB | Supervised Scratch | 100% | 20 | 32 | 1e-4 | 0.934 | 0.931 | 0.931 | 0.930 | 0.931 | 0.931 | 0.985 |
| Stage 5 | ResNet-18 | RGB | ImageNet Fine-tune | 100% | 20 | 32 | 1e-4 | 0.995 | 0.988 | 0.986 | 0.989 | 0.986 | 0.988 | 1.000 |
| Stage 5 | ResNet-18 | RGB | SimCLR Fine-tune | 100% | 50+20 | 64+32 | 3e-4 + 1e-4 | 0.979 | 0.985 | 0.985 | 0.985 | 0.985 | 0.985 | 0.999 |
| Stage 5 | ResNet-18 | Grayscale | Supervised Scratch | 100% | 20 | 32 | 1e-4 | 0.962 | 0.936 | 0.940 | 0.935 | 0.940 | 0.936 | 0.989 |
| Stage 5 | ResNet-18 | Grayscale | ImageNet Fine-tune | 100% | 20 | 32 | 1e-4 | 0.996 | 0.995 | 0.995 | 0.995 | 0.995 | 0.995 | 1.000 |
| Stage 5 | ResNet-18 | Grayscale | SimCLR Fine-tune | 100% | 50+20 | 64+32 | 3e-4 + 1e-4 | 0.981 | 0.993 | 0.993 | 0.992 | 0.993 | 0.993 | 0.999 |
| Stage 6 | ResNet-18 | Grayscale | Supervised Scratch | 100% | 20 | 32 | 1e-4 | 0.962 | 0.936 | 0.940 | 0.935 | 0.940 | 0.936 | 0.989 |
| Stage 6 | ResNet-18 | Grayscale | ImageNet Fine-tune | 100% | 20 | 32 | 1e-4 | 0.996 | 0.995 | 0.995 | 0.995 | 0.995 | 0.995 | 1.000 |
| Stage 6 | ResNet-18 | Grayscale | SimCLR Fine-tune | 100% | 50+20 | 64+32 | 3e-4 + 1e-4 | 0.986 | 0.993 | 0.993 | 0.992 | 0.993 | 0.993 | 1.000 |
| Stage 6 | ResNet-18 | Grayscale | Supervised Scratch | 50% | 20 | 32 | 1e-4 | 0.838 | 0.836 | 0.841 | 0.837 | 0.841 | 0.835 | 0.939 |
| Stage 6 | ResNet-18 | Grayscale | ImageNet Fine-tune | 50% | 20 | 32 | 1e-4 | 0.995 | 0.988 | 0.987 | 0.988 | 0.987 | 0.988 | 1.000 |
| Stage 6 | ResNet-18 | Grayscale | SimCLR Fine-tune | 50% | 50+20 | 64+32 | 3e-4 + 1e-4 | 0.962 | 0.983 | 0.983 | 0.982 | 0.983 | 0.983 | 0.998 |
| Stage 6 | ResNet-18 | Grayscale | Supervised Scratch | 20% | 20 | 32 | 1e-4 | 0.767 | 0.760 | 0.757 | 0.757 | 0.757 | 0.757 | 0.869 |
| Stage 6 | ResNet-18 | Grayscale | ImageNet Fine-tune | 20% | 20 | 32 | 1e-4 | 0.972 | 0.951 | 0.953 | 0.949 | 0.953 | 0.951 | 0.994 |
| Stage 6 | ResNet-18 | Grayscale | SimCLR Fine-tune | 20% | 50+20 | 64+32 | 3e-4 + 1e-4 | 0.951 | 0.971 | 0.972 | 0.969 | 0.972 | 0.970 | 0.995 |
| Stage 6 | ResNet-18 | Grayscale | Supervised Scratch | 10% | 20 | 32 | 1e-4 | 0.627 | 0.613 | 0.642 | 0.689 | 0.642 | 0.596 | 0.677 |
| Stage 6 | ResNet-18 | Grayscale | ImageNet Fine-tune | 10% | 20 | 32 | 1e-4 | 0.942 | 0.966 | 0.963 | 0.968 | 0.963 | 0.965 | 0.993 |
| Stage 6 | ResNet-18 | Grayscale | SimCLR Fine-tune | 10% | 50+20 | 64+32 | 3e-4 + 1e-4 | 0.924 | 0.914 | 0.919 | 0.914 | 0.919 | 0.914 | 0.974 |
