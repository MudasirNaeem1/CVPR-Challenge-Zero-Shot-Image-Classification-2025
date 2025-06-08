# CVPR Challenge: Zero-Shot Image Classification

<div align="center">
  
![Zero-Shot Learning](https://img.shields.io/badge/Task-Zero--Shot%20Learning-blue)
![CLIP](https://img.shields.io/badge/Model-CLIP-orange)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![Transformers](https://img.shields.io/badge/Architecture-Transformers-green)
![CVPR](https://img.shields.io/badge/Conference-CVPR%202025-lightgrey)

</div>

## Overview
This repository contains the solution for the CVPR 2025 Zero-Shot Image Classification Challenge. The challenge focuses on developing models that can classify images into categories never seen during training, demonstrating true zero-shot learning capabilities.

<div align="center">
  
```mermaid
graph TD
    A[Input Image] --> B[CLIP Image Encoder]
    C[Class Names/Descriptions] --> D[CLIP Text Encoder]
    B --> E[Image Embeddings]
    D --> F[Text Embeddings]
    E --> G[Zero-Shot Matching]
    F --> G
    G --> H[Class Prediction]
    
    style A fill:#f9d5e5,stroke:#333,stroke-width:2px
    style H fill:#eeeeee,stroke:#333,stroke-width:2px
```

*Zero-Shot Classification Pipeline*
</div>

## Problem Statement
Zero-shot learning addresses the fundamental challenge of recognizing objects or categories that were not present in the training data. In real-world applications, it's impossible to train models on all possible classes. The CVPR challenge evaluates approaches that can bridge this semantic gap by leveraging visual-semantic relationships.

<div align="center">
  
![Zero-Shot Learning Challenge](https://mermaid.ink/img/pako:eNp1kMFqwzAMhl9F-NRB8wI5FDLWwWB73Nplh7q2uhjiKNhKWUvy7nPawdgt0kHS_3_6JfXgokfQ0B43sbo5b0kUt04CDT6dZIiE7Mwj5Vj8z4YRMiZqXEDNdW6H5HGDxbGFVp5PrUGVhGQ-RRIXFZFpTrWmX3Kfk-gHaV1h5lF-88t6W3G9UUhxsYW8PqF3xvpU_lR_yMqzazP6kq_LN9C35IOtRxtxOQtGmz0YipgPTqKFCOYFdB_R_VAKCeGisFtnUqRzOVuMVkMwmYbgWlS-dw6D7nxnYXwH4xEDvY6ga47DPbBrp-YN_KQcyA?type=png)

*Traditional vs. Zero-Shot Learning*
</div>

## Technologies Used
- **Python**: Primary programming language
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library for state-of-the-art models
- **scikit-learn**: For clustering and data analysis
- **CLIP (Contrastive Language-Image Pre-training)**: OpenAI's model connecting text and images
- **NumPy/Pandas**: For data handling and manipulation
- **Matplotlib/Seaborn**: For visualization

<div align="center">
  
![Tech Stack](https://mermaid.ink/img/pako:eNptks9Kw0AQxl9lmJMWtAcPvdWLgkVEUBTa20DG3Yjp_mF3oxDy7s6mSRptz_N9v29mdmd2gq1jICGdWLHy5rqx4sfhrhCp24rGhQDCc-qVQ10hE3-ETHNhFfPgjDy2TJP5Gg9Rr4GiaNL4VMYwDhXHARbJgn1w1uJqfAzZyXw5RvqNm9a_wv5uXnNPMZxwFKRPVNjrKrUo2-6yXyFSj9U6uVBYSz2r0Jxpq08C5a7GwrMcKVeCJxoiDZQyU-7sUFFbDy_N1G90I0TGaHXpj4-B73s24rBp-ZiuPzdMT9lksnQf5lR-LIX8XHI9_P9VJyXSv2p0BH67HzOQXZ9PFgqaAFJmD3KuWWRHvmSU7J1jVLIE6aPtPMimBe7wBnLnTYs2gOwrZk-yi53cR1CZNq7O0iR6k9Wo_vE3yBGzew?type=png)

*Technology Stack Overview*
</div>

## Model Architecture
The solution primarily utilizes CLIP (Contrastive Language-Image Pre-training), a neural network trained on a variety of image-text pairs. Key components include:

1. **CLIP ViT-B/32**: Vision Transformer base model from OpenAI that encodes images into embeddings
2. **Text Encoders**: Convert class names and descriptions into the same embedding space
3. **Embedding Alignment**: Methods to align visual and textual embeddings

<div align="center">
  
![CLIP Architecture](https://mermaid.ink/img/pako:eNqFkj9PwzAQxb-KdROoJOiAhAQD6oD4BwPqSnBw7SuxcGxju0VV1O9OmhapVAvMZ7_3_N7dZedKO0ksp73uxN2DtaL1LnTiKO9j49rQGNrZQIaXL0rTzsEgx-iNj9-ttkfhXw8i0kZVAEXDNx-eALgAW4jGQXRbH1pq5B4kBYf0QM3E9YkJPFPMpJ_ZQ0IVNa68PTQxpKRnYOULUKYPtvyZrPBGLCBqbdQ7sXFG1XWcZMCrEoJp4wE9_fWZh9VqjWEBVHxG7lx0ZQPbvhfdtpuIDmJUUO1jmKcDnHNKJD0HGqG5FLF04PsvdP0UxQcKvSHPfEiK0Lk2p3UOpYsOz7Oy5KZ8lPzSRyU7KhLPfI6zk4rylY1-zLTPHzgn-3eCz0_1LBl7ymPojwqzFtPsJ-rD-PUlv3b5FY8HRjvTSlpq70KLW5GfwDJ_NWe5TS3Wx-bM_AXTCP1f?type=png)

*CLIP Model Architecture*
</div>

## Methodological Approach
The implementation follows several key steps:

1. **Image Embedding Extraction**: Using CLIP to extract embeddings for all test images
2. **Text Prompt Engineering**: Creating effective prompts for zero-shot classification
3. **Clustering Analysis**: 
   - K-Means clustering of image embeddings to discover underlying patterns
   - Agglomerative clustering with cosine similarity for improved class discovery
4. **Ensemble Methods**: Combining multiple embedding approaches to improve accuracy
5. **Zero-Shot Inference**: Matching image embeddings with textual class descriptions without direct training

<div align="center">
  
![Method Pipeline](https://mermaid.ink/img/pako:eNqNksFqwzAMhl9F-FQY6QvsUMZgMNhhY4xlh7q2kiDiyNhKeQh59zlrSkhH2Y6W_u_XL8kHbq1FzniweXt3b4yoRN1kUvd9rG3d1RXvTEuapx8V801LU-QYbR_H21rtkXjnS7Fvo8xJVL1dQiQRfTwbvgThgvwGVxb6HRdDWe2CjNxSeqC64X0GQ560MBngmaKXYehP4OVfpokZbWrSBPEv0kRHF6r2Qcnq0ofm45LHc3u3i2d7Rc7qytcYTCc66HcY46XXcO1jbw1ltOK2aQxFJlczPFEoaIHKw3LQRaIZYlFYijjJXY4vTMekJ98gZ-zzLj4RB_Z1y7NKN3_LcUZSrsvKl1CZvgbpn21rl1Mu7zFE7JhT4vKIIwv7NeUbHdxUgOX5gT7jL8lGXaCSk8oVKBcOq3KN3VhVQxXW6CgqtWZ1-Q9DsvdL?type=png)

*Methodological Approach Pipeline*
</div>

## Files in the Repository
- `CVPR_Final_SOTA.ipynb`: Main implementation notebook with the complete pipeline
- `CVPR_SOTA (1).ipynb`: Alternative approach and experiments
- `CVPR_ZEROSHOT_REPORT_IJSRSET.pdf`: Detailed research report on the approach
- `ANN PRESENTATION CVPR.pptx`: Presentation explaining the methodology and results
- `ensemble_text_embeddings.pt`: Pre-computed ensemble of text embeddings
- `cluster_to_labels.pkl` & `cluster_centroids.pkl`: Outputs from clustering analysis
- `clip_train_embeddings.pkl`: Stored embeddings for efficient inference

## Results and Performance
The solution achieves state-of-the-art performance on zero-shot image classification tasks by effectively leveraging:
- Novel prompt engineering techniques
- Multi-modal embedding alignment strategies
- Cluster-based class discovery
- Ensemble methods to improve robustness

<div align="center">
  
![Performance Metrics](https://mermaid.ink/img/pako:eNp1kbFuwzAMRH-F0FQg1dgpQ4YsRYcAHbuEWrhlGVFpyKREFYH_vZSTtmgRjyLvjncS74YbS5DBeeeefnrXeWnb-nAVOSRhiZLwPrxI3KaEWmwLhMzB4MBD_yZ6i9HWQiGLx00xUWv9F9LRg9zl1JRDMYOqlJU2nJ4hNEcMwjvVMYUGunRQsRFWVNZUDqJbKRCv88lWtjR7pOp0Kgk3VjJfU4Rw5rvjFrKYzxF1dxaJozXCQCuqWRjf5RCBylVf6OOIiVZ5JDlLH5DgwBKJZQdltW5a3G4YFpUjbXH7tPpSvOiOdHOb5hbmX_ZbTOTcIl1MgSqtVt9D5-pjmZV5X7p4WXGsnq-kc3h35PlbDnMdOI_iNn9g_gVhRaA9?type=png)

*Performance Metrics Across Datasets*
</div>

## Challenges and Solutions
- **Domain Gap**: Addressed through prompt engineering and ensemble methods
- **Semantic Ambiguity**: Reduced with improved text-image alignment techniques
- **Computational Efficiency**: Optimized through embedding pre-computation and efficient inference strategies
- **Class Imbalance**: Mitigated with clustering-based approaches

<div align="center">
  
![Challenges and Solutions](https://mermaid.ink/img/pako:eNp1kk1rwzAMhv-K0KkwGuyyl0Ghs8GgO-6w0l58iGOriS0HS-mi_PfZSftx2Npb9Pp59Eo-mI1JaDDYpPbpxXTe26Z6-5AnuU9RKGb37ltKtxzF43KQDsawt8Hz6yfWloS3Ffa-8nE5o1jbfADuPPhT-D5U-6J6wXhwHkUOUAnZY5Bc5y6_BYcjPrU_HN1WE_a1LSRJ-AwcMRH7RD2Vx5IjRTTxBFLvVDjWdpNjUjVa48BGrcROZypngxpNadHoMoqYWMuqJXotRk7p6sqpQi10pqF0WnXVilIomUNk0RnVH6tnkrbFEwV_Zn-Dny6NyoGiPnMfgxqU2yX0PsqfF1b2p21Ys-HYQldS1Gbb47QsHJpYqw9x6rjsS5zvVxQXz0bLVrM0UrnfUP0BxhLDfw?type=png)

*Challenges and Solutions Overview*
</div>

## Future Work
- Exploration of more advanced transformer architectures
- Integration of additional modalities beyond text and images
- Self-supervised fine-tuning approaches to improve domain adaptation
- Improved prompt engineering strategies for better zero-shot generalization

<div align="center">
  
![Future Work](https://mermaid.ink/img/pako:eNp1krFuwzAMRH-F4BQg1S9kylKga5YgtbREURFpyKRkBEH-vZRdI0W7eRTvjneU9k5a11GBqY3S9NObUgfvyvurslBdF52QxPf-S2KtiWVYDcqI0ewp0PjtGaNliLbBPjQxLicIa1P2og7Q-QfyPUQbqBuDjxQjf4LT-ahSLWmUfDxU04wOb7Bs5mAPrCE0UHmTjopIuHREVVyuZXG1LzlQoKwX5m6kbZlDzCKbopVJrHVJ9bzJO_JLrbOvQfQxZ-QaGaFYR9TN_ZDNt_OOXOazwJk_C5-DK9VvfuU9_3XZWP6k61qKnbXmF6jl8EQRA-30lPK0n7e7qdKYP_twZ3TxrM-3AuVYiP8_aP4AJR7G3g?type=png)

*Future Research Directions*
</div>

## References
- OpenAI's CLIP (Contrastive Language-Image Pre-training)
- CVPR 2025 Zero-Shot Image Classification Challenge guidelines
- Related work in zero-shot learning and vision-language models

## Contributors
- Research team members working on the CVPR 2025 Challenge submission
