# PetVision AI — Cat vs Dog Classifier

**PetVision AI** is a deep learning project that demonstrates the full lifecycle of an AI solution: from baseline model to state-of-the-art transfer learning, and finally to deployment in a user-friendly web app.

---

## Project Overview

The goal of PetVision AI is simple: **classify images of cats and dogs with high accuracy.**
But the project goes beyond that — it’s a demonstration of **how to structure, experiment, and deploy AI models**.

---

## Workflow

1. **Baseline CNN**

   * Implemented a simple Convolutional Neural Network trained from scratch.
   * Served as a reference model with moderate accuracy.

2. **Transfer Learning with ResNet50**

   * Leveraged ImageNet-pretrained ResNet50.
   * Initially froze the convolutional base and trained only the classification head.

3. **Progressive Fine-Tuning**

   * Gradually unfroze deeper layers:

     * Head only
     * Conv4 block
     * Conv4 + Conv5 blocks
   * Achieved **99%+ accuracy, precision, recall, and F1-score**.
   * Confusion matrix showed near-perfect classification.

4. **Model Evaluation**

   * Metrics: Accuracy, Precision, Recall, F1-score.
   * Visualizations: Training curves, classification reports, confusion matrices.

5. **Deployment**

   * Built a **Streamlit web app**.
   * Supports multiple image formats (JPEG, PNG, AVIF, WebP).
   * Upload any pet image and get instant predictions.

---

## Results

* **Accuracy:** > 99%
* **Precision / Recall / F1-score:** 0.99 (cats and dogs)
* **Confusion Matrix:** Almost flawless predictions.

---

## Tech Stack

* **Python** (TensorFlow / Keras, NumPy, Pandas, Matplotlib, Seaborn)
* **Transfer Learning:** ResNet50
* **Deployment:** Streamlit
* **Data Source:** TensorFlow Datasets — *Cats vs Dogs*

---

## Key Takeaways

PetVision AI demonstrates a **structured AI workflow**:

* Start with a baseline
* Iterate with transfer learning
* Fine-tune with evidence
* Deploy into a real-world app

---

## Next Steps

* Extend classification to more pet species.
* Integrate explainability (Grad-CAM visualizations).
* Package app as a Docker container for easier deployment.


