# Learning Style: Myth or Reality? 🎓
## A Multi-Task Predictive Audit of Student Performance

This project conducts a multi-task predictive audit of student performance to investigate the empirical validity of learning styles. Using a dataset of 14,003 records, we developed a machine learning pipeline to predict Exam Scores and Final Grades while employing SHAP interpretability, ablation experiments, and K-Means clustering to determine if LearningStyle labels provide genuine predictive power or merely recapitulate behavioral data.

---

### 🏛️ Institutional Context
* **Institution:** Polytechnic Institute of Santarém
* **Course:** Master’s in Applied Informatics
* **Discipline:** Applied Artificial Intelligence
* **Semester:** 2nd Semester (2025/2026)

### 🚀 Project Scope
The project addresses four key research questions:
1. **RQ1:** Can ML predict scores and grades with meaningful accuracy?
2. **RQ2:** Does the `LearningStyle` label add statistically meaningful predictive power?
3. **RQ3:** Are labels independently derived or just a recapitulation of behavior?
4. **RQ4:** What are the realistic harms of using such models in schools?

### 📁 Repository Structure
The repository is organized according to the official technical requirements:

* 📂 `data/`: Contains the `student_performance.csv` (Kaggle Student Performance Dataset).
* 📂 `notebooks/`: Includes `analysis.ipynb` with the full pipeline (EDA to evaluation).
* 📂 `models/`: Saved `joblib` files for the best-performing models.
* 📂 `figures/`: All plots (SHAP, PCA, Ablation, etc.) saved at 150 dpi.
* 📄 `app.py`: Streamlit application for non-technical users.
* 📄 `article.pdf`: Final conference-style article (IEEE/ACM format).
* 📄 `requirements.txt`: List of all dependencies with pinned versions.

### 🛠️ Setup & Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/felipesabbado/2026-IPS-aai-project.git](https://github.com/felipesabbado/2026-IPS-aai-project.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

### 🧠 Core Methodology
This project implements:
* **Predictive Modeling:** Ridge Regression, Random Forest, and XGBoost.
* **Interpretability:** SHAP TreeExplainer, Permutation Importance, and Ablation Study.
* **Validation:** K-Means clustering and Adjusted Rand Index (ARI) to audit labels.
* **Fairness:** Performance gap analysis by gender and demographic subgroups.

---
> **Disclaimer:** This model is a decision-support tool and should not be used to make final academic decisions without human review.