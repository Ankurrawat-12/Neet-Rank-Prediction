# NEET Quiz Performance Analysis & Rank Prediction

## 📌 Project Overview

This project analyzes **NEET quiz performance** and provides:

- **Personalized recommendations** based on weak topics.
- **NEET rank prediction** using a machine learning model.
- **API endpoints** for accessing insights and rank predictions.

## 🚀 Features

✅ **Performance Analysis** – Identifies weak topics & trends.  
✅ **NEET Rank Prediction** – Uses an optimized ML model.  
✅ **FastAPI Integration** – Exposes endpoints for easy access.  
✅ **Feature Engineering** – Includes difficulty level, speed & accuracy.  

## 🏗️ Tech Stack

- **Python**
- **FastAPI** (Backend API)
- **Scikit-Learn** (Machine Learning)
- **Pandas** (Data Handling)
- **Requests** (API Fetching)
- **Uvicorn** (ASGI Server)

## 📂 Installation & Setup

1️⃣ **Clone the Repository:**

```sh
git clone https://github.com/Ankurrawat-12/Neet-Rank-Prediction.git
cd neet-quiz-analysis
```

2️⃣ **Install Dependencies:**

```sh
pip install -r requirements.txt
```

3️⃣ **Run the API Server:**

```sh
uvicorn main:app --reload
```

## 📊 API Endpoints

### 1️⃣ **Quiz Insights**

🔗 Endpoint: `GET /quiz-insights`

- Returns performance analysis & study recommendations.

### 2️⃣ **NEET Rank Prediction**

🔗 Endpoint: `GET /predict-rank`

- Predicts the NEET rank based on quiz performance.

## 📷 Screenshots

![img.png](img.png)

![img_1.png](img-1.png)

![img_2.png](img-2.png)

![img_3.png](img-3.png)

![img_4.png](img-4.png)

![img_5.png](img-5.png)



[//]: # (## 🎥 Demo Video)

[//]: # ()
[//]: # (- A **2-5 min walkthrough** of how the script & API work.)

[//]: # (- Explain input data, predictions & insights.)

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or PRs.

## 🛠️ Future Enhancements

🚀 **Better Rank Estimation** – Train on real NEET performance data.  
🚀 **AI-Based Recommendations** – Generate study plans dynamically.  
🚀 **UI Dashboard** – Build a frontend to visualize insights.  

## 📜 License

This project is **open-source** under the MIT License.
