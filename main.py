import requests
import pandas as pd
import ssl
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI
from fastapi.responses import Response
import uvicorn


# Suppress SSL errors in Google Colab
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize FastAPI app
app = FastAPI()

# API Endpoints
quiz_api_url = "https://jsonkeeper.com/b/LLQT"
submission_api_url = "https://api.jsonserve.com/rJvd7g"
historical_api_url = "https://api.jsonserve.com/XgAgFJ"


# Function to fetch data from an API and handle errors
def fetch_api_data(url):
    try:
        response = requests.get(url, verify=False, timeout=10)  # Disable SSL verification
        response.raise_for_status()  # Raise error if request fails
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None


# Fetching Data
quiz_data = fetch_api_data(quiz_api_url)["quiz"]
submission_data = fetch_api_data(submission_api_url)
historical_data = fetch_api_data(historical_api_url)

# Convert Quiz Questions to DataFrame
quiz_questions = quiz_data["questions"] if quiz_data else []
quiz_df = pd.DataFrame(quiz_questions)

# Convert Submission Data to DataFrame
submission_df = pd.DataFrame([submission_data]) if submission_data else pd.DataFrame()

# Convert Historical Quiz Data to DataFrame
historical_df = pd.DataFrame(historical_data) if historical_data else pd.DataFrame()

# Prepare data for NEET Rank Prediction Model
if "score" in historical_df.columns and "accuracy" in historical_df.columns and "speed" in historical_df.columns:
    historical_df["accuracy"] = historical_df["accuracy"].str.replace("%", "").astype(float)
    historical_df["rank"] = historical_df["rank_text"].str.extract(r"#?(\d+)").astype(float)
    historical_df.dropna(subset=["rank"], inplace=True)

    # Feature Engineering: Adding Difficulty Performance
    difficulty_levels = ["Easy", "Medium", "Hard"]
    for level in difficulty_levels:
        historical_df[level] = 0  # Default 0 if difficulty level not found

    for index, row in historical_df.iterrows():
        user_responses = row["response_map"] if "response_map" in row else {}
        for q in quiz_questions:
            if q["id"] in user_responses:
                difficulty = q.get("difficulty_level", "Medium")  # Assume Medium if not specified
                if difficulty in difficulty_levels:
                    historical_df.at[index, difficulty] += 1

    # Define feature set
    feature_columns = ["score", "accuracy", "speed", "Easy", "Medium", "Hard"]
    X = historical_df[feature_columns]
    y = historical_df["rank"]

    # Normalize Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train a Tuned RandomForestRegressor Model
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Cross-Validation Score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    print(f"Cross-Validation MAE: {abs(cv_scores.mean())}")


@app.get("/")
def root():
    return {"message": "Welcome to Neet rank Prediction API"}

@app.get("/favicon.ico")
def favicon():
    return Response(content=b"", media_type="image/x-icon")


# API Endpoint for Quiz Insights
@app.get("/quiz-insights")
def get_quiz_insights():
    if "score" in submission_df.columns and "accuracy" in submission_df.columns:
        # Identify weak topics and not attempted topics
        weak_topics = []
        not_attempted_topics = []
        topic_accuracy = {}
        response_map = submission_data.get("response_map", {})
        response_map = {int(k): v for k, v in response_map.items()}

        for index, row in quiz_df.iterrows():
            question_id = row["id"]
            correct_option = next((opt["id"] for opt in row["options"] if opt["is_correct"]), None)
            if question_id in response_map:
                selected_option = response_map[question_id]
                if row["topic"] not in topic_accuracy:
                    topic_accuracy[row["topic"]] = {"correct": 0, "total": 0}
                topic_accuracy[row["topic"]]["total"] += 1
                if selected_option == correct_option:
                    topic_accuracy[row["topic"]]["correct"] += 1
            else:
                not_attempted_topics.append(row["topic"])

        topic_accuracy_percent = {topic: (data["correct"] / data["total"]) * 100 for topic, data in
                                  topic_accuracy.items()}
        weak_topics = [topic for topic, acc in topic_accuracy_percent.items() if acc < 60]

        # Generate recommendations
        recommendations = []
        if weak_topics:
            recommendations.append(f"Focus more on these weak topics: {', '.join(weak_topics)}")
        else:
            recommendations.append("Great job! Keep practicing to maintain accuracy.")

        if not_attempted_topics:
            recommendations.append(f"Try attempting these topics: {', '.join(set(not_attempted_topics))}")

        # Time Management Recommendations
        speed = float(submission_df["speed"].values[0]) if "speed" in submission_df.columns else None
        if speed:
            if speed > 100:
                recommendations.append("Slow down while answering to improve accuracy.")
            elif speed < 90:
                recommendations.append("Speed up your responses to improve overall test performance.")

        insights = {
            "User Score": int(submission_df["score"].values[0]),
            "Accuracy": str(submission_df["accuracy"].values[0]),
            "Weak Topics": list(set(weak_topics)),
            "Not Attempted Topics": list(set(not_attempted_topics)),
            "Recommendations": recommendations
        }
        return insights
    return {"error": "No submission data available"}


# API Endpoint for NEET Rank Prediction
@app.get("/predict-rank")
def predict_neet_rank():
    if "score" in submission_df.columns and "accuracy" in submission_df.columns and "speed" in submission_df.columns:
        student_score = int(submission_df["score"].values[0])
        student_accuracy = float(submission_df["accuracy"].values[0].replace("%", ""))
        student_speed = float(submission_df["speed"].values[0])

        # Extract difficulty level performance from latest quiz
        difficulty_count = {"Easy": 0, "Medium": 0, "Hard": 0}
        response_map = submission_data.get("response_map", {})
        for q in quiz_questions:
            if q["id"] in response_map:
                difficulty = q.get("difficulty_level", "Medium")
                if difficulty in difficulty_count:
                    difficulty_count[difficulty] += 1

        # Prepare input for prediction
        student_features = [[student_score, student_accuracy, student_speed,
                             difficulty_count["Easy"], difficulty_count["Medium"], difficulty_count["Hard"]]]
        student_features_scaled = scaler.transform(student_features)
        predicted_rank = int(model.predict(student_features_scaled)[0])

        return {"Predicted NEET Rank": predicted_rank}
    return {"error": "No submission data available"}
