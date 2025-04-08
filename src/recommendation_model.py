import json
import logging
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
import faiss
import os

# Configure logging (only INFO and above will be shown)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Get the base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Define paths to CSV files with absolute paths - removed session-related files
CSV_PATHS = {
    "mentees": os.path.join(BASE_DIR, "data", "updated_mentees (2).csv"),
    "mentors": os.path.join(BASE_DIR, "data", "mentors (5).csv"),
    "users": os.path.join(BASE_DIR, "data", "indian_users_fixed.csv"),
    "mentee_tags": os.path.join(BASE_DIR, "data", "_MenteeTags (5).csv"),
    "mentor_tags": os.path.join(BASE_DIR, "data", "mentor_tags.csv"),
    "tags": os.path.join(BASE_DIR, "data", "tags_with_skills_mapping.csv"),
    "reviews": os.path.join(BASE_DIR, "data", "reviews.csv")
}

# Function to check if all required CSV files exist
def verify_csv_files():
    missing_files = []
    for key, path in CSV_PATHS.items():
        if not os.path.exists(path):
            missing_files.append(f"{key}: {path}")
    
    if missing_files:
        logging.error(f"Missing required CSV files:\n" + "\n".join(missing_files))
        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info(f"Base directory set to: {BASE_DIR}")
        return False
    return True

def get_role_id_by_user_id(user_id: str, role: str) -> str:
    try:
        if not os.path.exists(CSV_PATHS[role + "s"]):
            raise FileNotFoundError(f"Required CSV file not found: {CSV_PATHS[role + 's']}")
            
        if role == 'mentee':
            mentees_df = pd.read_csv(CSV_PATHS["mentees"])
            result = mentees_df[mentees_df["user_id"] == user_id]["id"].values
        elif role == 'mentor':
            mentors_df = pd.read_csv(CSV_PATHS["mentors"])
            result = mentors_df[mentors_df["user_id"] == user_id]["id"].values
        else:
            raise ValueError("Role must be 'mentee' or 'mentor'.")
        
        if len(result) > 0:
            return result[0]
        else:
            raise ValueError(f"No {role}_id found for user_id: {user_id}")
    except Exception as e:
        logging.error(f"Error fetching {role}_id: {e}")
        raise

def get_language_by_role_id(role_id: str, role: str) -> str:
    try:
        users_df = pd.read_csv(CSV_PATHS["users"])
        if role == 'mentee':
            mentees_df = pd.read_csv(CSV_PATHS["mentees"])
            user_id = mentees_df[mentees_df["id"] == role_id]["user_id"].values
        elif role == 'mentor':
            mentors_df = pd.read_csv(CSV_PATHS["mentors"])
            user_id = mentors_df[mentors_df["id"] == role_id]["user_id"].values
        else:
            raise ValueError("Role must be 'mentee' or 'mentor'.")
        
        if len(user_id) > 0:
            language = users_df[users_df["id"] == user_id[0]]["language"].values
            return language[0] if len(language) > 0 else None
        else:
            return None
    except Exception as e:
        logging.error(f"Error fetching language for {role} ID {role_id}: {e}")
        raise

def get_interests_by_mentee_id(mentee_id: str) -> list:
    try:
        mentee_tags_df = pd.read_csv(CSV_PATHS["mentee_tags"])
        tags_df = pd.read_csv(CSV_PATHS["tags"])
        
        # Filter to get tag IDs associated with this mentee - using proper column names
        tag_ids = mentee_tags_df[mentee_tags_df["mentee_id"] == mentee_id]["tag_id"].values
        
        # Get tag names from tag IDs
        interests = tags_df[tags_df["tag_id"].isin(tag_ids)]["tag_name"].values
        
        return list(interests)
    except Exception as e:
        logging.error(f"Error fetching mentee interests: {e}")
        raise

def get_expertise_by_mentor_id(mentor_id: str) -> list:
    try:
        mentor_tags_df = pd.read_csv(CSV_PATHS["mentor_tags"])
        tags_df = pd.read_csv(CSV_PATHS["tags"])
        
        # Filter to get tag IDs associated with this mentor - using proper column names
        tag_ids = mentor_tags_df[mentor_tags_df["mentor_id"] == mentor_id]["tag_id"].values
        
        # Get tag names from tag IDs
        expertise = tags_df[tags_df["tag_id"].isin(tag_ids)]["tag_name"].values
        
        return list(expertise)
    except Exception as e:
        logging.error(f"Error fetching mentor expertise: {e}")
        raise

def fetch_all_tags():
    try:
        tags_df = pd.read_csv(CSV_PATHS["tags"])
        return dict(zip(tags_df["tag_id"], tags_df["tag_name"]))
    except Exception as e:
        logging.error(f"Error fetching tags: {e}")
        raise

def fetch_mentor_data():
    try:
        mentors_df = pd.read_csv(CSV_PATHS["mentors"])
        mentor_tags_df = pd.read_csv(CSV_PATHS["mentor_tags"])
        tags_df = pd.read_csv(CSV_PATHS["tags"])
        users_df = pd.read_csv(CSV_PATHS["users"])
        
        result = []
        for _, mentor in mentors_df.iterrows():
            mentor_id = mentor["id"]
            user_id = mentor["user_id"]
            
            # Get tags/skills for this mentor - using proper column names
            tag_ids = mentor_tags_df[mentor_tags_df["mentor_id"] == mentor_id]["tag_id"].values
            skills = tags_df[tags_df["tag_id"].isin(tag_ids)]["tag_name"].tolist()
            
            # Get language from users table
            language = None
            user_rows = users_df[users_df["id"] == user_id]
            if not user_rows.empty:
                language = user_rows.iloc[0]["language"]
            
            result.append({
                "mentor_id": mentor_id,
                "experience_years": mentor.get("experience_years", 0) if not pd.isna(mentor.get("experience_years")) else 0,
                "rating": mentor.get("rating", 0) if not pd.isna(mentor.get("rating")) else 0,
                "number_of_mentees_mentored": mentor.get("number_of_mentees_mentored", 0) if not pd.isna(mentor.get("number_of_mentees_mentored")) else 0,
                "number_of_sessions": mentor.get("number_of_sessions", 0) if not pd.isna(mentor.get("number_of_sessions")) else 0,
                "skills": skills,
                "language": language
            })
        
        return result
    except Exception as e:
        logging.error(f"Error fetching mentor data: {e}")
        raise

def train_rf_with_real_data():
    """Train Random Forest model using real mentor data from CSV files"""
    try:
        # Get all mentors data
        mentor_list = fetch_mentor_data()
        
        if not mentor_list:
            logging.warning("No mentor data available for RF training")
            return None, None
            
        # Create features from real data
        X_train = []
        for mentor in mentor_list:
            # Extract features: rating, mentees mentored, sessions, language (dummy binary feature)
            rating = mentor.get("rating", 0)
            mentees = mentor.get("number_of_mentees_mentored", 0)
            sessions = mentor.get("number_of_sessions", 0)
            exp_years = mentor.get("experience_years", 0)
            
            X_train.append([rating, mentees, sessions, exp_years])
        
        X_train = np.array(X_train)
        
        # Create target values based on business rules
        # Higher ratings and experience = higher scores
        y_train = np.array([
            (0.4 * (mentor.get("rating", 0)/5.0)) +                    # 40% weight to rating
            (0.3 * min(1.0, mentor.get("experience_years", 0)/10)) +   # 30% weight to experience (capped at 10 years)
            (0.2 * min(1.0, mentor.get("number_of_mentees_mentored", 0)/50)) +  # 20% weight to mentees mentored
            (0.1 * min(1.0, mentor.get("number_of_sessions", 0)/100))   # 10% weight to sessions
            for mentor in mentor_list
        ])
        
        # Train RF model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Initialize scaler with training data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        
        logging.info(f"Trained RF model on {len(X_train)} real mentor profiles")
        return rf_model, scaler
        
    except Exception as e:
        logging.error(f"Error training RF model with real data: {e}")
        return None, None

def generate_recommendations(
    input_tags: list,
    role_id: str,
    current_language: str,
    exclude_current: bool = False,
    limit: int = 10
):
    """
    Two-layer recommendation system:
    1. FAISS for initial tag-based retrieval with consideration for experience
    2. RandomForest for re-ranking with language matching as a feature
    """
    try:
        # Get all tags and mentor data
        all_tags = list(set(fetch_all_tags().values()))
        mentor_list = fetch_mentor_data()
        
        # Exclude current mentor if requested
        if exclude_current:
            mentor_list = [mentor for mentor in mentor_list if mentor["mentor_id"] != role_id]
        
        if not all_tags or not mentor_list:
            logging.warning("No tags or mentors found")
            return []
        
        # Build a MultiLabelBinarizer for candidate expertise tags
        mlb = MultiLabelBinarizer(classes=all_tags)
        mentor_skills_list = [mentor["skills"] for mentor in mentor_list]
        mlb.fit(mentor_skills_list)
        
        # Encode input tags using the same binarizer
        input_encoded = mlb.transform([input_tags]).astype('float32')
        mentor_encoded = mlb.transform(mentor_skills_list).astype('float32')
        
        # Normalize vectors for cosine similarity
        def safe_normalize(matrix):
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return matrix / norms
        
        input_norm = safe_normalize(input_encoded)
        mentor_norm = safe_normalize(mentor_encoded)
        
        # Build FAISS index for similarity search
        d = mentor_norm.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(mentor_norm)
        
        # Search for similar mentors
        D, I = index.search(input_norm, len(mentor_list))
        
        logging.info(f"Input tags: {input_tags}")
        logging.info(f"Found {len(mentor_list)} potential mentors")
        logging.info(f"Top similarity scores: {list(zip(D[0][:5], I[0][:5]))}")  # Show top 5
        
        # ---- FAISS layer: Retrieve candidates based on tag similarity ----
        layer1_candidates = []
        for dist, idx in zip(D[0], I[0]):
            if dist > 0.2:  # Lower threshold to include more candidates
                mentor = mentor_list[idx]
                layer1_candidates.append(mentor)
        
        # If no candidates found after filtering, return empty list
        if not layer1_candidates:
            return []
            
        # ---- Random Forest layer: Re-rank candidates ----
        # Train a Random Forest model using real data
        rf_model, scaler = train_rf_with_real_data()
        
        if rf_model is None:
            logging.warning("Failed to train RF model with real data, using similarity scores only")
            # Fall back to similarity-based scoring
            recommendations = []
            for idx, candidate in enumerate(layer1_candidates):
                candidate_language = candidate.get("language", "")
                language_match = 1 if candidate_language and current_language and candidate_language.lower() == current_language.lower() else 0
                recommendations.append({
                    "mentor_id": candidate["mentor_id"],
                    "final_score": float(D[0][I[0] == idx][0]) + (0.1 * language_match),
                    "similarity": float(D[0][I[0] == idx][0]),
                    "language_match": language_match
                })
        else:
            # Extract features for RF model from candidates and predict scores
            recommendations = []
            for candidate in layer1_candidates:
                # Build features for RF model: rating, mentees, sessions, experience
                rf_features = np.array([
                    candidate.get("rating", 0),
                    candidate.get("number_of_mentees_mentored", 0),
                    candidate.get("number_of_sessions", 0),
                    candidate.get("experience_years", 0)
                ]).reshape(1, -1)
                
                # Normalize features using the same scaler used during training
                rf_features_scaled = scaler.transform(rf_features)
                
                # Predict score using RF model
                rf_score = rf_model.predict(rf_features_scaled)[0]
                
                # Add similarity boost to RF score (weighted 30%)
                similarity_idx = np.where(I[0] == mentor_list.index(candidate))[0]
                similarity = float(D[0][similarity_idx][0]) if len(similarity_idx) > 0 else 0
                
                # Language match bonus
                candidate_language = candidate.get("language", "")
                language_match = 1 if candidate_language and current_language and candidate_language.lower() == current_language.lower() else 0
                
                # Calculate final score (RF score + similarity boost + language bonus)
                final_score = (0.5 * rf_score) + (0.4 * similarity) + (0.1 * language_match)
                
                recommendations.append({
                    "mentor_id": candidate["mentor_id"],
                    "final_score": float(final_score),
                    "rf_score": float(rf_score),
                    "similarity": similarity,
                    "language_match": language_match
                })
        
        # Sort by final score and limit results
        recommendations.sort(key=lambda x: x["final_score"], reverse=True)
        return recommendations[:limit]
        
    except Exception as e:
        logging.error(f"Error generating recommendations: {e}")
        return []

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Invalid usage"}))
        sys.exit(1)
    
    user_type = sys.argv[1]  # 'mentee' or 'mentor'
    user_id = sys.argv[2]
    
    try:
        logging.info(f"User ID: {user_id}, Type: {user_type}")
        
        # Verify CSV files exist
        if not verify_csv_files():
            print(json.dumps({"error": "Required CSV files missing. Check logs for details."}))
            sys.exit(1)

        if user_type == 'mentee':
            mentee_id = get_role_id_by_user_id(user_id, 'mentee')
            mentee_language = get_language_by_role_id(mentee_id, 'mentee')
            interests = get_interests_by_mentee_id(mentee_id)
            logging.info(f"Mentee interests: {interests}")
            recommendations = generate_recommendations(interests, mentee_id, mentee_language)
            print(json.dumps({"recommendations": recommendations}, indent=4))
        elif user_type == 'mentor':
            mentor_id = get_role_id_by_user_id(user_id, 'mentor')
            mentor_language = get_language_by_role_id(mentor_id, 'mentor')
            expertise = get_expertise_by_mentor_id(mentor_id)
            recommendations = generate_recommendations(expertise, mentor_id, mentor_language, exclude_current=True)
            print(json.dumps({"recommendations": recommendations}, indent=4))
        else:
            raise ValueError("Invalid user type. Use 'mentee' or 'mentor'.")
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        print(json.dumps({"error": str(e)}))
        sys.exit(1)