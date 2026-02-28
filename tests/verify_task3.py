import nlp_engine as nlp
import os

def final_verify():
    print("--- Task 3 Final Verification ---")
    
    # Load resources
    try:
        nlp.load_resources()
    except Exception as e:
        print(f"Resource loading failed: {e}")
        return

    # Test JD
    jd = "We seek a Data Scientist with python and sql expertise to build machine learning models."
    
    # Test Resume Text
    resume_text = "Experienced Data Scientist. Skills: Python, SQL, Machine Learning, Deep Learning. Role: Data Scientist."
    
    # 1. Role Prediction
    role, conf = nlp.get_ml_prediction(resume_text)
    print(f"Predicted Role: {role} (Confidence: {conf}%)")
    
    # 2. Skill Extraction
    matcher = nlp.get_skills_matcher(['python', 'sql', 'machine learning'])
    skills = nlp.extract_skills(resume_text, matcher)
    print(f"Extracted Skills: {skills}")
    
    # 3. Match Scoring
    score, semantic, skill_pct = nlp.calculate_match_score(jd, resume_text, ['Python', 'Sql', 'Machine Learning'], skills)
    print(f"Final ML Score: {score}%")
    print(f"Semantic Sim: {semantic}%")
    print(f"Skill Match: {skill_pct}%")
    
    # 4. Gap Check
    gap = nlp.identify_gap(['Python', 'Java', 'Sql'], skills)
    print(f"Identified Gap: {gap}")

if __name__ == "__main__":
    final_verify()
