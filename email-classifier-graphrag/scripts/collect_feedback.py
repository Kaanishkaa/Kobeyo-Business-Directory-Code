import json
import requests

# These are the corrections based on actual business types
corrections = {
    # Hair salons - typically neither HR nor Sales
    "ariannahair.inc@gmail.com": None,  # Hair salon
    "info@thehairparloron8th.com": None,  # Hair salon
    "hairbymina68@gmail.com": None,  # Hair salon
    "info@lasandwichbar.com": None,  # Restaurant
    "info@colesfrenchdip.com": None,  # Restaurant
    
    # Restaurants/Food - Sales related
    "info@bangbang.la": None,  # Restaurant
    "dtla@bottegalouie.com": "Sales",  # Restaurant location
    "sales@bottegalouie.com": "Sales",  # Clear sales
    "info@bottegalouie.com": None,  # Restaurant info
    "weho@bottegalouie.com": "Sales",  # Restaurant location
    "customerservice@philippes.com": "Sales",  # Customer service

    "catering@mikesdelionline.com": "Sales",  # Catering service
    
    # Cleaning services - Sales related
    "info@maidforla.com": None,  # Cleaning service
    "sales@maidthis.com": "Sales",  # Clear sales
    "contact@lacleaningco.com": "Sales",  # Cleaning service
    
    # Support emails - typically Sales/Customer Service
    "support@logunova.com": "Sales",
    "support@sparkleshinela.com": "Sales",
    "support@schedulista.com": "Sales",
    "support@marchingmaids.com": "Sales",
    
    # Business development
    "business@arcagenolus.com": "Sales",  # Business inquiries
    
    # Only clear HR case
    "careers@fashionnova.com": "HR",  # Clear careers page
}

def submit_feedback(email, correct_category):
    """Submit feedback for an email"""
    response = requests.post("http://localhost:8000/feedback", 
        json={
            "email": email,
            "correct_category": correct_category
        })
    return response.json()

def main():
    print("📝 Collecting feedback for misclassified emails...")
    
    feedback_count = 0
    for email, correct_category in corrections.items():
        if correct_category:  # Only submit if we have a correction
            try:
                result = submit_feedback(email, correct_category)
                print(f"✅ {email} -> {correct_category}")
                feedback_count += 1
            except Exception as e:
                print(f"❌ Failed to submit {email}: {e}")
    
    print(f"\n✅ Submitted {feedback_count} corrections")
    
    # Check learning system status
    from continuous_learning import ContinuousLearningSystem
    learning_system = ContinuousLearningSystem()
    status = learning_system.get_learning_status()
    
    print(f"\n📊 Learning System Status:")
    print(f"   Total feedback: {status['current_feedback_count']}")
    print(f"   Progress to retrain: {status['progress_to_retrain']}")
    
    if status['ready_to_retrain']:
        print("\n🚀 Ready to retrain! Run: python scripts/continuous_learning.py retrain")

if __name__ == "__main__":
    main()