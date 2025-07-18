from promptParser import PromptParser

def main():
    print("Enter your prompt (e.g., 'animal services in San Francisco'):")
    user_input = input("> ").strip()

    if not user_input:
        print("❌ No prompt entered. Exiting.")
        return

    try:
        parser = PromptParser()
        parser.process_files(user_input)
        print("✅ Processing complete. Check the 'Outputs' folder on Google Drive for 'results.csv'.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
