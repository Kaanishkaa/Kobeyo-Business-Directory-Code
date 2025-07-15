# === Main Run ===
if __name__ == "__main__":
    user_prompt = input(
        "Enter a prompt (e.g., 'Animal care services in Los Angeles'): "
    )

    tagger = AnimalCareSkillTagger(OPENAI_API_KEY, GOOGLE_API_KEY)

    query, location = tagger.extract_query_location(user_prompt)
    if not query or not location:
        print("Could not determine service or location.")
        exit()

    print(f"Searching for: {query} in {location}")
    places = tagger.search_places(query, location)

    # Load existing entries
    existing_data = pd.DataFrame()
    existing_keys = set()
    try:
        existing_data = pd.read_csv("tagged_animal_businesses.csv")
        existing_keys = set(
            (row["Name"].strip().lower(), row["Description"].strip().lower())
            for _, row in existing_data.iterrows()
        )
    except FileNotFoundError:
        print("No existing CSV found. Creating a new one.")

    new_results = []
    for i, place in enumerate(places, 1):
        name = place.get("name", "").strip().lower()
        address = place.get("formatted_address", "").strip().lower()
        identifier = (name, address)

        if identifier in existing_keys:
            print(f"Skipping already processed: {place.get('name')} at {address}")
            continue

        print(f"Processing {i}/{len(places)}: {place.get('name')}")
        try:
            result = tagger.process_place(place)
            new_results.append(
                {
                    "Name": result.name,
                    "Description": result.description,
                    "Business Types": ", ".join(result.business_type),
                    "Potential Jobs": ", ".join(result.potential_jobs),
                    "Required Skills": ", ".join(result.required_skills),
                    "Skill IDs": ", ".join(map(str, result.skill_ids)),
                    "Skill Names": ", ".join(result.skill_names),
                }
            )
        except Exception as e:
            print(f"Error processing {place.get('name')}: {e}")

    if new_results:
        df_new = pd.DataFrame(new_results)
        if not existing_data.empty:
            df_new = pd.concat([existing_data, df_new], ignore_index=True)
        df_new.to_csv("tagged_animal_businesses.csv", index=False)
        print("✅ Appended new results to tagged_animal_businesses.csv")
    else:
        print("✅ No new businesses found. CSV unchanged.")
