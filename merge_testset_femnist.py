import os
import json

def merge_femnist_json(folder_path, output_path):
    """
    Merges multiple FEMNIST JSON files by combining users, num_samples, and user_data.

    Args:
        folder_path (str): Directory containing the JSON files.
        output_path (str): Path to save the merged JSON file.
    """

    merged_data = {"users": [], "num_samples": [], "user_data": {}}

    # Iterate over all JSON files in the folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            if os.path.normcase(os.path.realpath(file_path)) == os.path.normcase(os.path.realpath(output_path)):
                continue
            with open(file_path, "r") as f:
                data = json.load(f)

            # Merge "users" and "num_samples"


            # Merge "user_data"
            for user, user_info in data["user_data"].items():
                if user not in merged_data["user_data"]:
                    merged_data["user_data"][user] = {"x": [], "y": []}

                # Append data to existing user key
                merged_data["user_data"][user]["x"].extend(user_info["x"])
                merged_data["user_data"][user]["y"].extend(user_info["y"])

    merged_data["users"] = list(merged_data["user_data"])
    merged_data["num_samples"] = [len(merged_data["user_data"][user]["y"])
                                  for user in merged_data["users"]]

    # Save the merged data to a new JSON file
    with open(output_path, "w") as out_file:
        json.dump(merged_data, out_file)

    print(f"Merged JSON saved to {output_path}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Merge FEMNIST JSON files')
    parser.add_argument('--folder_path', required=True)
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()
    merge_femnist_json(args.folder_path, args.output_path)
