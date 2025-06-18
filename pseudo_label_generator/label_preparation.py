import os
import sys


def process_txt_files(folder_path):
    # Get a list of all .txt files in the folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    # Process each text file
    for file_name in txt_files:
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'r+') as file:
            lines = file.readlines()
            file.seek(0)  # Reset file position to the beginning

            for line in lines:
                values = line.strip().split(' ')

                if len(values) > 15:
                    values = values[:15]  # Remove the last value if there are more than 16

                file.write(' '.join(values) + '\n')  # Write the modified line back to the file

            if not lines:
                zeros_line = "DontCare -1 -1 -10 0.00 0.00 0.00 0.00 -1 -1 -1 -1000 -1000 -1000 -10"
                file.write(zeros_line + '\n')  # Write the zeros line if the file is empty

            file.truncate()  # Remove any remaining content after processing


# Retrieve the folder path from the command-line argument
if len(sys.argv) < 2:
    print("Please provide the folder path as following: Argument1 = path to openpcdet labels")
    sys.exit(1)

folder_path = sys.argv[1] + "/training/label_2/"
process_txt_files(folder_path)
