import os
import argparse

# Function to extract the score from a line
def get_score(line):
    return float(line.split()[-1])

def main(input_dir, output_dir):
    # List all TXT files in the input directory
    txt_files = sorted([file for file in os.listdir(input_dir) if file.endswith('.txt')])

    for txt_file in txt_files:
        with open(os.path.join(input_dir, txt_file), 'r') as file:
            lines = file.readlines()

        # Sort the lines based on the score in descending order
        sorted_lines = sorted(lines, key=get_score, reverse=True)

        # Separate "Car" and "DontCare" lines
        car_lines = [line for line in sorted_lines if line.startswith('Car')]
        dontcare_lines = [line for line in sorted_lines if line.startswith('DontCare')]

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Write the sorted lines to an output file with the same name
        output_file = os.path.join(output_dir, txt_file)
        with open(output_file, 'w') as file:
            for line in car_lines:
                file.write(line)
            for line in dontcare_lines:
                file.write(line)

        print(f"Sorted lines have been written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing input TXT files")
    parser.add_argument("output_dir", help="Directory to store sorted output TXT files")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
