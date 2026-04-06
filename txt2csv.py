import csv
import argparse
import re


def txt_to_csv(txt_file, csv_file, max_rows=None):
    """
    Convert a space-separated TXT file into a CSV file, supporting the option to specify a maximum number of rows
    """
    try:
        with open(txt_file, 'r', encoding='utf-8') as txtf, \
                open(csv_file, 'w', newline='', encoding='utf-8') as csvf:

            csv_writer = csv.writer(csvf)
            row_count = 0

            for line in txtf:
                if max_rows is not None and row_count >= max_rows:
                    break
                line = line.strip()
                fields = re.split(r'\s+', line)  
                csv_writer.writerow(fields)
                row_count += 1

        print(f"Conversion successful! A total of {row_count} rows have been converted, and the CSV file has been saved to: {csv_file}")

    except FileNotFoundError:
        print(f"Error: File {txt_file} not found")
    except Exception as e:
        print(f"An error occurred during the conversion process: {str (e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert TXT files separated by multiple spaces to CSV files')
    parser.add_argument('--txt-file', help='TXT file path',
                        default=r'')
    parser.add_argument('--csv-file', help='CSV file path',
                        default=r'')
    parser.add_argument('--max-rows', type=int, default=100)

    args = parser.parse_args()
    txt_to_csv(args.txt_file, args.csv_file, args.max_rows)
