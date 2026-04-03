import sys
import csv

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <csv_file> <column_name>")
        sys.exit(1)

    csv_file, col_name = sys.argv[1], sys.argv[2]

    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        values = [float(row[col_name]) for row in reader]

    print(sum(values) / len(values))

if __name__ == "__main__":
    main()
