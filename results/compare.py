def read_solutions(filename):
    """
    Reads a CSV file containing multiple solution blocks.
    We assume that:
      - Each solution block consists of one or more rows of numbers (separated by commas or whitespace).
      - A blank line (or a line starting with "Solution") indicates the beginning of a new solution block.
    Returns a list of sets, each containing the numbers for one solution.
    """
    solutions = []
    current_solution = set()
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # A blank line or a line starting with "Solution" signals a new block.
            if not line or line.lower().startswith("solution"):
                if current_solution:
                    solutions.append(current_solution)
                    current_solution = set()
                continue
            # Split the line by comma if present, otherwise by whitespace.
            if "," in line:
                parts = [x.strip() for x in line.split(",") if x.strip()]
            else:
                parts = line.split()
            try:
                numbers = [int(x) for x in parts]
                current_solution.update(numbers)
            except ValueError:
                # If the line doesn't contain valid integers, skip it.
                continue
    # Append the last solution block if it has any numbers.
    if current_solution:
        solutions.append(current_solution)
    return solutions

def check_solutions(solutions, expected_count=175):
    """
    Checks that each solution has exactly expected_count unique nodes.
    Prints a message for each solution.
    """
    for i, sol in enumerate(solutions, start=1):
        if len(sol) != expected_count:
            print(f"Warning: Solution {i} has {len(sol)} unique nodes (expected {expected_count}).")
        else:
            print(f"Solution {i} has exactly {expected_count} unique nodes.")

def compare_solutions(solutions):
    """
    Compares each pair of solutions.
    For each pair, prints:
      - Numbers in solution i that are not in solution j.
      - Numbers in solution j that are not in solution i.
    """
    num_solutions = len(solutions)
    for i in range(num_solutions):
        for j in range(i + 1, num_solutions):
            diff_i_j = solutions[i] - solutions[j]
            diff_j_i = solutions[j] - solutions[i]
            print(f"Difference between Solution {i+1} and Solution {j+1}:")
            print(f"  In Solution {i+1} but not in Solution {j+1}: {sorted(diff_i_j)}")
            print(f"  In Solution {j+1} but not in Solution {i+1}: {sorted(diff_j_i)}\n")

def main():
    filename = "four175.csv"  # Update with your CSV filename
    solutions = read_solutions(filename)
    print(f"Found {len(solutions)} solutions in the file.\n")
    
    # Check each solution for exactly 175 unique nodes.
    check_solutions(solutions, expected_count=175)
    print("")
    
    # Compare the solutions pairwise.
    compare_solutions(solutions)

if __name__ == "__main__":
    main()
