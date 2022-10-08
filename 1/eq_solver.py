import glob


def has_no_digit(start, end, first_char):
    return (end - start == 1 and first_char in ['-', '+']) or start == end


def get_coefficients(line):
    variables = ['x', 'y', 'z']
    coefficients = [0, 0, 0]
    start = 0

    for i in range(3):
        end = line.find(variables[i])
        if has_no_digit(start, end, line[start]):
            coefficients[i] = 1
            if line[start] == '-':
                coefficients[i] = -1
        elif end != -1:
            coefficients[i] = (int(line[start:end]))
        if end != -1:
            start = end + 1

    return coefficients


def get_constant(line):
    return int(line.split('=')[1])


def get_matrices_and_solution_from_file(file_name):
    with open(file_name) as file:
        file_contents = file.read()

    a = []
    b = []
    solution = ''

    for line in file_contents.split('\n'):
        if line == '':
            break
        if 'solution' in line:
            solution = line
            break
        a.append(get_coefficients(line))
        b.append(get_constant(line))

    return a, b, solution


def get_determinant(a, i_to_delete, j_to_delete):
    determinant = []

    for i in range(3):
        row = []
        for j in range(3):
            if i != i_to_delete and j != j_to_delete:
                row.append(a[i][j])
        if len(row) > 0:
            determinant.append(row)

    return determinant


def get_minors(a):
    minors = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for i in range(3):
        for j in range(3):
            determinant = get_determinant(a, i, j)
            minors[i][j] = determinant[0][0] * determinant[1][1] - determinant[1][0] * determinant[0][1]

    return minors


def get_transpose(a):
    transpose = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for i in range(3):
        for j in range(3):
            transpose[j][i] = a[i][j]

    return transpose


def get_cofactor_matrix(minors):
    minors[0][1] = minors[0][1] * -1
    minors[1][0] = minors[1][0] * -1
    minors[1][2] = minors[1][2] * -1
    minors[2][1] = minors[2][1] * -1
    return minors


def get_inverse(a):
    minors = get_minors(a)
    a_determinant = a[0][0] * minors[0][0] - a[0][1] * minors[0][1] + a[0][2] * minors[0][2]
    if a_determinant == 0:
        return None
    a_inverse = get_transpose(get_cofactor_matrix(minors))
    for i in range(3):
        for j in range(3):
            a_inverse[i][j] = a_inverse[i][j] / a_determinant

    return a_inverse


def multiply_matrices(a, b):
    result = []
    for i in range(3):
        row_sum = 0
        for j in range(3):
            row_sum += a[i][j] * b[j]
        result.append([row_sum])
    return result


def solve_system(file_name):
    a, b, file_solution = get_matrices_and_solution_from_file(file_name)
    a_inverse = get_inverse(a)
    if a_inverse is None:
        print('-> There are no solutions for this system of equations, because the determinant for A is 0!')
    else:
        x = multiply_matrices(a_inverse, b)
        print('-> The solution to the system is: (', x[0][0], ',', x[1][0], ',', x[2][0], ')')

    if len(file_solution) > 0:
        print('-> And it should have been:', file_solution.split('=')[1])


def solve_all_systems():
    for file_name in glob.glob('*.txt'):
        print("~~~~~~ The result for the file with the name =", file_name, "~~~~~~~~~")
        solve_system(file_name)
        print()


solve_all_systems()
