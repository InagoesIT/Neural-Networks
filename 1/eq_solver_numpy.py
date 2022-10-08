import glob
import numpy as np
from numpy.linalg import LinAlgError


def has_no_digit(start, end, first_char):
    return (end - start == 1 and first_char in ['-', '+']) or start == end


def get_coefficients(line):
    variables = np.array(['x', 'y', 'z'])
    coefficients = np.array([0, 0, 0])
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

    a = np.empty(shape=(3, 3), dtype=int)
    b = np.empty(shape=3, dtype=int)
    solution = ''
    i = 0

    for line in file_contents.split('\n'):
        if line == '':
            break
        if 'solution' in line:
            solution = line
            break
        a[i] = get_coefficients(line)
        b[i] = get_constant(line)
        i = i + 1

    return a, b, solution


def solve_system(file_name):
    a, b, file_solution = get_matrices_and_solution_from_file(file_name)

    try:
        a_inverse = np.linalg.inv(a)
    except LinAlgError:
        print('-> There are no solutions for this system of equations, because the determinant for A is 0!')
    else:
        x = np.dot(a_inverse, b)
        print('-> The solution to the system is: (', x[0], ',', x[1], ',', x[2], ')')

    if len(file_solution) > 0:
        print('-> And it should have been:', file_solution.split('=')[1])


def solve_all_systems():
    for file_name in glob.glob('*.txt'):
        print("~~~~~~ The result for the file with the name =", file_name, "~~~~~~~~~")
        solve_system(file_name)
        print()


solve_all_systems()
