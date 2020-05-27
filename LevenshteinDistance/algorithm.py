import numpy as np
import tabulate as tb


#  алгоритм динамического программирования
def edit_distance_func(word_1, word_2):
    n = len(word_1) + 1
    m = len(word_2) + 1

    # инициализируем D
    D = np.zeros(shape=(n, m), dtype=np.int)
    D[:, 0] = range(n)
    D[0, :] = range(m)

    # backtrack matrix
    B = np.zeros(shape=(n, m), dtype=[("del", 'b'), ("sub", 'b'), ("ins", 'b')])
    B[1:, 0] = (1, 0, 0)
    B[0, 1:] = (0, 0, 1)

    for i, l_1 in enumerate(word_1, start=1):
        for j, l_2 in enumerate(word_2, start=1):
            deletion = D[i-1,j] + 1
            insertion = D[i, j-1] + 1
            substitution = D[i-1,j-1] + (0 if l_1 == l_2 else 2)
            mo = np.min([deletion, insertion, substitution])
            B[i,j] = (deletion == mo, substitution == mo, insertion == mo)
            D[i,j] = mo
    return D, B


# backtrace function
def backtrace(matrix_b):
    i, j = matrix_b.shape[0]-1, matrix_b.shape[1]-1
    backtrace_ids = [(i, j)]

    while (i, j) != (0, 0):
        if matrix_b[i, j][1]:
            i, j = i-1, j-1
        elif matrix_b[i, j][0]:
            i, j = i-1, j
        elif matrix_b[i, j][2]:
            i, j = i, j-1
        backtrace_ids.append((i, j))

    return backtrace_ids


# alignment function
def align(word_1, word_2, bt):

    aligned_word_1 = []
    aligned_word_2 = []
    operations = []

    backtrace = bt[::-1]

    for k in range(len(backtrace) - 1):
        i_0, j_0 = backtrace[k]
        i_1, j_1 = backtrace[k+1]

        if i_1 > i_0 and j_1 > j_0:
            if word_1[i_0] == word_2[j_0]:
                w_1_letter = word_1[i_0]
                w_2_letter = word_2[j_0]
                op = " "
            else:  # операция замены
                w_1_letter = word_1[i_0]
                w_2_letter = word_2[j_0]
                op = "s"
        elif i_0 == i_1:  # операция вставки
                w_1_letter = " "
                w_2_letter = word_2[j_0]
                op = "i"
        else:  # операция удаления
                w_1_letter = word_1[i_0]
                w_2_letter = " "
                op = "d"

        aligned_word_1.append(w_1_letter)
        aligned_word_2.append(w_2_letter)
        operations.append(op)

    return aligned_word_1, aligned_word_2, operations


# creating table
def create_table(word_1, word_2, D, B, bt):
    w_1 = word_1.upper()
    w_2 = word_2.upper()

    w_1 = "#" + w_1
    w_2 = "#" + w_2

    table = []
    table.append([""] + list(w_2))

    max_n_len = len(str(np.max(D)))
    print(max_n_len)

    for i, l_1 in enumerate(w_1):
        row = [l_1]
        for j, l_2 in enumerate(w_2):
            v, d, h = B[i, j]
            direction = ("⇑" if v else "") +\
                ("⇖" if d else "") +\
                ("⇐" if h else "")

            dist = str(D[i, j])

            cell_str = "{direction} {star}{dist}{star}".format(
                                         direction=direction,
                                         star=" *"[((i, j) in bt)],
                                         dist=dist)
            row.append(cell_str)
        table.append(row)

    return table


# testing
word_1 = "привет"
word_2 = "право"

D, B = edit_distance_func(word_1, word_2)
bt = backtrace(B)
print(D)


edit_distance_table = create_table(word_1, word_2, D, B, bt)
alignment_table = align(word_1, word_2, bt)

print("minimum edit distance + backtrace:")
print(tb.tabulate(edit_distance_table, stralign="right", tablefmt="orgtbl"))

print("\nresult:")
print("/* i = insertion, s = substitution, d = deletion */")
print(tb.tabulate(alignment_table, tablefmt="orgtbl"))