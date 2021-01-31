import numpy as np

global board

board = np.array(bc.board)


def get_row(row_index, startpoint, endpoint):
    current_row = np.array([], dtype=int)

    for i in range(startpoint, endpoint + 1):
        current_row = np.append(current_row, board[row_index, i])

    return current_row


def get_column(column_index, startpoint, endpoint):
    current_column = np.array([], dtype=int)

    for i in range(startpoint, endpoint + 1):
        current_column = np.append(current_column, board[i, column_index])

    return current_column


def square_operation(startpoint, column_start, column_end, guess):
    state = False
    square = np.array([], dtype=int)

    for i in range(startpoint, (startpoint + 3)):
        square = np.append(square, get_row(i, column_start, column_end))

    if guess in square:
        return state

    # Last missing number in square 3x3
    if ((np.count_nonzero(square)) == 8) and ((45 - (np.sum(square)) == guess)):
        state = True
        return state

    square = square.reshape((3, 3))
    flag, iterator, indices = 0, 0, np.where(square == 0)

    coordinates_row = np.array(indices[0])
    coordinates_col = np.array(indices[1])

    while iterator < (np.size(indices, 1)):
        if (
            proof_lines(
                (coordinates_row[iterator] + startpoint),
                (coordinates_col[iterator] + column_start),
                guess,
            )
            == True
        ) and (flag < 1):
            state, flag, iterator = True, flag + 1, iterator + 1

        elif (
            proof_lines(
                (coordinates_row[iterator] + startpoint),
                (coordinates_col[iterator] + column_start),
                guess,
            )
            == True
        ) and (flag >= 1):
            state = False
            return state

        elif flag > 1:
            state = False
            return state

        else:
            iterator += 1
    return state


def proof_square(row_index, column_index, guess):
    # Column 1
    if column_index <= 2:
        if row_index <= 2:
            state = square_operation(0, 0, 2, guess)
        elif row_index >= 3 and row_index <= 5:
            state = square_operation(3, 0, 2, guess)
        elif row_index >= 6 and row_index <= 8:
            state = square_operation(6, 0, 2, guess)
        return state

    # Column 2
    elif column_index >= 3 and column_index <= 5:
        if row_index <= 2:
            state = square_operation(0, 3, 5, guess)
        elif row_index >= 3 and row_index <= 5:
            state = square_operation(3, 3, 5, guess)
        elif row_index >= 6 and row_index <= 8:
            state = square_operation(6, 3, 5, guess)
        return state

    # Column 3
    elif column_index >= 6 and column_index <= 8:
        if row_index <= 2:
            state = square_operation(0, 6, 8, guess)
        elif row_index >= 3 and row_index <= 5:
            state = square_operation(3, 6, 8, guess)
        elif row_index >= 6 and row_index <= 8:
            state = square_operation(6, 6, 8, guess)
        return state


def check_square(startpoint, column_start, column_end, guess):
    state = False
    square = np.array([], dtype=int)

    for i in range(startpoint, (startpoint + 3)):
        square = np.append(square, get_row(i, column_start, column_end))

    if guess in square:
        return True


def proof_lines(row_index, column_index, guess):
    if ((guess in get_row(row_index, 0, 8)) == False) and (
        (guess in get_column(column_index, 0, 8)) == False
    ):
        return True
    return False


def row_operator(row_index, column_index, guess):
    count_line_col = 0
    count_square_row = 0

    for i in range(8):
        if (board[row_index, i] == 0) and ((guess in get_column(i, 0, 8)) == False):
            count_line_col += 1
            if count_line_col > 1:
                return False
            valid_index = i


def col_operator(row_index, column_index, guess):
    count_line_row = 0
    count_square_col = 0


def proof_position(row_index, column_index, guess):
    if (proof_lines(row_index, column_index, guess) == True) and (
        proof_square(row_index, column_index, guess) == True
    ):
        return True

    elif (row_operator(row_index, column_index, guess) == True) and (
            proof_lines(row_index, column_index, guess) == True):
        return True

    elif (col_operator(row_index, column_index, guess) == True) and (
            proof_lines(row_index, column_index, guess) == True):
        return True


def solve():
    guess_all = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    if np.count_nonzero(board) == 81:
        return

    for row in range(np.size(board, 1)):
        for column in range(np.size(board, 1)):

            if board[row, column] == 0:
                for i in range(len(guess_all)):
                    guess = guess_all[i]

                    if proof_position(row, column, guess) == True:
                        board[row, column] = guess
                        # print("In Reihe %d und Spalte %d", (row, column))
                        solve()


solve()
print(board)
