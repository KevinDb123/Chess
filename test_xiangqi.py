from xiangqi import Board, Piece


def clear_board(b):
    for r in range(10):
        for c in range(9):
            b.grid[r][c] = None


def test_cannon_capture_with_screen():
    b = Board()
    clear_board(b)
    b.turn = 'r'
    b.set_piece(9, 4, Piece('r', 'G'))
    b.set_piece(7, 1, Piece('r', 'C'))
    b.set_piece(5, 1, Piece('r', 'S'))
    b.set_piece(0, 1, Piece('b', 'G'))
    moves = b.legal_moves_from(7, 1)
    assert (7, 1, 0, 1) in moves


def test_horse_leg_blocking():
    b = Board()
    clear_board(b)
    b.turn = 'r'
    b.set_piece(9, 4, Piece('r', 'G'))
    b.set_piece(0, 0, Piece('b', 'G'))
    b.set_piece(7, 2, Piece('r', 'H'))
    b.set_piece(7, 3, Piece('r', 'S'))
    moves = b.legal_moves_from(7, 2)
    assert (7, 2, 6, 4) not in moves


def test_flying_general_check():
    b = Board()
    clear_board(b)
    b.set_piece(9, 4, Piece('r', 'G'))
    b.set_piece(0, 4, Piece('b', 'G'))
    assert b.is_in_check('r') is True
    assert b.is_in_check('b') is True


def test_soldier_post_river_moves():
    b = Board()
    clear_board(b)
    b.turn = 'r'
    b.set_piece(9, 4, Piece('r', 'G'))
    b.set_piece(0, 0, Piece('b', 'G'))
    b.set_piece(4, 4, Piece('r', 'S'))
    moves = b.legal_moves_from(4, 4)
    assert (4, 4, 4, 3) in moves
    assert (4, 4, 4, 5) in moves
    assert (4, 4, 3, 4) in moves


def test_general_no_moves_stalemate_rule():
    b = Board()
    clear_board(b)
    b.turn = 'r'
    b.set_piece(9, 4, Piece('r', 'G'))
    b.set_piece(0, 0, Piece('b', 'G'))
    b.set_piece(8, 4, Piece('r', 'S'))
    b.set_piece(9, 5, Piece('r', 'S'))
    b.set_piece(9, 3, Piece('r', 'S'))
    moves = b.general_legal_moves('r')
    assert len(moves) == 0
    assert b.is_stalemate('r') is False


def test_stalemate_is_draw():
    b = Board()
    clear_board(b)
    b.turn = 'r'
    b.set_piece(9, 4, Piece('r', 'G'))
    b.set_piece(0, 0, Piece('b', 'G'))
    # 封死红方所有合法步：用黑兵堵住但不将军
    b.set_piece(8, 4, Piece('b', 'S'))
    b.set_piece(9, 3, Piece('b', 'S'))
    b.set_piece(9, 5, Piece('b', 'S'))
    # 校验：不在将军但无合法步，视为和棋
    assert b.is_in_check('r') is False
    assert b.has_legal_moves('r') is False
    assert b.is_stalemate('r') is True


if __name__ == '__main__':
    test_cannon_capture_with_screen()
    test_horse_leg_blocking()
    test_flying_general_check()
    test_soldier_post_river_moves()
    print('ok')
