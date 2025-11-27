import tkinter as tk
from tkinter import filedialog
from dataclasses import dataclass
import math
try:
    import winsound
except Exception:
    winsound = None
import os
import pickle
import threading
import time
import random
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None

BOARD_ROWS = 10
BOARD_COLS = 9


@dataclass
class Piece:
    color: str
    kind: str


class Board:
    def __init__(self):
        self.grid = [[None for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        self.turn = 'r'
        self._setup()
        self.history_keys = [self.position_key()]

    def _setup(self):
        order = ['R', 'H', 'E', 'A', 'G', 'A', 'E', 'H', 'R']
        for c in range(BOARD_COLS):
            self.grid[0][c] = Piece('b', order[c])
            self.grid[9][c] = Piece('r', order[c])
        for c in [1, 7]:
            self.grid[2][c] = Piece('b', 'C')
            self.grid[7][c] = Piece('r', 'C')
        for c in [0, 2, 4, 6, 8]:
            self.grid[3][c] = Piece('b', 'S')
            self.grid[6][c] = Piece('r', 'S')

    def in_bounds(self, r, c):
        return 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS

    def piece_at(self, r, c):
        if not self.in_bounds(r, c):
            return None
        return self.grid[r][c]

    def set_piece(self, r, c, piece):
        self.grid[r][c] = piece

    def position_key(self):
        items = []
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                p = self.grid[r][c]
                if p:
                    items.append((r, c, p.color, p.kind))
        items.sort()
        return (self.turn, tuple(items))

    def is_threefold_repetition(self):
        if not self.history_keys:
            return False
        current = self.history_keys[-1]
        return sum(1 for k in self.history_keys if k == current) >= 3

    def square_attacked_by(self, attacker_color, r, c):
        for rr in range(BOARD_ROWS):
            for cc in range(BOARD_COLS):
                p = self.grid[rr][cc]
                if p and p.color == attacker_color:
                    for _, _, tr, tc in self.generate_moves_from(rr, cc):
                        if tr == r and tc == c:
                            return True
        return False

    def min_attacker_value(self, attacker_color, r, c):
        vals = {'R': 5, 'H': 3, 'E': 2, 'A': 2, 'G': 100, 'C': 4, 'S': 1}
        m = None
        for rr in range(BOARD_ROWS):
            for cc in range(BOARD_COLS):
                p = self.grid[rr][cc]
                if p and p.color == attacker_color:
                    for _, _, tr, tc in self.generate_moves_from(rr, cc):
                        if tr == r and tc == c:
                            v = vals.get(p.kind, 0)
                            if m is None or v < m:
                                m = v
        return m

    def find_general(self, color):
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                p = self.grid[r][c]
                if p and p.color == color and p.kind == 'G':
                    return r, c
        return None

    def palace_contains(self, color, r, c):
        if color == 'b':
            return 0 <= r <= 2 and 3 <= c <= 5
        return 7 <= r <= 9 and 3 <= c <= 5

    def crossed_river(self, color, r):
        if color == 'b':
            return r >= 5
        return r <= 4

    def path_clear(self, r1, c1, r2, c2):
        if r1 == r2:
            step = 1 if c2 > c1 else -1
            for cc in range(c1 + step, c2, step):
                if self.grid[r1][cc] is not None:
                    return False
            return True
        if c1 == c2:
            step = 1 if r2 > r1 else -1
            for rr in range(r1 + step, r2, step):
                if self.grid[rr][c1] is not None:
                    return False
            return True
        return False

    def count_between(self, r1, c1, r2, c2):
        if r1 == r2:
            step = 1 if c2 > c1 else -1
            cnt = 0
            for cc in range(c1 + step, c2, step):
                if self.grid[r1][cc] is not None:
                    cnt += 1
            return cnt
        if c1 == c2:
            step = 1 if r2 > r1 else -1
            cnt = 0
            for rr in range(r1 + step, r2, step):
                if self.grid[rr][c1] is not None:
                    cnt += 1
            return cnt
        return 10

    def generate_moves_from(self, r, c):
        p = self.piece_at(r, c)
        if not p:
            return []
        color = p.color
        moves = []
        if p.kind == 'G':
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc) and self.palace_contains(color, nr, nc):
                    q = self.piece_at(nr, nc)
                    if q is None or q.color != color:
                        moves.append((r, c, nr, nc))
            og = self.find_general('b' if color == 'r' else 'r')
            if og and og[1] == c and self.count_between(r, c, og[0], og[1]) == 0:
                moves.append((r, c, og[0], og[1]))
        elif p.kind == 'A':
            for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc) and self.palace_contains(color, nr, nc):
                    q = self.piece_at(nr, nc)
                    if q is None or q.color != color:
                        moves.append((r, c, nr, nc))
        elif p.kind == 'E':
            for dr, dc in [(2, 2), (2, -2), (-2, 2), (-2, -2)]:
                nr, nc = r + dr, c + dc
                mr, mc = r + dr // 2, c + dc // 2
                if self.in_bounds(nr, nc) and self.piece_at(mr, mc) is None:
                    if not self.crossed_river(color, nr):
                        q = self.piece_at(nr, nc)
                        if q is None or q.color != color:
                            moves.append((r, c, nr, nc))
        elif p.kind == 'H':
            for dr, dc in [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]:
                nr, nc = r + dr, c + dc
                if not self.in_bounds(nr, nc):
                    continue
                if abs(dr) == 1:
                    br, bc = r, c + (1 if dc > 0 else -1)
                else:
                    br, bc = r + (1 if dr > 0 else -1), c
                if self.piece_at(br, bc) is not None:
                    continue
                q = self.piece_at(nr, nc)
                if q is None or q.color != color:
                    moves.append((r, c, nr, nc))
        elif p.kind == 'R':
            for dc in [1, -1]:
                cc = c + dc
                while self.in_bounds(r, cc):
                    q = self.piece_at(r, cc)
                    if q is None:
                        moves.append((r, c, r, cc))
                    else:
                        if q.color != color:
                            moves.append((r, c, r, cc))
                        break
                    cc += dc
            for dr in [1, -1]:
                rr = r + dr
                while self.in_bounds(rr, c):
                    q = self.piece_at(rr, c)
                    if q is None:
                        moves.append((r, c, rr, c))
                    else:
                        if q.color != color:
                            moves.append((r, c, rr, c))
                        break
                    rr += dr
        elif p.kind == 'C':
            for dc in [1, -1]:
                cc = c + dc
                while self.in_bounds(r, cc) and self.piece_at(r, cc) is None:
                    moves.append((r, c, r, cc))
                    cc += dc
                cc2 = cc + dc
                while self.in_bounds(r, cc2):
                    q = self.piece_at(r, cc2)
                    if q is not None:
                        if q.color != color and self.count_between(r, c, r, cc2) == 1:
                            moves.append((r, c, r, cc2))
                        break
                    cc2 += dc
            for dr in [1, -1]:
                rr = r + dr
                while self.in_bounds(rr, c) and self.piece_at(rr, c) is None:
                    moves.append((r, c, rr, c))
                    rr += dr
                rr2 = rr + dr
                while self.in_bounds(rr2, c):
                    q = self.piece_at(rr2, c)
                    if q is not None:
                        if q.color != color and self.count_between(r, c, rr2, c) == 1:
                            moves.append((r, c, rr2, c))
                        break
                    rr2 += dr
        elif p.kind == 'S':
            f = -1 if color == 'r' else 1
            nr, nc = r + f, c
            if self.in_bounds(nr, nc):
                q = self.piece_at(nr, nc)
                if q is None or q.color != color:
                    moves.append((r, c, nr, nc))
            if self.crossed_river(color, r):
                for dc in [-1, 1]:
                    nr, nc = r, c + dc
                    if self.in_bounds(nr, nc):
                        q = self.piece_at(nr, nc)
                        if q is None or q.color != color:
                            moves.append((r, c, nr, nc))
        return moves

    def move(self, r1, c1, r2, c2):
        p = self.grid[r1][c1]
        self.grid[r2][c2] = p
        self.grid[r1][c1] = None
        self.turn = 'b' if self.turn == 'r' else 'r'
        self.history_keys.append(self.position_key())

    def simulate_move(self, r1, c1, r2, c2):
        p = self.grid[r1][c1]
        captured = self.grid[r2][c2]
        self.grid[r2][c2] = p
        self.grid[r1][c1] = None
        return captured

    def undo_move(self, r1, c1, r2, c2, captured):
        p = self.grid[r2][c2]
        self.grid[r1][c1] = p
        self.grid[r2][c2] = captured

    def is_in_check(self, color):
        gpos = self.find_general(color)
        if gpos is None:
            return True
        gr, gc = gpos
        oc = 'b' if color == 'r' else 'r'
        rr = gr - 1
        while rr >= 0:
            q = self.grid[rr][gc]
            if q is not None:
                if q.color == oc and q.kind == 'G':
                    return True
                break
            rr -= 1
        rr = gr + 1
        while rr < BOARD_ROWS:
            q = self.grid[rr][gc]
            if q is not None:
                if q.color == oc and q.kind == 'G':
                    return True
                break
            rr += 1
        for dr, dc in [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]:
            hr, hc = gr - dr, gc - dc
            if self.in_bounds(hr, hc):
                q = self.grid[hr][hc]
                if q is not None and q.color == oc and q.kind == 'H':
                    if abs(dr) == 1:
                        br, bc = hr, hc + (1 if -dc > 0 else -1)
                    else:
                        br, bc = hr + (1 if -dr > 0 else -1), hc
                    if self.piece_at(br, bc) is None:
                        return True
        f = 1 if color == 'r' else -1
        sr, sc = gr + f, gc
        for rr, cc in [(sr, sc), (gr, gc - 1), (gr, gc + 1)]:
            if self.in_bounds(rr, cc):
                q = self.grid[rr][cc]
                if q is not None and q.color == oc and q.kind == 'S':
                    if rr == gr and abs(cc - gc) == 1 and self.crossed_river(oc, rr):
                        return True
                    if rr == sr and cc == gc:
                        return True
        for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            ar, ac = gr - dr, gc - dc
            if self.in_bounds(ar, ac):
                q = self.grid[ar][ac]
                if q is not None and q.color == oc and q.kind == 'A':
                    if self.palace_contains(oc, ar, ac) and self.palace_contains(color, gr, gc):
                        return True
        for dr, dc in [(2, 2), (2, -2), (-2, 2), (-2, -2)]:
            er, ec = gr - dr, gc - dc
            mr, mc = gr - dr // 2, gc - dc // 2
            if self.in_bounds(er, ec):
                q = self.grid[er][ec]
                if q is not None and q.color == oc and q.kind == 'E':
                    if self.piece_at(mr, mc) is None and not self.crossed_river(oc, er):
                        return True
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            rr, cc = gr + dr, gc + dc
            while self.in_bounds(rr, cc):
                q = self.grid[rr][cc]
                if q is not None:
                    if q.color == oc and q.kind == 'R':
                        return True
                    if q.color == oc and q.kind == 'C':
                        if self.count_between(rr, cc, gr, gc) == 1:
                            return True
                    break
                rr += dr
                cc += dc
        return False

    def legal_moves_from(self, r, c):
        p = self.piece_at(r, c)
        if not p or p.color != self.turn:
            return []
        res = []
        for r1, c1, r2, c2 in self.generate_moves_from(r, c):
            cap = self.simulate_move(r1, c1, r2, c2)
            ok = not self.is_in_check(p.color)
            self.undo_move(r1, c1, r2, c2, cap)
            if ok:
                res.append((r1, c1, r2, c2))
        return res

    def has_legal_moves(self, color):
        saved_turn = self.turn
        self.turn = color
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                p = self.grid[r][c]
                if p and p.color == color:
                    if self.legal_moves_from(r, c):
                        self.turn = saved_turn
                        return True
        self.turn = saved_turn
        return False

    def general_legal_moves(self, color):
        saved_turn = self.turn
        self.turn = color
        pos = self.find_general(color)
        if not pos:
            self.turn = saved_turn
            return []
        r, c = pos
        moves = self.legal_moves_from(r, c)
        self.turn = saved_turn
        return moves

    def is_stalemate(self, color):
        return (not self.is_in_check(color)) and (not self.has_legal_moves(color))


class XiangqiUI:
    def __init__(self, root):
        self.root = root
        self.board = Board()
        self.cell = 68
        self.margin = 48
        self.width = self.margin * 2 + self.cell * (BOARD_COLS - 1)
        self.height = self.margin * 2 + self.cell * (BOARD_ROWS - 1)
        self.themes = {
            '暗黑霓虹': {
                'bg_start': (26, 26, 26), 'bg_end': (56, 46, 36),
                'board_line': '#b07f2e', 'board_text': '#b07f2e', 'marker': '#c89b43',
                'red_piece': '#e74c3c', 'black_piece': '#2c3e50', 'piece_border': '#d5b16a',
                'hint': '#27ae60', 'hint_glow': '#2ecc71', 'select_ring': '#27ae60', 'select_outer': '#2ecc71',
                'info_red': '#e74c3c', 'info_blue': '#3498db', 'trail': '#f1c40f'
            },
            '古典木纹': {
                'bg_start': (80, 60, 40), 'bg_end': (140, 100, 60),
                'board_line': '#8b5a2b', 'board_text': '#5d3a1a', 'marker': '#b07f2e',
                'red_piece': '#c0392b', 'black_piece': '#3d3d3d', 'piece_border': '#c9a66b',
                'hint': '#16a085', 'hint_glow': '#1abc9c', 'select_ring': '#16a085', 'select_outer': '#1abc9c',
                'info_red': '#c0392b', 'info_blue': '#2c7dd1', 'trail': '#f39c12'
            },
            '翡翠玉纹': {
                'bg_start': (10, 36, 24), 'bg_end': (28, 80, 56),
                'board_line': '#2ecc71', 'board_text': '#27ae60', 'marker': '#2ecc71',
                'red_piece': '#e84118', 'black_piece': '#273c75', 'piece_border': '#7fddb0',
                'hint': '#00a8ff', 'hint_glow': '#4cd137', 'select_ring': '#00a8ff', 'select_outer': '#4cd137',
                'info_red': '#e84118', 'info_blue': '#00a8ff', 'trail': '#7fddb0'
            },
            '高对比': {
                'bg_start': (16, 16, 16), 'bg_end': (16, 16, 16),
                'board_line': '#ffffff', 'board_text': '#ffffff', 'marker': '#ffffff',
                'red_piece': '#ff3b30', 'black_piece': '#000000', 'piece_border': '#ffffff',
                'hint': '#00ff00', 'hint_glow': '#00ffff', 'select_ring': '#ffdd00', 'select_outer': '#ffffff',
                'info_red': '#ff3b30', 'info_blue': '#00b0ff', 'trail': '#ffd300'
            }
        }
        self.theme_name = tk.StringVar(value='暗黑霓虹')
        self.theme = self.themes[self.theme_name.get()]
        root.configure(bg='#111111')
        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg='#1a1a1a', highlightthickness=0)
        self.canvas.pack(pady=12)
        self.info = tk.Label(root, text='', font=('Microsoft YaHei', 14, 'bold'), bg='#111111', fg='#f1c40f')
        self.info.pack(pady=4)
        topbar = tk.Frame(root, bg='#111111')
        topbar.pack()
        self.btn_reset = tk.Button(topbar, text='重开', command=self.reset_game, bg='#34495e', fg='#ecf0f1', activebackground='#2c3e50', relief='flat')
        self.btn_reset.pack(side='left', padx=6)
        tk.Label(topbar, text='主题：', bg='#111111', fg='#bdc3c7').pack(side='left')
        self.theme_menu = tk.OptionMenu(topbar, self.theme_name, *self.themes.keys(), command=self.on_theme_change)
        self.theme_menu.config(bg='#2c3e50', fg='#ecf0f1', activebackground='#34495e', relief='flat')
        self.theme_menu.pack(side='left', padx=6)
        tk.Label(topbar, text='音效：', bg='#111111', fg='#bdc3c7').pack(side='left')
        self.sound_name = tk.StringVar(value='提示音')
        self.sound_menu = tk.OptionMenu(topbar, self.sound_name, '提示音', '叮当', '问号', '错误', '静音', command=self.on_sound_change)
        self.sound_menu.config(bg='#2c3e50', fg='#ecf0f1', activebackground='#34495e', relief='flat')
        self.sound_menu.pack(side='left', padx=6)
        self.btn_sound_file = tk.Button(topbar, text='选择WAV', command=self.choose_sound_file, bg='#2c3e50', fg='#ecf0f1', activebackground='#34495e', relief='flat')
        self.btn_sound_file.pack(side='left', padx=6)
        tk.Label(topbar, text='模式：', bg='#111111', fg='#bdc3c7').pack(side='left')
        self.mode_name = tk.StringVar(value='双人')
        self.mode_menu = tk.OptionMenu(topbar, self.mode_name, '双人', '人机-红控', '人机-黑控', 'AI-自对弈', command=self.on_mode_change)
        self.mode_menu.config(bg='#2c3e50', fg='#ecf0f1', activebackground='#34495e', relief='flat')
        self.mode_menu.pack(side='left', padx=6)
        self.btn_ai_step = tk.Button(topbar, text='AI落子', command=self.ai_make_move, bg='#8e44ad', fg='#ecf0f1', activebackground='#6c3483', relief='flat')
        self.btn_ai_step.pack(side='left', padx=6)
        self.btn_ai_hint = tk.Button(topbar, text='AI建议', command=self.toggle_ai_hint, bg='#8e44ad', fg='#ecf0f1', activebackground='#6c3483', relief='flat')
        self.btn_ai_hint.pack(side='left', padx=6)
        tk.Label(topbar, text='AI间隔：', bg='#111111', fg='#bdc3c7').pack(side='left')
        self.ai_delay_name = tk.StringVar(value='极快')
        self.ai_delay_ms = 0
        self.ai_delay_menu = tk.OptionMenu(topbar, self.ai_delay_name, '极快', '快速', '正常', '慢速', '很慢', command=self.on_ai_delay_change)
        self.ai_delay_menu.config(bg='#2c3e50', fg='#ecf0f1', activebackground='#34495e', relief='flat')
        self.ai_delay_menu.pack(side='left', padx=6)
        tk.Label(topbar, text='搜索深度：', bg='#111111', fg='#bdc3c7').pack(side='left')
        self.search_depth_name = tk.StringVar(value='2')
        self.search_depth_menu = tk.OptionMenu(topbar, self.search_depth_name, '1', '2', '3', command=self.on_search_depth_change)
        self.search_depth_menu.config(bg='#2c3e50', fg='#ecf0f1', activebackground='#34495e', relief='flat')
        self.search_depth_menu.pack(side='left', padx=6)
        tk.Label(topbar, text='多样性：', bg='#111111', fg='#bdc3c7').pack(side='left')
        self.ai_diversity_name = tk.StringVar(value='中')
        self.ai_diversity_menu = tk.OptionMenu(topbar, self.ai_diversity_name, '低', '中', '高', command=self.on_ai_diversity_change)
        self.ai_diversity_menu.config(bg='#2c3e50', fg='#ecf0f1', activebackground='#34495e', relief='flat')
        self.ai_diversity_menu.pack(side='left', padx=6)
        tk.Label(topbar, text='开局随机：', bg='#111111', fg='#bdc3c7').pack(side='left')
        self.ai_opening_name = tk.StringVar(value='3')
        self.ai_opening_menu = tk.OptionMenu(topbar, self.ai_opening_name, '0', '3', '5', command=self.on_ai_opening_change)
        self.ai_opening_menu.config(bg='#2c3e50', fg='#ecf0f1', activebackground='#34495e', relief='flat')
        self.ai_opening_menu.pack(side='left', padx=6)
        tk.Label(topbar, text='训练：', bg='#111111', fg='#bdc3c7').pack(side='left')
        self.train_episodes_var = tk.StringVar(value='5000')
        self.train_alpha_var = tk.StringVar(value='0.003')
        self.train_epsilon_var = tk.StringVar(value='0.15')
        self.train_ep_entry = tk.Entry(topbar, textvariable=self.train_episodes_var, width=6, bg='#2c3e50', fg='#ecf0f1', insertbackground='#ecf0f1')
        self.train_ep_entry.pack(side='left')
        self.train_alpha_entry = tk.Entry(topbar, textvariable=self.train_alpha_var, width=6, bg='#2c3e50', fg='#ecf0f1', insertbackground='#ecf0f1')
        self.train_alpha_entry.pack(side='left')
        self.train_epsilon_entry = tk.Entry(topbar, textvariable=self.train_epsilon_var, width=6, bg='#2c3e50', fg='#ecf0f1', insertbackground='#ecf0f1')
        self.train_epsilon_entry.pack(side='left')
        tk.Label(topbar, text='时长(分钟)：', bg='#111111', fg='#bdc3c7').pack(side='left')
        self.train_minutes_var = tk.StringVar(value='0')
        self.train_minutes_entry = tk.Entry(topbar, textvariable=self.train_minutes_var, width=4, bg='#2c3e50', fg='#ecf0f1', insertbackground='#ecf0f1')
        self.train_minutes_entry.pack(side='left')
        self.btn_train = tk.Button(topbar, text='开始离线训练', command=self.start_offline_training, bg='#e67e22', fg='#ecf0f1', activebackground='#d35400', relief='flat')
        self.btn_train.pack(side='left', padx=6)
        self.btn_train_pause = tk.Button(topbar, text='暂停训练', command=self.pause_training, bg='#e74c3c', fg='#ecf0f1', activebackground='#c0392b', relief='flat')
        self.btn_train_pause.pack(side='left', padx=6)
        self.btn_train_resume = tk.Button(topbar, text='继续训练', command=self.resume_training, bg='#27ae60', fg='#ecf0f1', activebackground='#1e8449', relief='flat')
        self.btn_train_resume.pack(side='left', padx=6)
        self.btn_train_stop = tk.Button(topbar, text='停止训练', command=self.stop_training, bg='#34495e', fg='#ecf0f1', activebackground='#2c3e50', relief='flat')
        self.btn_train_stop.pack(side='left', padx=6)
        self.train_status = tk.Label(root, text='', font=('Microsoft YaHei', 12, 'bold'), bg='#111111', fg='#bdc3c7')
        self.train_status.pack(pady=2)
        self.auto_ai_var = tk.BooleanVar(value=True)
        self.chk_auto_ai = tk.Checkbutton(topbar, text='自动AI应手', variable=self.auto_ai_var, command=self.on_auto_ai_change, bg='#111111', fg='#ecf0f1', activebackground='#111111', selectcolor='#2c3e50')
        self.chk_auto_ai.pack(side='left', padx=6)
        self.btn_ai_vs_ai = tk.Button(topbar, text='开始自对弈', command=self.start_ai_vs_ai, bg='#16a085', fg='#ecf0f1', activebackground='#148f77', relief='flat')
        self.btn_ai_vs_ai.pack(side='left', padx=6)
        self.selected = None
        self.hints = []
        self.pulse_phase = 0.0
        self.last_move = None
        self.trail_items = []
        self.ai_weights = self.load_ai_model()
        self.ai_nn = self.load_ai_nn()
        self.ai_device = torch.device('cuda' if (torch and torch.cuda.is_available()) else 'cpu') if torch else None
        self.ai_hint_move = None
        self.ai_hint_on = True
        self.ai_vs_ai_running = False
        self.ai_history = []
        self.game_over = False
        self.winner = None
        self.training_running = False
        self.training_pause = False
        self.training_stop = False
        self.training_time_limit_s = None
        self.ai_noise = 0.15
        self.opening_random_moves = 3
        self.sound_mode = 'alias'
        self.sound_alias = 'SystemAsterisk'
        self.sound_file = None
        self.ai_hint_busy = False
        self.canvas.bind('<Button-1>', self.on_click)
        self.draw_background()
        self.draw_board()
        self.draw_pieces()
        self.update_info()
        self.animate()

    def grid_to_xy(self, r, c):
        x = self.margin + c * self.cell
        y = self.margin + r * self.cell
        return x, y

    def draw_background(self):
        self.canvas.delete('bg')
        steps = 40
        rs, gs, bs = self.theme['bg_start']
        re, ge, be = self.theme['bg_end']
        for i in range(steps):
            t = i / (steps - 1)
            r = int(rs + (re - rs) * t)
            g = int(gs + (ge - gs) * t)
            b = int(bs + (be - bs) * t)
            color = f"#{r:02x}{g:02x}{b:02x}"
            y1 = i * (self.height / steps)
            y2 = (i + 1) * (self.height / steps)
            self.canvas.create_rectangle(0, y1, self.width, y2, outline='', fill=color, tags='bg')

    def draw_board(self):
        for r in range(BOARD_ROWS):
            y = self.margin + r * self.cell
            x1 = self.margin
            x2 = self.margin + (BOARD_COLS - 1) * self.cell
            w = 3 if (r == 0 or r == BOARD_ROWS - 1) else 2
            self.canvas.create_line(x1, y, x2, y, width=w, fill=self.theme['board_line'], tags='board')
        for c in range(BOARD_COLS):
            x = self.margin + c * self.cell
            y1 = self.margin
            y2 = self.margin + (BOARD_ROWS - 1) * self.cell
            self.canvas.create_line(x, y1, x, y2, width=2, fill=self.theme['board_line'], tags='board')
        rp_y1 = self.margin + 0 * self.cell
        rp_y2 = self.margin + 2 * self.cell
        x1 = self.margin + 3 * self.cell
        y1 = rp_y1
        x2 = self.margin + 5 * self.cell
        y2 = rp_y2
        self.canvas.create_line(x1, y1, x2, y2, width=2, fill=self.theme['board_line'], tags='board')
        self.canvas.create_line(x1, y2, x2, y1, width=2, fill=self.theme['board_line'], tags='board')
        bp_y1 = self.margin + 7 * self.cell
        bp_y2 = self.margin + 9 * self.cell
        self.canvas.create_line(self.margin + 3 * self.cell, bp_y1, self.margin + 5 * self.cell, bp_y2, width=2, fill=self.theme['board_line'], tags='board')
        self.canvas.create_line(self.margin + 3 * self.cell, bp_y2, self.margin + 5 * self.cell, bp_y1, width=2, fill=self.theme['board_line'], tags='board')
        ry = self.margin + 4 * self.cell
        self.canvas.create_rectangle(self.margin, ry, self.margin + (BOARD_COLS - 1) * self.cell, ry, width=8, outline='', tags='board')
        self.canvas.create_text(self.width / 2 - 64, ry + 20, text='楚河', font=('Microsoft YaHei', 22, 'bold'), fill=self.theme['board_text'], tags='board')
        self.canvas.create_text(self.width / 2 + 64, ry - 20, text='汉界', font=('Microsoft YaHei', 22, 'bold'), fill=self.theme['board_text'], tags='board')
        for r in [3, 6]:
            for c in [0, 2, 4, 6, 8]:
                x, y = self.grid_to_xy(r, c)
                s = 7
                self.canvas.create_line(x - s, y - s, x + s, y - s, width=1, fill=self.theme['marker'], tags='board')
                self.canvas.create_line(x - s, y + s, x + s, y + s, width=1, fill=self.theme['marker'], tags='board')

    def draw_pieces(self):
        self.canvas.delete('piece')
        self.canvas.delete('sel')
        self.canvas.delete('hint')
        self.canvas.delete('hintglow')
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                p = self.board.piece_at(r, c)
                if p:
                    x, y = self.grid_to_xy(r, c)
                    rad = 24
                    base = self.theme['red_piece'] if p.color == 'r' else self.theme['black_piece']
                    border = self.theme['piece_border']
                    self.canvas.create_oval(x - rad + 2, y - rad + 3, x + rad + 2, y + rad + 3, fill='#0d0d0d', outline='', tags='piece')
                    self.canvas.create_oval(x - rad, y - rad, x + rad, y + rad, fill=base, outline=border, width=2, tags='piece')
                    self.canvas.create_oval(x - rad + 6, y - rad + 6, x - rad + 16, y - rad + 12, fill='#ffffff', outline='', tags='piece')
                    self.canvas.create_oval(x - rad + 4, y - rad + 4, x + rad - 4, y + rad - 4, outline=border, width=1, tags='piece')
                    self.canvas.create_line(x, y - rad + 4, x, y + rad - 4, fill=border, width=1, tags='piece')
                    self.canvas.create_line(x - rad + 4, y, x + rad - 4, y, fill=border, width=1, tags='piece')
                    name = self.display_name(p)
                    self.canvas.create_text(x, y, text=name, font=('Microsoft YaHei', 17, 'bold'), fill='#fff' if p.color == 'r' else '#ecf0f1', tags='piece')
        if self.selected:
            r, c = self.selected
            x, y = self.grid_to_xy(r, c)
            base = 28
            pulse = 4 * (0.5 + 0.5 * math.sin(self.pulse_phase))
            rad = base + pulse
            self.canvas.create_oval(x - rad, y - rad, x + rad, y + rad, outline=self.theme['select_ring'], width=3, tags='sel')
            self.canvas.create_oval(x - (rad + 6), y - (rad + 6), x + (rad + 6), y + (rad + 6), outline=self.theme['select_outer'], width=1, tags='sel')
        for _, _, r2, c2 in self.hints:
            x, y = self.grid_to_xy(r2, c2)
            self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, outline=self.theme['hint_glow'], width=2, tags='hintglow')
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill=self.theme['hint'], outline='', tags='hint')
        self.draw_trail()
        self.canvas.delete('aihint')
        if self.ai_hint_on and self.ai_hint_move:
            r1, c1, r2, c2 = self.ai_hint_move
            x1, y1 = self.grid_to_xy(r1, c1)
            x2, y2 = self.grid_to_xy(r2, c2)
            self.canvas.create_line(x1, y1, x2, y2, fill='#8e44ad', width=3, dash=(6, 4), tags='aihint')
            self.canvas.create_oval(x2 - 6, y2 - 6, x2 + 6, y2 + 6, outline='#8e44ad', width=2, tags='aihint')

    def display_name(self, p):
        if p.kind == 'R':
            return '車'
        if p.kind == 'H':
            return '馬'
        if p.kind == 'E':
            return '相' if p.color == 'r' else '象'
        if p.kind == 'A':
            return '士'
        if p.kind == 'G':
            return '帅' if p.color == 'r' else '将'
        if p.kind == 'C':
            return '炮'
        if p.kind == 'S':
            return '兵' if p.color == 'r' else '卒'
        return p.kind

    def on_click(self, event):
        c = round((event.x - self.margin) / self.cell)
        r = round((event.y - self.margin) / self.cell)
        if not self.board.in_bounds(r, c):
            return
        if self.selected:
            for m in self.hints:
                if m[2] == r and m[3] == c:
                    self.last_move = (m[0], m[1], m[2], m[3])
                    self.board.move(m[0], m[1], m[2], m[3])
                    self.play_sound()
                    self.selected = None
                    self.hints = []
                    self.draw_pieces()
                    self.update_info()
                    self.post_move_checks()
                    return
        p = self.board.piece_at(r, c)
        if p and p.color == self.board.turn:
            self.selected = (r, c)
            self.hints = self.board.legal_moves_from(r, c)
        else:
            self.selected = None
            self.hints = []
        self.draw_pieces()

    def post_move_checks(self):
        oc = 'b' if self.board.turn == 'r' else 'r'
        if self.board.is_threefold_repetition():
            self.info.config(text='和棋（重复局面）')
            self.game_over = True
            self.winner = None
            self.canvas.unbind('<Button-1>')
            return
        if self.board.is_in_check(self.board.turn):
            if not self.board.has_legal_moves(self.board.turn):
                winner = oc
                self.info.config(text=('红方胜利' if winner == 'r' else '黑方胜利'))
                self.game_over = True
                self.winner = winner
                self.canvas.unbind('<Button-1>')
                return
        else:
            if not self.board.has_legal_moves(self.board.turn):
                self.info.config(text='和棋')
                self.game_over = True
                self.winner = None
                self.canvas.unbind('<Button-1>')
                return
        self.update_info()
        self.maybe_auto_ai()

    def update_info(self):
        side = '红方' if self.board.turn == 'r' else '黑方'
        check = self.board.is_in_check(self.board.turn)
        txt = f'{side}被将军' if check else f'{side}行棋'
        col = self.theme['info_red'] if (check or self.board.turn == 'r') else self.theme['info_blue']
        self.info.config(text=txt, fg=col)
        self.update_ai_hint_async()

    def animate(self):
        self.pulse_phase += 0.25
        if self.selected or self.hints:
            self.draw_pieces()
        self.fade_trail()
        self.root.after(80, self.animate)

    def reset_game(self):
        self.board = Board()
        self.selected = None
        self.hints = []
        self.last_move = None
        self.trail_items = []
        self.ai_history = []
        self.game_over = False
        self.winner = None
        self.canvas.bind('<Button-1>', self.on_click)
        self.draw_background()
        self.draw_board()
        self.draw_pieces()
        self.update_info()

    def play_sound(self):
        if winsound:
            try:
                if self.sound_mode == 'file' and self.sound_file and os.path.isfile(self.sound_file):
                    winsound.PlaySound(self.sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
                elif self.sound_alias:
                    winsound.PlaySound(self.sound_alias, winsound.SND_ALIAS | winsound.SND_ASYNC)
                else:
                    winsound.MessageBeep()
            except Exception:
                try:
                    winsound.MessageBeep()
                except Exception:
                    pass

    def draw_trail(self):
        self.canvas.delete('trail')
        if not self.last_move:
            return
        r1, c1, r2, c2 = self.last_move
        x1, y1 = self.grid_to_xy(r1, c1)
        x2, y2 = self.grid_to_xy(r2, c2)
        color = self.theme['trail']
        self.trail_items = [
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=4, tags='trail'),
            self.canvas.create_oval(x1 - 5, y1 - 5, x1 + 5, y1 + 5, outline=color, width=2, tags='trail'),
            self.canvas.create_oval(x2 - 5, y2 - 5, x2 + 5, y2 + 5, outline=color, width=2, tags='trail'),
        ]

    def fade_trail(self):
        # simple fade-out by lifetime
        if self.trail_items:
            for item in list(self.trail_items):
                try:
                    self.canvas.itemconfig(item, width=max(1, int(self.canvas.itemcget(item, 'width')) - 1))
                except Exception:
                    pass
            # remove after few cycles
            self.canvas.after(360, lambda: self.canvas.delete('trail'))
            self.trail_items = []

    def on_theme_change(self, _):
        self.theme = self.themes[self.theme_name.get()]
        self.draw_background()
        self.draw_board()
        self.draw_pieces()
        self.update_info()

    def on_sound_change(self, _):
        name = self.sound_name.get()
        mapping = {
            '提示音': 'SystemAsterisk',
            '叮当': 'SystemExclamation',
            '问号': 'SystemQuestion',
            '错误': 'SystemHand',
            '静音': None,
        }
        self.sound_mode = 'alias'
        self.sound_alias = mapping.get(name)

    def choose_sound_file(self):
        path = filedialog.askopenfilename(title='选择WAV音效文件', filetypes=[('WAV 文件', '*.wav')])
        if path:
            self.sound_mode = 'file'
            self.sound_file = path
            self.sound_name.set('自定义')

    def on_mode_change(self, _):
        self.update_ai_hint_async()
        if self.mode_name.get() == 'AI-自对弈':
            self.start_ai_vs_ai()
        else:
            self.stop_ai_vs_ai()

    def toggle_ai_hint(self):
        self.ai_hint_on = not self.ai_hint_on
        self.update_ai_hint_async()

    def on_auto_ai_change(self):
        self.maybe_auto_ai()

    def load_ai_model(self):
        p = os.path.join(os.getcwd(), 'ai_model.pkl')
        if os.path.isfile(p):
            try:
                with open(p, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def load_ai_nn(self):
        if not torch:
            return None
        p = os.path.join(os.getcwd(), 'ai_model_nn.pth')
        if os.path.isfile(p):
            try:
                class MLP(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(5, 64),
                            nn.ReLU(),
                            nn.Linear(64, 32),
                            nn.ReLU(),
                            nn.Linear(32, 1),
                        )
                    def forward(self, x):
                        return self.net(x)
                m = MLP()
                m.load_state_dict(torch.load(p, map_location='cpu'))
                m.to(self.ai_device)
                m.eval()
                return m
            except Exception:
                return None
        return None

    def ai_features(self, color):
        vals = {'R': 5, 'H': 3, 'E': 2, 'A': 2, 'G': 100, 'C': 4, 'S': 1}
        mat = 0
        soldiers_crossed = 0
        for r in range(10):
            for c in range(9):
                p = self.board.grid[r][c]
                if not p:
                    continue
                v = vals.get(p.kind, 0)
                mat += v if p.color == color else -v
                if p.kind == 'S' and p.color == color and self.board.crossed_river(color, r):
                    soldiers_crossed += 1
        saved = self.board.turn
        self.board.turn = color
        mobility = 0
        for r in range(10):
            for c in range(9):
                p = self.board.grid[r][c]
                if p and p.color == color:
                    mobility += len(self.board.legal_moves_from(r, c))
        gen_moves = len(self.board.general_legal_moves(color))
        self.board.turn = saved
        in_check = 1 if self.board.is_in_check(color) else 0
        return {
            'material': mat,
            'mobility': mobility,
            'gen_moves': gen_moves,
            'in_check': in_check,
            'soldiers_crossed': soldiers_crossed,
        }

    def ai_evaluate(self, color):
        if not self.ai_weights:
            self.ai_weights = {'material': 1.0, 'mobility': 0.2, 'gen_moves': 0.5, 'in_check': -0.8, 'soldiers_crossed': 0.3}
        f = self.ai_features(color)
        s = 0.0
        for k, w in self.ai_weights.items():
            s += w * f.get(k, 0)
        return s

    def ai_evaluate_nn(self, color):
        if not self.ai_nn:
            return self.ai_evaluate(color)
        f = self.ai_features(color)
        x = torch.tensor([[f['material'], f['mobility'], f['gen_moves'], f['in_check'], f['soldiers_crossed']]], dtype=torch.float32)
        if self.ai_device:
            x = x.to(self.ai_device)
        with torch.no_grad():
            y = self.ai_nn(x)
        return float(y.item())

    def ai_all_moves(self, color):
        saved = self.board.turn
        self.board.turn = color
        moves = []
        for r in range(10):
            for c in range(9):
                p = self.board.grid[r][c]
                if p and p.color == color:
                    moves.extend(self.board.legal_moves_from(r, c))
        self.board.turn = saved
        return moves

    def ai_choose_move(self):
        color = self.board.turn
        moves = self.ai_all_moves(color)
        if not moves:
            return None
        depth = int(self.search_depth_name.get())
        if depth <= 1:
            scored = []
            for r1, c1, r2, c2 in moves:
                cap = self.board.simulate_move(r1, c1, r2, c2)
                val = self.ai_evaluate_nn(color)
                vals = {'R': 5, 'H': 3, 'E': 2, 'A': 2, 'G': 100, 'C': 4, 'S': 1}
                oc = 'b' if color == 'r' else 'r'
                cap_val = vals.get(cap.kind, 0) if cap else 0
                mover = self.board.grid[r2][c2]
                mover_val = vals.get(mover.kind, 0) if mover else 0
                if self.board.square_attacked_by(oc, r2, c2):
                    val += cap_val - 0.7 * mover_val
                else:
                    val += cap_val
                if self.board.is_in_check(oc):
                    val += 0.6
                if self.board.is_in_check(oc) and not self.board.has_legal_moves(oc):
                    val += 100.0
                # 反复来回与重复局面惩罚
                key_after = self.board.position_key()
                # 简单往返惩罚：与上一步完全反向
                lm = self.last_move
                if lm and lm[0] == r2 and lm[1] == c2 and lm[2] == r1 and lm[3] == c1:
                    val -= 0.5
                # 三次重复的前兆（当前已出现两次，下一步将达到三次）
                count = sum(1 for k in self.board.history_keys if k == key_after)
                if count >= 2:
                    val -= 0.4
                # 多样性噪声
                if self.ai_noise > 0:
                    val += random.gauss(0, self.ai_noise)
                self.board.undo_move(r1, c1, r2, c2, cap)
                scored.append((val, (r1, c1, r2, c2)))
            scored.sort(key=lambda x: x[0], reverse=True)
            ply = len(self.board.history_keys) - 1
            top_k = self.opening_random_moves if ply <= self.opening_random_moves else 1
            top_k = max(1, min(top_k, len(scored)))
            return random.choice(scored[:top_k])[1]
        else:
            return self.ai_search(depth)

    def ai_search(self, depth):
        def negamax(depth_left, color, alpha, beta):
            if depth_left == 0:
                val = self.ai_evaluate_nn(color)
                if self.ai_noise > 0:
                    val += random.gauss(0, self.ai_noise * 0.5)
                return val
            if self.board.is_in_check(color) and not self.board.has_legal_moves(color):
                return -10000.0
            if (not self.board.is_in_check(color)) and (not self.board.has_legal_moves(color)):
                return 0.0
            best = -1e9
            moves = self.ai_all_moves(color)
            oc = 'b' if color == 'r' else 'r'
            for r1, c1, r2, c2 in moves:
                cap = self.board.simulate_move(r1, c1, r2, c2)
                val = -negamax(depth_left - 1, oc, -beta, -alpha)
                vals = {'R': 5, 'H': 3, 'E': 2, 'A': 2, 'G': 100, 'C': 4, 'S': 1}
                cap_val = vals.get(cap.kind, 0) if cap else 0
                mover = self.board.grid[r2][c2]
                mover_val = vals.get(mover.kind, 0) if mover else 0
                if self.board.square_attacked_by(oc, r2, c2):
                    val += cap_val - 0.6 * mover_val
                else:
                    val += cap_val
                # 遇到重复局面适度拉低
                key_after = self.board.position_key()
                count = sum(1 for k in self.board.history_keys if k == key_after)
                if count >= 2:
                    val -= 0.2
                if self.board.is_in_check(oc):
                    val += 0.5
                if self.board.is_in_check(oc) and not self.board.has_legal_moves(oc):
                    val += 100.0
                self.board.undo_move(r1, c1, r2, c2, cap)
                if val > best:
                    best = val
                if val > alpha:
                    alpha = val
                if alpha >= beta:
                    break
            return best
        color = self.board.turn
        moves = self.ai_all_moves(color)
        scored = []
        oc = 'b' if color == 'r' else 'r'
        for r1, c1, r2, c2 in moves:
            cap = self.board.simulate_move(r1, c1, r2, c2)
            val = -negamax(depth - 1, oc, -1e9, 1e9)
            self.board.undo_move(r1, c1, r2, c2, cap)
            scored.append((val, (r1, c1, r2, c2)))
        scored.sort(key=lambda x: x[0], reverse=True)
        ply = len(self.board.history_keys) - 1
        top_k = self.opening_random_moves if ply <= self.opening_random_moves else 1
        top_k = max(1, min(top_k, len(scored)))
        return random.choice(scored[:top_k])[1]

    def ai_make_move(self):
        mode = self.mode_name.get()
        if mode == '双人':
            return
        if (mode == '人机-红控' and self.board.turn == 'b') or (mode == '人机-黑控' and self.board.turn == 'r'):
            mv = self.ai_choose_move()
            if mv:
                self.last_move = mv
                # 记录特征用于轻量训练
                self.ai_history.append((self.board.turn, self.ai_features(self.board.turn)))
                self.board.move(*mv)
                self.play_sound()
                self.draw_pieces()
                self.update_info()
                self.post_move_checks()
                if self.game_over:
                    self.update_ai_weights()
                    self.save_ai_model()

    def update_ai_hint(self):
        mode = self.mode_name.get()
        if not self.ai_hint_on:
            self.ai_hint_move = None
            self.canvas.delete('aihint')
            return
        if mode == '双人':
            color = self.board.turn
            self.ai_hint_move = self.ai_choose_move()
            self.draw_pieces()
            return
        if mode == '人机-红控' and self.board.turn == 'b':
            self.ai_hint_move = self.ai_choose_move()
        elif mode == '人机-黑控' and self.board.turn == 'r':
            self.ai_hint_move = self.ai_choose_move()
        else:
            self.ai_hint_move = None
        self.draw_pieces()

    def update_ai_hint_async(self):
        if self.ai_hint_busy:
            return
        mode = self.mode_name.get()
        if not self.ai_hint_on:
            self.ai_hint_move = None
            self.canvas.delete('aihint')
            return
        def worker():
            mv = None
            if mode == '双人':
                mv = self.ai_choose_move()
            elif mode == '人机-红控' and self.board.turn == 'b':
                mv = self.ai_choose_move()
            elif mode == '人机-黑控' and self.board.turn == 'r':
                mv = self.ai_choose_move()
            self.root.after(0, lambda: self._apply_ai_hint(mv))
        self.ai_hint_busy = True
        threading.Thread(target=worker, daemon=True).start()

    def _apply_ai_hint(self, mv):
        self.ai_hint_busy = False
        self.ai_hint_move = mv
        self.draw_pieces()

    def maybe_auto_ai(self):
        mode = self.mode_name.get()
        if self.auto_ai_var.get() and mode in ('人机-红控', '人机-黑控'):
            self.ai_make_move()

    def start_ai_vs_ai(self):
        if self.ai_vs_ai_running:
            return
        self.mode_name.set('AI-自对弈')
        self.ai_vs_ai_running = True
        self.btn_ai_vs_ai.config(text='停止自对弈', command=self.stop_ai_vs_ai)
        self.auto_ai_var.set(False)
        self.ai_vs_ai_step()

    def stop_ai_vs_ai(self):
        if not self.ai_vs_ai_running:
            return
        self.ai_vs_ai_running = False
        self.btn_ai_vs_ai.config(text='开始自对弈', command=self.start_ai_vs_ai)

    def ai_vs_ai_step(self):
        if not self.ai_vs_ai_running:
            return
        if self.game_over:
            self.update_ai_weights()
            self.save_ai_model()
            self.reset_game()
        mv = self.ai_choose_move()
        if mv:
            self.last_move = mv
            self.ai_history.append((self.board.turn, self.ai_features(self.board.turn)))
            self.board.move(*mv)
            self.play_sound()
            self.draw_pieces()
            self.update_info()
            self.post_move_checks()
        self.root.after(0, self.ai_vs_ai_step)

    def on_ai_delay_change(self, _):
        mapping = {
            '极快': 0,
            '快速': 0,
            '正常': 0,
            '慢速': 0,
            '很慢': 0,
        }
        self.ai_delay_ms = mapping.get(self.ai_delay_name.get(), 0)

    def on_search_depth_change(self, _):
        pass

    def on_ai_diversity_change(self, _):
        mapping = {
            '低': 0.0,
            '中': 0.15,
            '高': 0.35,
        }
        self.ai_noise = mapping.get(self.ai_diversity_name.get(), 0.15)

    def on_ai_opening_change(self, _):
        try:
            self.opening_random_moves = int(self.ai_opening_name.get())
        except Exception:
            self.opening_random_moves = 3

    def update_ai_weights(self):
        if not self.ai_history:
            return
        reward = 1 if self.winner == 'r' else -1
        for color, feats in self.ai_history:
            sgn = 1 if color == 'r' else -1
            for k, v in feats.items():
                self.ai_weights[k] = self.ai_weights.get(k, 0.0) + 0.005 * reward * sgn * v
        self.ai_history = []

    def save_ai_model(self):
        try:
            with open(os.path.join(os.getcwd(), 'ai_model.pkl'), 'wb') as f:
                pickle.dump(self.ai_weights, f)
        except Exception:
            pass

    def start_offline_training(self):
        if self.training_running:
            return
        try:
            episodes = int(self.train_episodes_var.get())
        except Exception:
            episodes = 5000
        try:
            alpha = float(self.train_alpha_var.get())
        except Exception:
            alpha = 0.003
        try:
            epsilon = float(self.train_epsilon_var.get())
        except Exception:
            epsilon = 0.15
        depth = int(self.search_depth_name.get())
        try:
            minutes = float(self.train_minutes_var.get())
            self.training_time_limit_s = int(minutes * 60) if minutes > 0 else None
        except Exception:
            self.training_time_limit_s = None
        self.training_pause = False
        self.training_stop = False
        self.training_running = True
        self.train_status.config(text='训练中...')
        threading.Thread(target=lambda: self.offline_train_worker(episodes, alpha, epsilon, depth), daemon=True).start()

    def offline_train_worker(self, episodes, alpha, epsilon, depth):
        start_t = time.time()
        if torch:
            class MLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(5, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                    )
                def forward(self, x):
                    return self.net(x)
            device = self.ai_device if self.ai_device else torch.device('cpu')
            model = MLP().to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            def eval_nn(color, f):
                x = torch.tensor([[f['material'], f['mobility'], f['gen_moves'], f['in_check'], f['soldiers_crossed']]], dtype=torch.float32, device=device)
                with torch.no_grad():
                    y = model(x)
                return float(y.item())
            def negamax(board, depth_left, color, alpha_b, beta_b):
                if depth_left == 0:
                    f = self.ai_features(color)
                    return eval_nn(color, f)
                if board.is_in_check(color) and not board.has_legal_moves(color):
                    return -10000.0
                if (not board.is_in_check(color)) and (not board.has_legal_moves(color)):
                    return 0.0
                best = -1e9
                moves = self.ai_all_moves(color)
                oc = 'b' if color == 'r' else 'r'
                for r1, c1, r2, c2 in moves:
                    cap = board.simulate_move(r1, c1, r2, c2)
                    val = -negamax(board, depth_left - 1, oc, -beta_b, -alpha_b)
                    board.undo_move(r1, c1, r2, c2, cap)
                    if val > best:
                        best = val
                    if val > alpha_b:
                        alpha_b = val
                    if alpha_b >= beta_b:
                        break
                return best
            i = 0
            while i < episodes:
                if self.training_stop:
                    break
                if self.training_time_limit_s is not None and (time.time() - start_t) >= self.training_time_limit_s:
                    break
                if self.training_pause:
                    time.sleep(0.1)
                    continue
                b = Board()
                history = []
                while True:
                    if self.training_stop:
                        break
                    if self.training_time_limit_s is not None and (time.time() - start_t) >= self.training_time_limit_s:
                        break
                    if self.training_pause:
                        time.sleep(0.1)
                        continue
                    color = b.turn
                    moves = self.ai_all_moves(color)
                    if not moves:
                        reward = 1 if b.is_in_check(color) and (color == 'b') else (-1 if b.is_in_check(color) and (color == 'r') else 0)
                        inputs = []
                        targets = []
                        for c0, f0 in history:
                            sgn = 1 if c0 == 'r' else -1
                            inputs.append([f0['material'], f0['mobility'], f0['gen_moves'], f0['in_check'], f0['soldiers_crossed']])
                            targets.append([reward * sgn])
                        if inputs:
                            x = torch.tensor(inputs, dtype=torch.float32, device=device)
                            y = torch.tensor(targets, dtype=torch.float32, device=device)
                            pred = model(x)
                            loss = ((pred - y) ** 2).mean()
                            opt.zero_grad()
                            loss.backward()
                            opt.step()
                        break
                    if torch.rand(1).item() < epsilon:
                        mv = random.choice(moves)
                    else:
                        best = None
                        best_val = -1e9
                        oc = 'b' if color == 'r' else 'r'
                        for r1, c1, r2, c2 in moves:
                            cap = b.simulate_move(r1, c1, r2, c2)
                            val = -negamax(b, depth - 1, oc, -1e9, 1e9)
                            b.undo_move(r1, c1, r2, c2, cap)
                            if val > best_val:
                                best_val = val
                                best = (r1, c1, r2, c2)
                        mv = best
                    f = self.ai_features(color)
                    history.append((color, f))
                    b.move(*mv)
                    oc = 'b' if b.turn == 'r' else 'r'
                    if b.is_in_check(b.turn) and not b.has_legal_moves(b.turn):
                        reward = 1 if oc == 'r' else -1
                        inputs = []
                        targets = []
                        for c0, f0 in history:
                            sgn = 1 if c0 == 'r' else -1
                            inputs.append([f0['material'], f0['mobility'], f0['gen_moves'], f0['in_check'], f0['soldiers_crossed']])
                            targets.append([reward * sgn])
                        x = torch.tensor(inputs, dtype=torch.float32, device=device)
                        y = torch.tensor(targets, dtype=torch.float32, device=device)
                        pred = model(x)
                        loss = ((pred - y) ** 2).mean()
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        break
                    if (not b.is_in_check(b.turn)) and (not b.has_legal_moves(b.turn)):
                        reward = 0
                        inputs = []
                        targets = []
                        for c0, f0 in history:
                            sgn = 1 if c0 == 'r' else -1
                            inputs.append([f0['material'], f0['mobility'], f0['gen_moves'], f0['in_check'], f0['soldiers_crossed']])
                            targets.append([reward * sgn])
                        if inputs:
                            x = torch.tensor(inputs, dtype=torch.float32, device=device)
                            y = torch.tensor(targets, dtype=torch.float32, device=device)
                            pred = model(x)
                            loss = ((pred - y) ** 2).mean()
                            opt.zero_grad()
                            loss.backward()
                            opt.step()
                        break
                if i % 200 == 0:
                    self.root.after(0, lambda n=i: self.train_status.config(text=f'训练进度 {n}/{episodes}'))
                i += 1
            torch.save(model.state_dict(), os.path.join(os.getcwd(), 'ai_model_nn.pth'))
            if self.training_stop:
                self.root.after(0, lambda: self.train_status.config(text='训练已停止(已保存最新模型)'))
            elif self.training_time_limit_s is not None and (time.time() - start_t) >= self.training_time_limit_s:
                self.root.after(0, lambda: self.train_status.config(text='训练到时完成(已保存最新模型)'))
            else:
                self.root.after(0, lambda: self.train_status.config(text='训练完成'))
            self.ai_nn = self.load_ai_nn()
        else:
            if not self.ai_weights:
                self.ai_weights = {'material': 1.0, 'mobility': 0.2, 'gen_moves': 0.5, 'in_check': -0.8, 'soldiers_crossed': 0.3}
            i = 0
            while i < episodes:
                if self.training_stop:
                    break
                if self.training_time_limit_s is not None and (time.time() - start_t) >= self.training_time_limit_s:
                    break
                if self.training_pause:
                    time.sleep(0.1)
                    continue
                b = Board()
                history = []
                while True:
                    if self.training_stop:
                        break
                    if self.training_time_limit_s is not None and (time.time() - start_t) >= self.training_time_limit_s:
                        break
                    if self.training_pause:
                        time.sleep(0.1)
                        continue
                    color = b.turn
                    mv = self.ai_choose_move()
                    if not mv:
                        reward = 1 if b.is_in_check(color) and (color == 'b') else (-1 if b.is_in_check(color) and (color == 'r') else 0)
                        for c0, f0 in history:
                            sgn = 1 if c0 == 'r' else -1
                            for k, v in f0.items():
                                self.ai_weights[k] = self.ai_weights.get(k, 0.0) + alpha * reward * sgn * v
                        break
                    f = self.ai_features(color)
                    history.append((color, f))
                    b.move(*mv)
                    oc = 'b' if b.turn == 'r' else 'r'
                    if b.is_in_check(b.turn) and not b.has_legal_moves(b.turn):
                        reward = 1 if oc == 'r' else -1
                        for c0, f0 in history:
                            sgn = 1 if c0 == 'r' else -1
                            for k, v in f0.items():
                                self.ai_weights[k] = self.ai_weights.get(k, 0.0) + alpha * reward * sgn * v
                        break
                    if (not b.is_in_check(b.turn)) and (not b.has_legal_moves(b.turn)):
                        break
                if i % 500 == 0:
                    self.root.after(0, lambda n=i: self.train_status.config(text=f'训练进度 {n}/{episodes}'))
                i += 1
            self.save_ai_model()
            if self.training_stop:
                self.root.after(0, lambda: self.train_status.config(text='训练已停止(已保存最新模型)'))
            elif self.training_time_limit_s is not None and (time.time() - start_t) >= self.training_time_limit_s:
                self.root.after(0, lambda: self.train_status.config(text='训练到时完成(已保存最新模型)'))
            else:
                self.root.after(0, lambda: self.train_status.config(text='训练完成'))
        self.training_running = False

    def pause_training(self):
        if self.training_running:
            self.training_pause = True
            self.train_status.config(text='训练已暂停')

    def resume_training(self):
        if self.training_running:
            self.training_pause = False
            self.train_status.config(text='训练中...')

    def stop_training(self):
        if self.training_running:
            self.training_stop = True


def main():
    root = tk.Tk()
    root.title('中国象棋')
    XiangqiUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
