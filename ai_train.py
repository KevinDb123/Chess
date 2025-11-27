import random
import pickle
from xiangqi import Board
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except Exception:
    torch = None


class XiangqiAI:
    def __init__(self):
        self.weights = {
            'material': 1.0,
            'mobility': 0.2,
            'gen_moves': 0.5,
            'in_check': -0.8,
            'soldiers_crossed': 0.3,
        }

    def value_map(self):
        return {'R': 5, 'H': 3, 'E': 2, 'A': 2, 'G': 100, 'C': 4, 'S': 1}

    def features(self, board: Board, color: str):
        vals = self.value_map()
        mat = 0
        soldiers_crossed = 0
        for r in range(10):
            for c in range(9):
                p = board.grid[r][c]
                if not p:
                    continue
                v = vals.get(p.kind, 0)
                mat += v if p.color == color else -v
                if p.kind == 'S' and p.color == color and board.crossed_river(color, r):
                    soldiers_crossed += 1
        saved = board.turn
        board.turn = color
        mobility = 0
        for r in range(10):
            for c in range(9):
                p = board.grid[r][c]
                if p and p.color == color:
                    mobility += len(board.legal_moves_from(r, c))
        gen_moves = len(board.general_legal_moves(color))
        board.turn = saved
        in_check = 1 if board.is_in_check(color) else 0
        return {
            'material': mat,
            'mobility': mobility,
            'gen_moves': gen_moves,
            'in_check': in_check,
            'soldiers_crossed': soldiers_crossed,
        }

    def evaluate(self, board: Board, color: str):
        f = self.features(board, color)
        s = 0.0
        for k, w in self.weights.items():
            s += w * f.get(k, 0)
        return s

    def all_moves(self, board: Board, color: str):
        saved = board.turn
        board.turn = color
        moves = []
        for r in range(10):
            for c in range(9):
                p = board.grid[r][c]
                if p and p.color == color:
                    moves.extend(board.legal_moves_from(r, c))
        board.turn = saved
        return moves

    def choose_move(self, board: Board, color: str, epsilon=0.1):
        moves = self.all_moves(board, color)
        if not moves:
            return None
        if random.random() < epsilon:
            return random.choice(moves)
        best = None
        best_val = -1e9
        for r1, c1, r2, c2 in moves:
            cap = board.simulate_move(r1, c1, r2, c2)
            val = self.evaluate(board, color)
            board.undo_move(r1, c1, r2, c2, cap)
            if val > best_val:
                best_val = val
                best = (r1, c1, r2, c2)
        return best


class PolicyValueNet(nn.Module):
    def __init__(self, board_width=9, board_height=10, channels=14, action_size=2048):
        super().__init__()
        self.board_width = board_width
        self.board_height = board_height
        self.channels = channels
        self.action_size = action_size
        # backbone
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # policy head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_height * board_width, action_size)
        # value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(1 * board_height * board_width, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: [B, C=14, H=10, W=9]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # policy
        p = self.policy_conv(x)
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)
        # value
        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return policy_logits, v


class XiangqiAI_NN:
    def __init__(self):
        self.model = PolicyValueNet()
        self.device = torch.device('cuda' if torch and torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

    def encode_board(self, board: Board, color: str):
        kinds = ['R', 'H', 'E', 'A', 'G', 'C', 'S']
        tensor = torch.zeros(14, 10, 9, dtype=torch.float32, device=self.device)
        for r in range(10):
            for c in range(9):
                p = board.grid[r][c]
                if not p:
                    continue
                try:
                    idx = kinds.index(p.kind)
                except ValueError:
                    continue
                ch = idx if p.color == 'r' else 7 + idx
                tensor[ch, r, c] = 1.0
        if color == 'b':
            tensor = torch.flip(tensor, dims=[1])
        return tensor.unsqueeze(0)

    def predict(self, board: Board, color: str):
        x = self.encode_board(board, color)
        with torch.no_grad():
            policy_logits, value = self.model(x)
        return policy_logits, float(value.item())

    def all_moves(self, board: Board, color: str):
        saved = board.turn
        board.turn = color
        moves = []
        for r in range(10):
            for c in range(9):
                p = board.grid[r][c]
                if p and p.color == color:
                    moves.extend(board.legal_moves_from(r, c))
        board.turn = saved
        return moves

    def negamax(self, board: Board, depth: int, color: str, alpha: float, beta: float):
        if depth == 0:
            _, val = self.predict(board, color)
            return val
        if board.is_in_check(color) and not board.has_legal_moves(color):
            return -10000.0
        if (not board.is_in_check(color)) and (not board.has_legal_moves(color)):
            return 0.0
        best = -1e9
        moves = self.all_moves(board, color)
        oc = 'b' if color == 'r' else 'r'
        for r1, c1, r2, c2 in moves:
            cap = board.simulate_move(r1, c1, r2, c2)
            val = -self.negamax(board, depth - 1, oc, -beta, -alpha)
            board.undo_move(r1, c1, r2, c2, cap)
            if val > best:
                best = val
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break
        return best

    def choose_move(self, board: Board, color: str, depth=2, epsilon=0.1):
        moves = self.all_moves(board, color)
        if not moves:
            return None
        if random.random() < epsilon:
            return random.choice(moves)
        best = None
        best_val = -1e9
        oc = 'b' if color == 'r' else 'r'
        for r1, c1, r2, c2 in moves:
            cap = board.simulate_move(r1, c1, r2, c2)
            val = -self.negamax(board, depth - 1, oc, -1e9, 1e9)
            board.undo_move(r1, c1, r2, c2, cap)
            if val > best_val:
                best_val = val
                best = (r1, c1, r2, c2)
        return best



def train_and_save(path='ai_model_sota.pth', episodes=200, alpha=0.01, epsilon=0.2, depth=1, use_search=False, max_plies=160):
    if torch:
        ai = XiangqiAI_NN()
        value_criterion = nn.MSELoss()
        for _ in range(episodes):
            b = Board()
            history = []  # store (encoded_tensor, dummy_policy, color)
            ply = 0
            while True:
                color = b.turn
                x = ai.encode_board(b, color)
                dummy_policy = torch.zeros((1, ai.model.action_size), dtype=torch.float32, device=ai.device)
                history.append((x, dummy_policy, color))
                if use_search:
                    mv = ai.choose_move(b, color, depth=depth, epsilon=epsilon)
                else:
                    # 1-ply 快速选择：模拟一步后用价值网络评估
                    moves = ai.all_moves(b, color)
                    if not moves:
                        mv = None
                    else:
                        best = None
                        best_val = -1e9
                        oc = 'b' if color == 'r' else 'r'
                        for r1, c1, r2, c2 in moves:
                            cap = b.simulate_move(r1, c1, r2, c2)
                            _, v = ai.predict(b, oc)
                            val = -v
                            b.undo_move(r1, c1, r2, c2, cap)
                            if val > best_val:
                                best_val = val
                                best = (r1, c1, r2, c2)
                        mv = best
                if not mv:
                    reward = 1 if b.is_in_check(color) else 0
                    total_loss = 0.0
                    for x0, pol0, c0 in history:
                        sgn = 1.0 if c0 == 'r' else -1.0
                        policy_logits, value_pred = ai.model(x0)
                        value_target = torch.tensor([[reward * sgn]], dtype=torch.float32, device=ai.device)
                        value_loss = value_criterion(value_pred, value_target)
                        policy_loss = torch.zeros_like(value_loss)
                        total_loss = total_loss + value_loss + 1e-4 * policy_loss
                    ai.opt.zero_grad()
                    total_loss.backward()
                    ai.opt.step()
                    break
                b.move(*mv)
                ply += 1
                oc = 'b' if b.turn == 'r' else 'r'
                if b.is_in_check(b.turn) and not b.has_legal_moves(b.turn):
                    reward = 1 if oc == 'r' else -1
                    total_loss = 0.0
                    for x0, pol0, c0 in history:
                        sgn = 1.0 if c0 == 'r' else -1.0
                        policy_logits, value_pred = ai.model(x0)
                        value_target = torch.tensor([[reward * sgn]], dtype=torch.float32, device=ai.device)
                        value_loss = value_criterion(value_pred, value_target)
                        policy_loss = torch.zeros_like(value_loss)
                        total_loss = total_loss + value_loss + 1e-4 * policy_loss
                    ai.opt.zero_grad()
                    total_loss.backward()
                    ai.opt.step()
                    break
                if (not b.is_in_check(b.turn)) and (not b.has_legal_moves(b.turn)):
                    reward = 0
                    total_loss = 0.0
                    for x0, pol0, c0 in history:
                        sgn = 1.0 if c0 == 'r' else -1.0
                        policy_logits, value_pred = ai.model(x0)
                        value_target = torch.tensor([[reward * sgn]], dtype=torch.float32, device=ai.device)
                        value_loss = value_criterion(value_pred, value_target)
                        policy_loss = torch.zeros_like(value_loss)
                        total_loss = total_loss + value_loss + 1e-4 * policy_loss
                    ai.opt.zero_grad()
                    total_loss.backward()
                    ai.opt.step()
                    break
                if ply >= max_plies:
                    # 到步数上限视作和棋，价值为0
                    reward = 0
                    total_loss = 0.0
                    for x0, pol0, c0 in history:
                        sgn = 1.0 if c0 == 'r' else -1.0
                        policy_logits, value_pred = ai.model(x0)
                        value_target = torch.tensor([[reward * sgn]], dtype=torch.float32, device=ai.device)
                        value_loss = value_criterion(value_pred, value_target)
                        policy_loss = torch.zeros_like(value_loss)
                        total_loss = total_loss + value_loss + 1e-4 * policy_loss
                    ai.opt.zero_grad()
                    total_loss.backward()
                    ai.opt.step()
                    break
        torch.save(ai.model.state_dict(), path)
    else:
        ai = XiangqiAI()
        ai.self_play(episodes=episodes, alpha=alpha, epsilon=epsilon)
        with open('ai_model.pkl', 'wb') as f:
            pickle.dump(ai.weights, f)


def main():
    train_and_save()


if __name__ == '__main__':
    main()
