import numpy as np
import random
import torch

from core.config import GAMMA


class MCTSNode:
    def __init__(self, env, parent=None):
        self.env = env
        self.parent = parent

        self.children = {}
        self.N = np.zeros(4, dtype=np.float32)
        self.W = np.zeros(4, dtype=np.float32)
        self.Q = np.zeros(4, dtype=np.float32)
        self.P = np.zeros(4, dtype=np.float32)

        self.is_expanded = False


class MCTS:
    def __init__(self, agent, simulations=50, c=1.5):
        self.agent = agent
        self.sims = simulations
        self.c = c

    def evaluate(self, env):
        s = self.agent.get_state(env)

        #with torch.no_grad():
        #    pi, v = self.agent.net(torch.FloatTensor(s).unsqueeze(0).to(self.agent.device))
        x = torch.tensor(
            s,
            dtype=torch.float32,
            device=self.agent.device
        ).unsqueeze(0)

        with torch.no_grad():
            pi, v = self.agent.net(x)

        policy = pi.cpu().numpy()[0]
        value = v.item()

        #print("MCTS tensor device:", torch.tensor([1, 2, 3]).device)

        return policy, value

#########################
# SELECT + EXPAND + BACKPROP
#########################
    def run(self, env, temp=1.0, training=True):
        root = MCTSNode(env.clone())
        root.P, root_value = self.evaluate(env)

        # 🔥 DIRICHLET NOISE (ТОЛЬКО ДЛЯ ROOT)
        if training:
            noise = np.random.dirichlet([0.3] * 4)
            root.P = 0.75 * root.P + 0.25 * noise

        root.is_expanded = True

        for _ in range(self.sims):
            self.simulate(root)

        def tree_depth(node):
            if not node.children:
                return 0
            return 1 + max(
                tree_depth(child)
                for child, _, _ in node.children.values()
            )
        debug_depth = tree_depth(root)

        visits = root.N.astype(np.float32)


        # ===== TEMP = 0 =====
        if temp == 0:
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
        # ===== NORMAL TEMP =====
            visits = visits ** (1 / temp)  # temp
            probs = visits / (visits.sum() + 1e-8)

        debug = {
            "visits": root.N.copy(),
            "policy": probs.copy(),
            "root": root,
            "depth": debug_depth
        }
        return probs, debug

    def simulate(self, node):
        path = []
        cur = node

        while cur.is_expanded:
            total_N = cur.N.sum() + 1

            U = self.c * cur.P * np.sqrt(total_N) / (1 + cur.N)

            #Усиль exploration
            if random.random() < 0.05:
                a = random.randint(0, 3)
            else:
                a = np.argmax(cur.Q + U)

            path.append((cur, a))

            if a not in cur.children:
                env = cur.env.clone()
                r, done = env.step(a)
                child = MCTSNode(env, cur)
                cur.children[a] = (child, r, done)

            child, r, done = cur.children[a]


            # 🔥 shaping: тупики
            #r += cur.env.dead_end_penalty(child.env.pos)
            #r += child.env.flood_fill_penalty(child.env.pos)

            # 🔥 NEW: возврат домой
            if np.sum(child.env.grid == 1) == 0:
                sx, sy = child.env.start_pos
                x, y = child.env.pos

                dist = abs(x - sx) + abs(y - sy)
                r -= 0.1 * dist

            if done:
                value = r #np.tanh(r / 100)
                break

            cur = child

        else:
            p, v = self.evaluate(cur.env)
            cur.P = p
            cur.is_expanded = True
            value = v

        # backprop
        for node, a in reversed(path):
            child, r, done = node.children[a]
            value = np.tanh(r / 5.0) + GAMMA * value

            node.N[a] += 1
            node.W[a] += value
            node.Q[a] = node.W[a] / node.N[a]

def extract_rollout(root, max_depth=20):
    path = []
    node = root

    for _ in range(max_depth):
        if not node.children:
            break

        # если есть посещения — идём по ним
        if node.N.sum() > 0:
            actions = np.argsort(-node.N)
        else:
            actions = list(node.children.keys())

        chosen = None

        for a in actions:
            a = int(a)
            if a in node.children:
                chosen = a
                break

        if chosen is None:
            break

        child, _, done = node.children[chosen]
        path.append(child.env.pos)

        if done:
            break

        node = child

    return path