from core.config import MOVE_COST, TURN_COST, CUT_COST


class EnergySystem:
    def __init__(self, max_energy=100.0):
        self.max_energy = float(max_energy)
        self.energy = float(max_energy)

    def reset(self):
        self.energy = self.max_energy

    def spend_move(self):
        self.energy -= MOVE_COST

    def spend_turn(self):
        self.energy -= TURN_COST

    def spend_cut(self):
        self.energy -= CUT_COST

    def spend(self, amount):
        self.energy -= amount

    def can_spend(self, amount):
        return self.energy >= amount

    def can_reach(self, cost, reserve=5.0):
        return self.energy >= cost + reserve

    def recharge(self):
        self.energy = self.max_energy