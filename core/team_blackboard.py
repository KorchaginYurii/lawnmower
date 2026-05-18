class TeamBlackboard:
    def __init__(self):
        self.claimed_sectors = {}   # sector_id -> robot_id
        self.robot_positions = {}   # robot_id -> pos
        self.shared_memory = None

    def update_robot(self, robot_id, pos):
        self.robot_positions[robot_id] = pos

    def claim_sector(self, robot_id, sector_id):
        if sector_id is None:
            return False

        owner = self.claimed_sectors.get(sector_id)

        if owner is None or owner == robot_id:
            self.claimed_sectors[sector_id] = robot_id
            return True

        return False

    def release_sector(self, robot_id, sector_id):
        if sector_id is None:
            return

        if self.claimed_sectors.get(sector_id) == robot_id:
            del self.claimed_sectors[sector_id]

    def sector_owner(self, sector_id):
        return self.claimed_sectors.get(sector_id)

    def is_sector_available(self, robot_id, sector_id):
        owner = self.sector_owner(sector_id)
        return owner is None or owner == robot_id

    def update_shared_memory(self, memory):
        if memory is None:
            return

        if self.shared_memory is None:
            self.shared_memory = memory.copy()
            return

        known = memory.seen == 1

        self.shared_memory.map[known] = memory.map[known]
        self.shared_memory.seen[known] = 1

    def sync_memory(self, local_memory):
        if self.shared_memory is None:
            return local_memory

        known = self.shared_memory.seen == 1

        local_memory.map[known] = self.shared_memory.map[known]
        local_memory.seen[known] = 1

        return local_memory

