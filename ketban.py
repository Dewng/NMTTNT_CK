import pandas as pd
import numpy as np
import heapq
import json
import os
import glob
from collections import deque
import re
import time


# ==========================================
# 0. BẢNG PHÂN LOẠI (KHÔNG PHỤ THUỘC EXCEL)
# ==========================================
def _norm_key(s: str) -> str:
    """Chuẩn hoá chuỗi để so khớp (không phân biệt hoa/thường, chuẩn hoá dấu gạch)."""
    if s is None:
        return ""
    s = str(s).strip()
    if s in ["", "nan", "NaN", "-"]:
        return "-"
    s = s.replace("—", "-").replace("–", "-").replace("−", "-")
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


# Bảng Trường/ngành nghề -> Ngành nghề con
DEFAULT_INDUSTRY_GROUPS = {
    "Sinh viên": ["Sinh viên"],
    "Công nghệ & Kỹ thuật": [
        "Công nghệ thông tin", "Kỹ thuật phần mềm", "Trí tuệ nhân tạo", "Kỹ sư điện – điện tử",
        "Cơ khí – tự động hóa", "Kỹ thuật ô tô", "Kỹ thuật xây dựng", "Kỹ sư môi trường",
        "An ninh mạng", "Khoa học dữ liệu",
    ],
    "Kinh tế – Tài chính – Kinh doanh": [
        "Kế toán", "Kiểm toán", "Tài chính – ngân hàng", "Bảo hiểm", "Đầu tư – chứng khoán",
        "Quản trị kinh doanh", "Quản lý chuỗi cung ứng", "Thương mại điện tử",
        "Marketing – truyền thông", "Nhân sự",
    ],
    "Y tế – Giáo dục – Xã hội": [
        "Bác sĩ", "Dược sĩ", "Điều dưỡng", "Kỹ thuật viên y học", "Giáo viên – giảng viên",
        "Tư vấn giáo dục", "Tâm lý học", "Công tác xã hội", "Luật sư", "Quan hệ công chúng",
    ],
    "Dịch vụ – Du lịch – Giải trí": [
        "Du lịch – lữ hành", "Nhà hàng – khách sạn", "Tiếp viên hàng không", "Tổ chức sự kiện",
        "Hướng dẫn viên du lịch", "Thiết kế đồ họa", "Thiết kế thời trang", "Biên tập nội dung số",
        "Sản xuất video", "Truyền thông xã hội",
    ],
    "Lao động kỹ năng – Thẩm mỹ – Sáng tạo": [
        "Thẩm mỹ – làm đẹp", "Chăm sóc sắc đẹp", "Nghệ thuật biểu diễn", "Nhiếp ảnh", "Làm phim",
        "Công nghệ thực phẩm", "Kiến trúc", "Ngôn ngữ học", "Hành chính – thư ký", "Quân đội – công an",
    ],
}

# Dùng để nhận biết khi người dùng nhập thẳng "tên trường/ngành nghề" thay vì ngành con
INDUSTRY_GROUP_KEYS_NORM = {_norm_key(g) for g in DEFAULT_INDUSTRY_GROUPS.keys()}

# Bảng Sở thích -> Sở thích con
DEFAULT_INTEREST_GROUPS = {
    "Sáng tạo": ["Vẽ tranh", "Chụp ảnh", "Viết lách", "Làm đồ thủ công"],
    "Giải trí": ["Nghe nhạc", "Xem phim", "Chơi nhạc cụ", "Chơi game"],
    "Vận động": ["Tập gym", "Yoga", "Chạy bộ", "Đi bộ", "Đạp xe", "Bơi lội"],
    "Thư giãn": ["Đọc sách", "Thiền định", "Làm vườn", "Nấu ăn", "Làm bánh"],
    "Khám phá": ["Du lịch", "Học ngoại ngữ", "Khám phá ẩm thực", "Tham gia hoạt động tình nguyện"],
}

# Tạo bảng tra nhanh: ngành nghề con -> trường/ngành nghề
INDUSTRY_CHILD_TO_GROUP = {}
for _grp, _items in DEFAULT_INDUSTRY_GROUPS.items():
    INDUSTRY_CHILD_TO_GROUP[_norm_key(_grp)] = _grp
    for _it in _items:
        INDUSTRY_CHILD_TO_GROUP[_norm_key(_it)] = _grp


def infer_industry_group(industry_value: str) -> str:
    """Suy ra Trường/ngành nghề từ giá trị 'Lĩnh vực/ngành nghề' trong Excel."""
    k = _norm_key(industry_value)
    return INDUSTRY_CHILD_TO_GROUP.get(k, "-")


# ==========================================
# 1. TẢI DỮ LIỆU TỪ EXCEL
# ==========================================
def load_data(folder_path, json_filename='ketban.json'):
    excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    excel_files = [f for f in excel_files if not os.path.basename(f).startswith('~$')]

    if not excel_files:
        print("Lỗi: Không tìm thấy file Excel (.xlsx) trong E:\\ttnt")
        return None, {}, [], {}

    data_file = excel_files[0]
    print(f"--- Đang nạp dữ liệu: {os.path.basename(data_file)} ---")

    try:
        df = pd.read_excel(data_file, engine='openpyxl')
        json_path = os.path.join(folder_path, json_filename)
        with open(json_path, 'r', encoding='utf-8') as f:
            full_json = json.load(f)
            return (df,
                    full_json.get('locations', {}),
                    full_json.get('bonus_config', []),
                    full_json.get('interest_groups', {}))
    except Exception as e:
        print(f"Lỗi hệ thống: {e}")
        return None, {}, [], {}


# ==========================================
# 2. ĐỐI TƯỢNG NGƯỜI DÙNG
# ==========================================
class User:
    def __init__(self, uid, name, dob, gender, location, interests, industry, marital, friends_str):
        def clean(val):
            if pd.isna(val) or str(val).strip() in ["", "nan", "-"]:
                return "-"
            return str(val).strip()

        self.id = str(uid)
        self.name = clean(name).title()
        self.dob = clean(dob)
        self.gender = clean(gender).title()
        self.location = clean(location).title()

        # Ngành nghề con (đọc từ Excel / input)
        self.industry = clean(industry).title()
        # Trường/ngành nghề suy ra từ mapping
        self.industry_group = infer_industry_group(self.industry)

        self.marital = clean(marital).title()

        self.interests = [x.strip().title() for x in str(interests).split(';') if x.strip()] if clean(interests) != "-" else []
        self.friends_ids = [x.strip() for x in str(friends_str).split(',') if x.strip().isdigit()] if clean(friends_str) != "-" else []

    @classmethod
    def from_row(cls, row):
        return cls(
            row['Số thứ tự'], row['Họ và tên'], row['Ngày sinh'], row['Giới tính'],
            row['Nơi ở'], row['Sở thích'], row.get('Lĩnh vực/ngành nghề', '-'),
            row.get('Tình trạng hôn nhân', '-'), row.get('Bạn chung (ID)', '')
        )


class SocialGraph:
    def __init__(self, users, loc_map, bonus_rules, interest_groups):
        self.users = {u.id: u for u in users}

        # adj_list: dùng để DUYỆT BFS/DFS/A* (có thể chứa liên kết gợi ý NEW_USER)
        self.adj_list = {u.id: set(u.friends_ids) for u in users}

        # friend_adj: CHỈ bạn bè thật từ Excel (để tính "bạn chung" chính xác)
        self.friend_adj = {u.id: set(u.friends_ids) for u in users}

        # strong_neighbors: "bạn proxy" tự động cho NEW_USER (không cần nhập)
        self.strong_neighbors = {}

        self.loc_map = loc_map
        self.bonus_rules = bonus_rules

        # Gộp bảng sở thích JSON (nếu có) với bảng mặc định
        merged_groups = {k: list(v) for k, v in DEFAULT_INTEREST_GROUPS.items()}
        for g, items in (interest_groups or {}).items():
            if g not in merged_groups:
                merged_groups[g] = list(items)
            else:
                merged_groups[g] = list(dict.fromkeys(merged_groups[g] + list(items)))
        self.interest_groups = merged_groups

        # Dạng chuẩn hoá để so khớp nhanh
        self._interest_groups_norm = {g: {_norm_key(x) for x in items} for g, items in self.interest_groups.items()}

    def _proxy_friend_set(self, uid: str) -> set:
        """
        Tập dùng để tính "bạn chung":
        - Nếu là NEW_USER: dùng strong_neighbors[NEW_USER] (tự động, không cần nhập)
        - Nếu là user thường: dùng friend_adj (bạn thật từ Excel)
        """
        if uid in self.strong_neighbors:
            return self.strong_neighbors.get(uid, set())
        return self.friend_adj.get(uid, set())

    def add_new_user(self, new_user):
        """
        Thêm NEW_USER + tạo liên kết gợi ý để BFS/DFS có candidate.
        Đồng thời tạo strong_neighbors (tự động) để tính "bạn chung" mà KHÔNG cần nhập.
        """
        self.users[new_user.id] = new_user
        self.adj_list[new_user.id] = set()

        # NEW_USER không có bạn thật trong Excel => friend_adj rỗng
        self.friend_adj[new_user.id] = set()

        # 1) Tạo liên kết gợi ý (loose) để duyệt candidate
        new_loc_val = self.loc_map.get(new_user.location)
        for uid, u in self.users.items():
            if uid == new_user.id:
                continue

            conn = False
            if new_user.location != "-" and new_user.location == u.location:
                conn = True
            elif new_loc_val is not None and self.loc_map.get(u.location) == new_loc_val:
                conn = True
            elif set(new_user.interests) & set(u.interests):
                conn = True
            elif new_user.industry_group != "-" and new_user.industry_group == u.industry_group:
                conn = True

            if conn:
                self.adj_list[new_user.id].add(uid)
                self.adj_list[uid].add(new_user.id)

        # 2) Tạo "strong_neighbors" (strict) để tính bạn chung, tránh bị dính hàng loạt
        strong = set()
        new_int_norm = {_norm_key(x) for x in new_user.interests}

        for uid, u in self.users.items():
            if uid == new_user.id:
                continue

            u_int_norm = {_norm_key(x) for x in u.interests}
            common_int = new_int_norm & u_int_norm

            is_strong = False
            # Cùng nơi ở (mạnh)
            if new_user.location != "-" and new_user.location == u.location:
                is_strong = True
            # Cùng trường/ngành nghề (mạnh)
            elif new_user.industry_group != "-" and new_user.industry_group == u.industry_group:
                is_strong = True
            # Trùng >= 2 sở thích (mạnh)
            elif len(common_int) >= 2:
                is_strong = True

            if is_strong:
                strong.add(uid)

        self.strong_neighbors[new_user.id] = strong

    def calculate_score(self, user_a, user_b):
        """
        3.3.2. Quy tắc cộng điểm (MỚI)

        - Trùng nơi ở: +1
        - Có ít nhất 1 bạn chung: +1
        - Sở thích:
            + Trùng sở thích (trong trường sở thích) → +2 điểm / sở thích
            + Trùng trường sở thích → +1 điểm
              (cùng 1 nhóm nhưng không trùng sở thích con trong nhóm đó)
        - Ngành nghề:
            + Trùng ngành nghề (trong trường ngành nghề) → +2 điểm
              (ngành nghề con trùng nhau)
            + Trùng trường ngành nghề → +1 điểm
              (trường/ngành nghề trùng nhau, khi không trùng ngành con)
        """
        score = 0

        # 1) Trùng nơi ở (+1)
        if user_a.location != "-" and user_a.location == user_b.location:
            score += 1

        # 2) Có ít nhất 1 bạn chung (+1) - dùng proxy friend set (tự động cho NEW_USER)
        if self._proxy_friend_set(user_a.id) & self._proxy_friend_set(user_b.id):
            score += 1

        # 3) Sở thích: trùng sở thích con (+2 / sở thích)
        a_int = {_norm_key(x) for x in user_a.interests}
        b_int = {_norm_key(x) for x in user_b.interests}
        common = a_int & b_int
        score += len(common) * 2

        # 4) Sở thích: trùng trường sở thích (+1)
        # Cùng nhóm nhưng KHÔNG trùng sở thích con trong chính nhóm đó.
        for members_norm in self._interest_groups_norm.values():
            a_in = a_int & members_norm
            b_in = b_int & members_norm
            if a_in and b_in and not (a_in & b_in):
                score += 1
                break

        # 5) Ngành nghề:
        # +2 nếu trùng ngành nghề con (industry) (và industry không phải là tên group)
        a_ind = _norm_key(user_a.industry)
        b_ind = _norm_key(user_b.industry)

        if a_ind != "-" and a_ind == b_ind and a_ind not in INDUSTRY_GROUP_KEYS_NORM:
            score += 2
        else:
            # +1 nếu trùng trường/ngành nghề (industry_group)
            if user_a.industry_group != "-" and user_a.industry_group == user_b.industry_group:
                score += 1

        return score


# ==========================================
# 3. THUẬT TOÁN (BFS, DFS, A*)
# ==========================================
def run_bfs(graph, start_id):
    results = []
    queue = deque([start_id])
    visited = {start_id}
    while queue:
        curr = queue.popleft()
        if curr != start_id:
            s = graph.calculate_score(graph.users[start_id], graph.users[curr])
            if s > 0:
                results.append({'user': graph.users[curr], 'score': s})
        for n in graph.adj_list.get(curr, []):
            if n in graph.users and n not in visited:
                visited.add(n)
                queue.append(n)
    return results


def run_dfs(graph, start_id, max_depth=3):
    results = []
    stack = [(start_id, 0)]
    visited = {start_id}
    while stack:
        curr, depth = stack.pop()
        if curr != start_id:
            s = graph.calculate_score(graph.users[start_id], graph.users[curr])
            if s > 0:
                results.append({'user': graph.users[curr], 'score': s})
        if depth < max_depth:
            for n in graph.adj_list.get(curr, []):
                if n in graph.users and n not in visited:
                    visited.add(n)
                    stack.append((n, depth + 1))
    return results


def run_astar(graph, start_id, goal_id):
    open_set = [(0, start_id, [start_id])]
    visited = set()
    while open_set:
        f, curr, path = heapq.heappop(open_set)
        if curr == goal_id:
            return path
        if curr in visited:
            continue
        visited.add(curr)
        for n in graph.adj_list.get(curr, []):
            if n in graph.users and n not in visited:
                heapq.heappush(open_set, (len(path), n, path + [n]))
    return None


# ==========================================
# 4. HIỂN THỊ
# ==========================================
def display_profile(u, label, me_id, graph, score, show_score=True):
    # Bạn chung hiển thị theo đúng tập đang dùng để chấm điểm
    common_ids = graph._proxy_friend_set(me_id).intersection(graph._proxy_friend_set(u.id))
    common_names = [graph.users[cid].name for cid in common_ids if cid in graph.users]

    print(f"\n{label}. {u.name.upper()} (+{score})")
    print(f"Ngày sinh: {u.dob}")
    print(f"Giới tính: {u.gender}")
    print(f"Nơi ở: {u.location}")
    print(f"Ngành nghề: {u.industry} (Trường: {u.industry_group})")
    print(f"Sở thích: {', '.join(u.interests) if u.interests else '-'} ")
    print(f"Tình trạng hôn nhân: {u.marital}")
    print(f"Bạn chung: {', '.join(common_names) if common_names else '-'} ")
    print("-" * 45)


def get_input():
    print("-" * 50)
    print("   NHẬP THÔNG TIN CỦA BẠN ")
    print("-" * 50)
    n = input("1. Họ và tên *: ").strip() or "-"
    d = input("2. Ngày sinh *: ").strip() or "-"
    g = input("3. Giới tính *: ").strip() or "-"
    l = input("4. Nơi ở *: ").strip() or "-"
    ind = input("5. Ngành nghề: ").strip() or "-"
    its = input("6. Sở thích (cách nhau bởi ;)*: ").strip() or "-"
    m = input("7. Tình trạng hôn nhân *: ").strip() or "-"
    # KHÔNG CẦN mục 8
    return User("NEW_USER", n, d, g, l, its, ind, m, "")


def main():
    path = r"E:\ttnt"
    df, l_m, b_r, i_g = load_data(path)
    if df is None:
        return

    users = [User.from_row(r) for _, r in df.iterrows()]
    graph = SocialGraph(users, l_m, b_r, i_g)

    me = get_input()
    graph.add_new_user(me)

    start_exec = time.time()
    bfs_res = run_bfs(graph, me.id)
    dfs_res = run_dfs(graph, me.id)

    combined = {c['user'].id: c for c in bfs_res + dfs_res}.values()
    top_30_all = sorted(combined, key=lambda x: x['score'], reverse=True)[:30]

    print("\n" + "*" * 60 + "\n DANH SÁCH TOP 30 NGƯỜI ĐÃ LỌC\n" + "*" * 60)
    for i, c in enumerate(top_30_all):
        display_profile(c['user'], i + 1, me.id, graph, c['score'], True)

    if top_30_all:
        top_1 = top_30_all[0]
        print("\n" + "!" * 60 + "\n          GỢI Ý PHÙ HỢP NHẤT (TOP-1)\n" + "!" * 60)
        display_profile(top_1['user'], "TOP-1", me.id, graph, top_1['score'], True)

        print("\n CHI PHÍ/ĐƯỜNG ĐI ĐẾN TOP 1 (A*)")
        path_astar = run_astar(graph, me.id, top_1['user'].id)
        if path_astar:
            print(" -> ".join([graph.users[p].name for p in path_astar]))
        else:
            print("Không tìm thấy đường đi.")

    print(f"\n THỜI GIAN THỰC THI : {time.time() - start_exec:.4f} giây")

    print("\n" + "=" * 60 + "\n DANH SÁCH TOP 30 LỌC TỪ BFS \n" + "=" * 60)
    for i, c in enumerate(sorted(bfs_res, key=lambda x: x['score'], reverse=True)[:30]):
        display_profile(c['user'], i + 1, me.id, graph, c['score'], True)

    print("\n" + "=" * 60 + "\n DANH SÁCH TOP 30 LỌC TỪ DFS \n" + "=" * 60)
    for i, c in enumerate(sorted(dfs_res, key=lambda x: x['score'], reverse=True)[:30]):
        display_profile(c['user'], i + 1, me.id, graph, c['score'], True)

    print(f"\n THỜI GIAN THỰC THI TỔNG CỘNG: {time.time() - start_exec:.4f} giây")


if __name__ == "__main__":
    main()
