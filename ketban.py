import pandas as pd
import numpy as np
import heapq
import json
import os
import glob
from collections import deque
import re
import time
import unicodedata

# ==========================================
# 0. CHUẨN HOÁ DỮ LIỆU (KHÔNG ĐỤNG EXCEL)
# Theo tiêu chí 3.2.4:
# 1) Nơi ở: HN -> Hà Nội, HCM/Ho Chi Minh -> Thành phố Hồ Chí Minh, ...
# 2) Ngành nghề theo trường: chuẩn hoá ngành con + suy ra trường/ngành nghề
# 3) Sở thích theo trường: chuẩn hoá sở thích con + suy ra trường sở thích
# 4) Bạn chung: chuẩn hoá theo ID (số), bỏ trùng
# ==========================================

def _strip_accents(s: str) -> str:
    """Bỏ dấu tiếng Việt để so khớp nhập liệu thiếu dấu."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)

def _norm_key(s: str) -> str:
    """Chuẩn hoá chuỗi để so khớp (không phân biệt hoa/thường, chuẩn hoá dấu gạch, khoảng trắng)."""
    if s is None:
        return ""
    s = str(s).strip()
    if s in ["", "nan", "NaN", "-"]:
        return "-"
    s = s.replace("—", "-").replace("–", "-").replace("−", "-")
    s = re.sub(r"\s+", " ", s)
    return s.casefold()

def _norm_key_ascii(s: str) -> str:
    """Key so khớp mạnh: bỏ dấu + chuẩn hoá."""
    return _norm_key(_strip_accents(s))

def _loc_simplify_ascii(s: str) -> str:
    """Chuẩn hoá key nơi ở: bỏ dấu + bỏ tiền tố tp/thành phố/tỉnh để map 'Hồ Chí Minh' -> 'Thành phố Hồ Chí Minh'."""
    k = _norm_key_ascii(s)
    if k == "-":
        return "-"
    k = k.replace(".", "")
    k = re.sub(r"\b(thanh pho|tp|tinh)\b", "", k)
    k = re.sub(r"\s+", " ", k).strip()
    return k

# ---- BẢNG TRƯỜNG/NGÀNH NGHỀ -> NGÀNH NGHỀ CON (CHUẨN) ----
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

# ---- BẢNG SỞ THÍCH -> SỞ THÍCH CON (CHUẨN) ----
DEFAULT_INTEREST_GROUPS = {
    "Sáng tạo": ["Vẽ tranh", "Chụp ảnh", "Viết lách", "Làm đồ thủ công"],
    "Giải trí": ["Nghe nhạc", "Xem phim", "Chơi nhạc cụ", "Chơi game"],
    "Vận động": ["Tập gym", "Yoga", "Chạy bộ", "Đi bộ", "Đạp xe", "Bơi lội"],
    "Thư giãn": ["Đọc sách", "Thiền định", "Làm vườn", "Nấu ăn", "Làm bánh"],
    "Khám phá": ["Du lịch", "Học ngoại ngữ", "Khám phá ẩm thực", "Tham gia hoạt động tình nguyện"],
}

# ---- Alias tối thiểu để đáp ứng 3.2.4 (có thể mở rộng thêm) ----
LOCATION_ALIASES = {
    "hn": "Hà Nội",
    "ha noi": "Hà Nội",
    "hanoi": "Hà Nội",

    "hcm": "Thành phố Hồ Chí Minh",
    "hcmc": "Thành phố Hồ Chí Minh",
    "tp hcm": "Thành phố Hồ Chí Minh",
    "tphcm": "Thành phố Hồ Chí Minh",
    "tp.hcm": "Thành phố Hồ Chí Minh",
    "ho chi minh": "Thành phố Hồ Chí Minh",
    "hồ chí minh": "Thành phố Hồ Chí Minh",
    "sai gon": "Thành phố Hồ Chí Minh",
    "saigon": "Thành phố Hồ Chí Minh",
    "sg": "Thành phố Hồ Chí Minh",
}

INDUSTRY_CHILD_ALIASES = {
    "it": "Công nghệ thông tin",
    "cntt": "Công nghệ thông tin",
    "cong nghe thong tin": "Công nghệ thông tin",

    "ktpm": "Kỹ thuật phần mềm",
    "software": "Kỹ thuật phần mềm",
    "software engineering": "Kỹ thuật phần mềm",

    "ai": "Trí tuệ nhân tạo",
    "tri tue nhan tao": "Trí tuệ nhân tạo",

    "attt": "An ninh mạng",
    "cyber security": "An ninh mạng",
    "security": "An ninh mạng",

    "ds": "Khoa học dữ liệu",
    "data science": "Khoa học dữ liệu",
    "khoa hoc du lieu": "Khoa học dữ liệu",
}

INTEREST_CHILD_ALIASES = {
    "chay": "Chạy bộ",
    "chay bo": "Chạy bộ",
    "run": "Chạy bộ",
    "jogging": "Chạy bộ",

    "gym": "Tập gym",
    "tap gym": "Tập gym",

    "game": "Chơi game",
    "choi game": "Chơi game",
    "gaming": "Chơi game",

    "nhac": "Nghe nhạc",
    "nghe nhac": "Nghe nhạc",

    "phim": "Xem phim",
    "xem phim": "Xem phim",

    "anh": "Chụp ảnh",
    "chup anh": "Chụp ảnh",
    "photo": "Chụp ảnh",

    "doc": "Đọc sách",
    "doc sach": "Đọc sách",
}

# ---- Tạo bảng tra nhanh: ngành nghề con -> trường/ngành nghề ----
INDUSTRY_CHILD_TO_GROUP = {}
for _grp, _items in DEFAULT_INDUSTRY_GROUPS.items():
    INDUSTRY_CHILD_TO_GROUP[_norm_key(_grp)] = _grp
    INDUSTRY_CHILD_TO_GROUP[_norm_key_ascii(_grp)] = _grp
    for _it in _items:
        INDUSTRY_CHILD_TO_GROUP[_norm_key(_it)] = _grp
        INDUSTRY_CHILD_TO_GROUP[_norm_key_ascii(_it)] = _grp

def infer_industry_group(industry_value: str) -> str:
    k = _norm_key(industry_value)
    ka = _norm_key_ascii(industry_value)
    return INDUSTRY_CHILD_TO_GROUP.get(k) or INDUSTRY_CHILD_TO_GROUP.get(ka) or "-"


class DataNormalizer:
    """Chuẩn hoá theo tiêu chí 3.2.4."""

    def __init__(self, known_locations=None):
        self.known_locations = set(known_locations or [])

        self._loc_lookup = {}
        for loc in self.known_locations:
            self._loc_lookup[_norm_key(loc)] = loc
            self._loc_lookup[_norm_key_ascii(loc)] = loc
            self._loc_lookup[_loc_simplify_ascii(loc)] = loc

        for k, v in LOCATION_ALIASES.items():
            self._loc_lookup[_norm_key(k)] = v
            self._loc_lookup[_norm_key_ascii(k)] = v
            self._loc_lookup[_loc_simplify_ascii(k)] = v

        self._interest_canon = {}
        for items in DEFAULT_INTEREST_GROUPS.values():
            for it in items:
                self._interest_canon[_norm_key_ascii(it)] = it

        self._industry_canon = {}
        for items in DEFAULT_INDUSTRY_GROUPS.values():
            for it in items:
                self._industry_canon[_norm_key_ascii(it)] = it
        for grp in DEFAULT_INDUSTRY_GROUPS.keys():
            self._industry_canon[_norm_key_ascii(grp)] = grp

    def normalize_location(self, raw: str) -> str:
        if raw is None:
            return "-"
        raw = str(raw).strip()
        if raw in ["", "nan", "NaN", "-"]:
            return "-"
        for key in (_norm_key(raw), _norm_key_ascii(raw), _loc_simplify_ascii(raw)):
            if key in self._loc_lookup:
                return self._loc_lookup[key]
        return raw.title()

    def normalize_industry_child(self, raw: str) -> str:
        if raw is None:
            return "-"
        raw = str(raw).strip()
        if raw in ["", "nan", "NaN", "-"]:
            return "-"
        k = _norm_key_ascii(raw)
        if k in INDUSTRY_CHILD_ALIASES:
            return INDUSTRY_CHILD_ALIASES[k]
        if k in self._industry_canon:
            return self._industry_canon[k]
        return raw.title()

    def normalize_interest_child(self, raw: str) -> str:
        if raw is None:
            return "-"
        raw = str(raw).strip()
        if raw in ["", "nan", "NaN", "-"]:
            return "-"
        k = _norm_key_ascii(raw)
        if k in INTEREST_CHILD_ALIASES:
            return INTEREST_CHILD_ALIASES[k]
        if k in self._interest_canon:
            return self._interest_canon[k]
        return raw.title()

    def normalize_friend_ids(self, raw: str):
        if raw is None:
            return []
        s = str(raw).strip()
        if s in ["", "nan", "NaN", "-"]:
            return []
        parts = [p.strip() for p in s.split(",")]
        ids = [p for p in parts if p.isdigit()]
        seen, out = set(), []
        for x in ids:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out


NORMALIZER = None


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

        raw_loc = clean(location)
        self.location = NORMALIZER.normalize_location(raw_loc) if NORMALIZER else raw_loc.title()

        raw_ind = clean(industry)
        self.industry = NORMALIZER.normalize_industry_child(raw_ind) if NORMALIZER else raw_ind.title()
        self.industry_group = infer_industry_group(self.industry)

        self.marital = clean(marital).title()

        raw_its = clean(interests)
        if raw_its == "-":
            self.interests = []
        else:
            tokens = [t.strip() for t in re.split(r"[;,]", str(raw_its)) if t.strip()]
            self.interests = [NORMALIZER.normalize_interest_child(t) if NORMALIZER else t.title() for t in tokens]

        self.friends_ids = NORMALIZER.normalize_friend_ids(friends_str) if NORMALIZER else \
            [x.strip() for x in str(friends_str).split(',') if x.strip().isdigit()]

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
        self.friend_adj = {u.id: set(u.friends_ids) for u in users}
        self.adj_list = {u.id: set(u.friends_ids) for u in users}
        self.strong_neighbors = {}

        self.loc_map = loc_map
        self.bonus_rules = bonus_rules

        merged_groups = {k: list(v) for k, v in DEFAULT_INTEREST_GROUPS.items()}
        for g, items in (interest_groups or {}).items():
            if g not in merged_groups:
                merged_groups[g] = list(items)
            else:
                merged_groups[g] = list(dict.fromkeys(merged_groups[g] + list(items)))
        self.interest_groups = merged_groups
        self._interest_groups_norm = {g: {_norm_key(x) for x in items} for g, items in self.interest_groups.items()}

    def _proxy_friend_set(self, uid: str) -> set:
        if uid in self.strong_neighbors:
            return self.strong_neighbors.get(uid, set())
        return self.friend_adj.get(uid, set())

    def common_friend_ids(self, id_a: str, id_b: str) -> set:
        return self._proxy_friend_set(id_a) & self._proxy_friend_set(id_b)

    def add_new_user(self, new_user):
        self.users[new_user.id] = new_user
        self.friend_adj[new_user.id] = set()
        self.adj_list[new_user.id] = set()

        new_loc_val = self.loc_map.get(new_user.location)
        new_int_set = {_norm_key(x) for x in new_user.interests}

        # Liên kết gợi ý để có candidate
        for uid, u in self.users.items():
            if uid == new_user.id:
                continue

            conn = False
            if new_user.location != "-" and new_user.location == u.location:
                conn = True
            elif new_loc_val is not None and self.loc_map.get(u.location) == new_loc_val:
                conn = True
            else:
                u_int_set = {_norm_key(x) for x in u.interests}
                if new_int_set & u_int_set:
                    conn = True
                elif new_user.industry_group != "-" and new_user.industry_group == u.industry_group:
                    conn = True

            if conn:
                self.adj_list[new_user.id].add(uid)
                self.adj_list[uid].add(new_user.id)

        # Strong neighbors để tính bạn chung tự động (giảm bị +1 hàng loạt)
        strong = set()
        for uid, u in self.users.items():
            if uid == new_user.id:
                continue

            u_int_set = {_norm_key(x) for x in u.interests}
            common_int = new_int_set & u_int_set

            is_strong = False
            if new_user.location != "-" and new_user.location == u.location:
                is_strong = True
            elif new_user.industry_group != "-" and new_user.industry_group == u.industry_group:
                is_strong = True
            elif len(common_int) >= 2:
                is_strong = True

            if is_strong:
                strong.add(uid)

        self.strong_neighbors[new_user.id] = strong

    def calculate_score(self, user_a, user_b):
        score = 0

        # Trùng nơi ở: +1
        if user_a.location != "-" and user_a.location == user_b.location:
            score += 1

        # Có ít nhất 1 bạn chung: +1
        if self.common_friend_ids(user_a.id, user_b.id):
            score += 1

        # Sở thích: +2 / sở thích trùng
        a_int = {_norm_key(x) for x in user_a.interests}
        b_int = {_norm_key(x) for x in user_b.interests}
        common = a_int & b_int
        score += len(common) * 2

        # Trùng trường sở thích: +1 (cùng nhóm nhưng không trùng sở thích con trong nhóm đó)
        for members_norm in self._interest_groups_norm.values():
            a_in = a_int & members_norm
            b_in = b_int & members_norm
            if a_in and b_in and not (a_in & b_in):
                score += 1
                break

        # Ngành nghề: +2 nếu trùng ngành con, else +1 nếu trùng trường/ngành nghề
        a_ind = _norm_key(user_a.industry)
        b_ind = _norm_key(user_b.industry)

        if a_ind != "-" and a_ind == b_ind and a_ind not in INDUSTRY_GROUP_KEYS_NORM:
            score += 2
        else:
            if user_a.industry_group != "-" and user_a.industry_group == user_b.industry_group:
                score += 1

        return score


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


def display_profile(u, label, me_id, graph, score):
    common_ids = graph.common_friend_ids(me_id, u.id)
    common_names = [graph.users[cid].name for cid in common_ids if cid in graph.users]

    print(f"\n{label}. {u.name.upper()} (+{score})")
    print(f"Ngày sinh: {u.dob}")
    print(f"Giới tính: {u.gender}")
    print(f"Nơi ở: {u.location}")
    print(f"Ngành nghề: {u.industry} (Trường: {u.industry_group})")
    print(f"Sở thích: {', '.join(u.interests) if u.interests else '-'}")
    print(f"Tình trạng hôn nhân: {u.marital}")
    print(f"Bạn chung: {', '.join(common_names) if common_names else '-'}")
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
    its = input("6. Sở thích (cách nhau bởi ; hoặc ,)*: ").strip() or "-"
    m = input("7. Tình trạng hôn nhân *: ").strip() or "-"
    return User("NEW_USER", n, d, g, l, its, ind, m, "")


def main():
    global NORMALIZER

    path = r"E:\ttnt"
    df, l_m, b_r, i_g = load_data(path)
    if df is None:
        return

    known_locations = df['Nơi ở'].dropna().astype(str).unique().tolist()
    NORMALIZER = DataNormalizer(known_locations=known_locations)

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
        display_profile(c['user'], i + 1, me.id, graph, c['score'])

    if top_30_all:
        top_1 = top_30_all[0]
        print("\n" + "!" * 60 + "\n          GỢI Ý PHÙ HỢP NHẤT (TOP-1)\n" + "!" * 60)
        display_profile(top_1['user'], "TOP-1", me.id, graph, top_1['score'])

        print("\n CHI PHÍ/ĐƯỜNG ĐI ĐẾN TOP 1 (A*)")
        path_astar = run_astar(graph, me.id, top_1['user'].id)
        if path_astar:
            print(" -> ".join([graph.users[p].name for p in path_astar]))
        else:
            print("Không tìm thấy đường đi.")

    print(f"\n THỜI GIAN THỰC THI : {time.time() - start_exec:.4f} giây")

    print("\n" + "=" * 60 + "\n DANH SÁCH TOP 30 LỌC TỪ BFS \n" + "=" * 60)
    for i, c in enumerate(sorted(bfs_res, key=lambda x: x['score'], reverse=True)[:30]):
        display_profile(c['user'], i + 1, me.id, graph, c['score'])

    print("\n" + "=" * 60 + "\n DANH SÁCH TOP 30 LỌC TỪ DFS \n" + "=" * 60)
    for i, c in enumerate(sorted(dfs_res, key=lambda x: x['score'], reverse=True)[:30]):
        display_profile(c['user'], i + 1, me.id, graph, c['score'])

    print(f"\n THỜI GIAN THỰC THI TỔNG CỘNG: {time.time() - start_exec:.4f} giây")


if __name__ == "__main__":
    main()
