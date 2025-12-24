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
# 1. TẢI DỮ LIỆU
# ==========================================
def load_data(folder_path, json_filename='ketban.json'):
    extensions = ['*.csv', ['*.xlsx'], ['*.xls']]
    data_file = None
    for ext in ['*.csv', '*.xlsx', '*.xls']:
        files = glob.glob(os.path.join(folder_path, ext))
        files = [f for f in files if not os.path.basename(f).startswith('~$') and not f.endswith('.json')]
        if files:
            data_file = max(files, key=os.path.getsize)
            break
    
    if not data_file: return None, {}, []
    try:
        df = pd.read_csv(data_file) if data_file.endswith('.csv') else pd.read_excel(data_file)
        json_path = os.path.join(folder_path, json_filename)
        with open(json_path, 'r', encoding='utf-8') as f:
            full_json = json.load(f)
            return df, full_json.get('locations', {}), full_json.get('bonus_config', [])
    except: return None, {}, []

# ==========================================
# 2. ĐỐI TƯỢNG NGƯỜI DÙNG
# ==========================================
class User:
    def __init__(self, uid, name, dob, gender, location, interests, industry, marital, friends_str):
        def clean(val):
            if pd.isna(val) or str(val).strip() in ["", "nan", "-"]: return "-"
            return str(val).strip()

        self.id = str(uid)
        self.name = clean(name).title()
        self.dob = clean(dob)
        self.gender = clean(gender).title()
        self.location = clean(location).title()
        self.industry = clean(industry).title()
        self.marital = clean(marital).title()
        
        self.interests = [x.strip().title() for x in str(interests).split(';') if x.strip()] if clean(interests) != "-" else []
        self.friends_ids = [x.strip() for x in str(friends_str).split(',') if x.strip().isdigit()] if clean(friends_str) != "-" else []

    @classmethod
    def from_row(cls, row):
        return cls(row['Số thứ tự'], row['Họ và tên'], row['Ngày sinh'], row['Giới tính'], 
                   row['Nơi ở'], row['Sở thích'], row.get('Lĩnh vực/ngành nghề', '-'), 
                   row.get('Tình trạng hôn nhân', '-'), row.get('Bạn chung (ID)', ''))

class SocialGraph:
    def __init__(self, users, loc_map, bonus_rules):
        self.users = {u.id: u for u in users}
        self.adj_list = {u.id: set(u.friends_ids) for u in users}
        
        # Định nghĩa trường sở thích (cần có từ dữ liệu)
        self.interest_categories = {
            'thể thao': ['bóng đá', 'bóng rổ', 'bơi lội', 'cầu lông'],
            'nghệ thuật': ['vẽ', 'âm nhạc', 'khiêu vũ', 'nhiếp ảnh'],
            'học tập': ['đọc sách', 'ngoại ngữ', 'lập trình', 'khoa học'],
            # ... thêm các trường khác
        }
    
    def get_interest_category(self, interest):
        """Xác định trường của một sở thích"""
        interest_lower = interest.lower()
        for category, interests_list in self.interest_categories.items():
            if interest_lower in [i.lower() for i in interests_list]:
                return category
        return None
    
    def calculate_metrics(self, user_a, user_b):
        score = 0
        
        # 1. Trùng nơi ở: +1
        if user_a.location != "-" and user_a.location == user_b.location:
            score += 1
        
        # 2. Trùng ngành nghề: +1
        if user_a.industry != "-" and user_a.industry == user_b.industry:
            score += 1
        
        # 3. Có bạn chung: +1
        common_friends = self.adj_list[user_a.id].intersection(self.adj_list[user_b.id])
        if common_friends:
            score += 1
        
        # 4. Xử lý sở thích
        interests_a = [i.lower() for i in user_a.interests]
        interests_b = [i.lower() for i in user_b.interests]
        
        # Trùng chính xác sở thích: +2 điểm/mỗi
        exact_matches = set(interests_a) & set(interests_b)
        score += len(exact_matches) * 2
        
        # Trùng trường sở thích: +1 điểm/mỗi trường
        categories_a = set()
        categories_b = set()
        
        for interest in interests_a:
            category = self.get_interest_category(interest)
            if category:
                categories_a.add(category)
        
        for interest in interests_b:
            category = self.get_interest_category(interest)
            if category:
                categories_b.add(category)
        
        # Loại bỏ các trường đã được tính trong exact_matches
        common_categories = categories_a & categories_b
        # Chỉ tính các trường có sở thích khác nhau (không trùng chính xác)
        for category in common_categories:
            # Lấy tất cả sở thích trong trường này của cả 2 người
            cat_interests_a = {i for i in interests_a if self.get_interest_category(i) == category}
            cat_interests_b = {i for i in interests_b if self.get_interest_category(i) == category}
            
            # Nếu có sở thích chung trong trường nhưng KHÔNG trùng chính xác
            if cat_interests_a != cat_interests_b:
                score += 1
        
        return score, len(exact_matches), (1 if (user_a.location == user_b.location and common_friends) else 0)

# ==========================================
# 3. THUẬT TOÁN
# ==========================================
def run_bfs(graph, start_id, limit=1000):
    candidates = []
    queue = deque([start_id]); visited = {start_id}
    while queue and len(candidates) < limit:
        curr = queue.popleft()
        if curr != start_id:
            s, ni, tb = graph.calculate_metrics(graph.users[start_id], graph.users[curr])
            if s > 0: candidates.append({'user': graph.users[curr], 'score': s, 'num_int': ni, 'tie_breaker': tb})
        for n in graph.adj_list.get(curr, []):
            if n in graph.users and n not in visited: visited.add(n); queue.append(n)
    return candidates

def run_dfs(graph, start_id, max_depth=3, limit=1000):
    candidates = []
    stack = [(start_id, 0)]; visited = {start_id}
    while stack and len(candidates) < limit:
        curr, depth = stack.pop()
        if curr != start_id:
            s, ni, tb = graph.calculate_metrics(graph.users[start_id], graph.users[curr])
            if s > 0: candidates.append({'user': graph.users[curr], 'score': s, 'num_int': ni, 'tie_breaker': tb})
        if depth < max_depth:
            for n in graph.adj_list.get(curr, []):
                if n in graph.users and n not in visited: visited.add(n); stack.append((n, depth + 1))
    return candidates

def run_astar(graph, start_id, goal_id):
    open_set = [(0, start_id, [start_id])]; visited = set()
    while open_set:
        f, curr, path = heapq.heappop(open_set)
        if curr == goal_id: return path
        if curr in visited: continue
        visited.add(curr)
        for n in graph.adj_list.get(curr, []):
            if n in graph.users and n not in visited: heapq.heappush(open_set, (len(path), n, path + [n]))
    return None

# ==========================================
# 4. HIỂN THỊ
# ==========================================
def display_user_profile(u, rank_label, me_id, graph, score):
    common_ids = graph.adj_list[me_id].intersection(graph.adj_list[u.id])
    common_names = [graph.users[cid].name for cid in common_ids if cid in graph.users]
    friends_display = ", ".join(common_names) if common_names else "-"
    
    # Định dạng yêu cầu: 1. NGUYỄN VĂN A (+5)
    print(f"\n{rank_label}. {u.name.upper()} (+{score})")
    print(f"Ngày sinh: {u.dob}")
    print(f"Giới tính: {u.gender}")
    print(f"Nơi ở: {u.location}")
    print(f"Ngành nghề: {u.industry}")
    print(f"Sở thích: {', '.join(u.interests) if u.interests else '-'}")
    print(f"Tình trạng hôn nhân: {u.marital}")
    print(f"Bạn chung: { {friends_display} }")
    print("-" * 40)

def get_input():
    print("-" * 50); print("   NHẬP THÔNG TIN CỦA BẠN "); print("-" * 50)
    n = input("1. Họ và tên *: ").strip() or "-"
    d = input("2. Ngày sinh *: ").strip() or "-"
    g = input("3. Giới tính *: ").strip() or "-"
    l = input("4. Nơi ở *: ").strip() or "-"
    ind = input("5. Ngành nghề: ").strip() or "-"
    its = input("6. Sở thích (cách nhau bởi ;)*: ").strip() or "-"
    m = input("7. Tình trạng hôn nhân *: ").strip() or "-"
    return User("NEW_USER", n, d, g, l, its, ind, m, "")

def main():
    path = r"E:\ttnt"
    df, l_m, b_r = load_data(path)
    if df is None: return
    users = [User.from_row(r) for _, r in df.iterrows()]
    graph = SocialGraph(users, l_m, b_r)
    me = get_input()
    graph.add_new_user(me)

    start_all = time.time()
    bfs_results = run_bfs(graph, me.id)
    dfs_results = run_dfs(graph, me.id)
    
    # KẾT HỢP BFS + DFS (Lọc trùng bằng ID)
    combined = {c['user'].id: c for c in bfs_results + dfs_results}.values()
    top_30_combined = sorted(combined, key=lambda x: (x['score'], x['num_int'], x['tie_breaker']), reverse=True)[:30]
    
    # 1. DANH SÁCH TOP 30 (KẾT HỢP)
    print("\n" + "*"*60)
    print(" DANH SÁCH TOP 30 NGƯỜI ĐÃ LỌC)")
    print("*"*60)
    for i, c in enumerate(top_30_combined):
        display_user_profile(c['user'], i+1, me.id, graph, c['score'])

    if top_30_combined:
        # GỢI Ý PHÙ HỢP NHẤT (TOP-1)
        top_1_data = top_30_combined[0]
        top_1_user = top_1_data['user']
        print("\n" + "!"*60)
        print("        GỢI Ý PHÙ HỢP NHẤT (TOP-1)")
        print("!"*60)
        display_user_profile(top_1_user, "TOP-1", me.id, graph, top_1_data['score'])

        # 2. ĐƯỜNG ĐI A*
        print("\n CHI PHÍ/ĐƯỜNG ĐI ĐẾN TOP 1 (A*)")
        path_ids = run_astar(graph, me.id, top_1_user.id)
        if path_ids:
            print(" -> ".join([graph.users[pid].name for pid in path_ids]))
        else: print("Không tìm thấy đường đi.")

    # 3. THỜI GIAN THỰC THI (KẾT HỢP)
    print(f"\n THỜI GIAN THỰC THI : {time.time() - start_all:.4f} giây")

    # 4. TOP 30 BFS
    print("\n" + "="*60)
    print(" DANH SÁCH TOP 30 LỌC TỪ BFS ")
    print("="*60)
    top_30_bfs = sorted(bfs_results, key=lambda x: (x['score'], x['num_int'], x['tie_breaker']), reverse=True)[:30]
    for i, c in enumerate(top_30_bfs):
        display_user_profile(c['user'], i+1, me.id, graph, c['score'])

    # 5. TOP 30 DFS
    print("\n" + "="*60)
    print(" DANH SÁCH TOP 30 LỌC TỪ DFS ")
    print("="*60)
    top_30_dfs = sorted(dfs_results, key=lambda x: (x['score'], x['num_int'], x['tie_breaker']), reverse=True)[:30]
    for i, c in enumerate(top_30_dfs):
        display_user_profile(c['user'], i+1, me.id, graph, c['score'])

    # 6. THỜI GIAN THỰC THI TỔNG CỘNG
    print(f"\n THỜI GIAN THỰC THI TỔNG CỘNG: {time.time() - start_all:.4f} giây")

if __name__ == "__main__":
    main()