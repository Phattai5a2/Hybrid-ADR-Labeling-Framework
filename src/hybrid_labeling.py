
import os, pandas as pd, torch, html, re, time, warnings
from transformers import pipeline
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score

warnings.filterwarnings("ignore")

# ====================== CẤU HÌNH ======================
DEVICE = 0 if torch.cuda.is_available() else -1
BATCH_SIZE = 128 
MODEL_NAME = "sileod/deberta-v3-large-tasksource-nli"

print(f"KHỞI CHẠY V10.5 - FINAL REVISED")
print("-" * 85)


sider_dict = {}
try:
    if os.path.exists('meddra_all_se.tsv') and os.path.exists('drug_names.tsv'):
        df_se = pd.read_csv('meddra_all_se.tsv', sep='\t', header=None, usecols=[0, 5], names=['cid', 'adr'], dtype=str)
        df_name = pd.read_csv('drug_names.tsv', sep='\t', header=None, names=['cid', 'drug_name'], dtype=str)
        df_sider = pd.merge(df_se, df_name, on='cid')
        temp_dict = df_sider.groupby('drug_name')['adr'].apply(list).to_dict()
        for d, adrs in temp_dict.items():
            filtered = [re.escape(a.lower()) for a in adrs if len(a) > 4]
            if filtered:
                sider_dict[d.lower()] = re.compile(r'\b(' + '|'.join(filtered) + r')\b', re.I)
        print(f"SIDER Regex Ready: {len(sider_dict)} drugs")
except Exception as e:
    print(f"Lỗi SIDER: {e}")

# ====================== KHỞI TẠO MODEL ======================
zero_shot_pipe = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME,
    device=DEVICE,
    batch_size=BATCH_SIZE,
    model_kwargs={"torch_dtype": torch.float16} if DEVICE == 0 else {}
)

# ====================== ĐỌC DỮ LIỆU ======================
df_all = pd.read_csv('drugsCom_ALL_215k.csv', low_memory=False)
df_all['review_clean'] = df_all['review'].fillna("").apply(html.unescape)

# ====================== RULES TỐI ƯU ======================
candidate_labels = ["side effects", "no side effects"]
reg_no_side = re.compile(r"no (side effects?|problems?|issues?|complaints?)", re.I)
reg_contrast = re.compile(r"\b(but|however|although|yet|still)\b", re.I)

results = [0] * len(df_all)
start_time = time.time()

# Chuyển dữ liệu sang List để loop nhanh hơn
reviews = df_all['review_clean'].tolist()
drug_names = df_all['drugName'].fillna("").str.lower().tolist()

for i in tqdm(range(0, len(reviews), BATCH_SIZE), desc="⚡ Processing"):
    end_idx = min(i + BATCH_SIZE, len(reviews))
    batch_revs = reviews[i:end_idx]
    batch_drugs = drug_names[i:end_idx]
    
    ai_queue_texts, ai_queue_indices, ai_queue_sider = [], [], []
    
    for j, (rev, drug) in enumerate(zip(batch_revs, batch_drugs)):
        g_idx = i + j
        rev_low = rev.lower()
        
        # Rule 1: No side effects (Skip AI)
        if reg_no_side.search(rev_low) and not reg_contrast.search(rev_low):
            continue
            
        # SIDER Check
        has_sider = drug in sider_dict and sider_dict[drug].search(rev_low)
        
        # Thêm vào hàng đợi AI
        ai_queue_texts.append(rev[:700]) # Cắt bớt độ dài để tăng tốc
        ai_queue_indices.append(g_idx)
        ai_queue_sider.append(has_sider)
    
    if ai_queue_texts:
        outputs = zero_shot_pipe(
            ai_queue_texts,
            candidate_labels=candidate_labels,
            hypothesis_template="The patient experienced {}." # Template tối ưu cho ADR
        )
        
        for k, out in enumerate(outputs):
            # Lấy score của nhãn "side effects"
            score = out['scores'][0] if out['labels'][0] == "side effects" else out['scores'][1]
            
            # Ngưỡng động (Dynamic Threshold)
            threshold = 0.42 if ai_queue_sider[k] else 0.58
            results[ai_queue_indices[k]] = 1 if score >= threshold else 0

# ====================== ĐÁNH GIÁ & FIX LỖI INDEX ======================
df_all['y_ai_v10.5'] = results
df_all.to_csv('drugsCom_V10.5_Final.csv', index=False)

if os.path.exists('Phieu_Khao_Sat_Chuyen_Gia.xlsx'):
    df_expert = pd.read_excel('Phieu_Khao_Sat_Chuyen_Gia.xlsx')
    
    # Ép ID về string để khớp tuyệt đối
    df_all['uniqueID_str'] = df_all['uniqueID'].astype(str).str.replace('.0', '')
    df_expert['ID_str'] = df_expert['Mã Đánh Giá'].astype(str).str.replace('.0', '')
    
    # Khớp dữ liệu bằng Merge (Chống lệch dòng)
    df_compare = pd.merge(df_expert, df_all[['uniqueID_str', 'y_ai_v10.5']], 
                          left_on='ID_str', right_on='uniqueID_str', how='inner')
    
    y_true = df_compare['Nhãn Chuyên Gia (1=Có, 0=Không)'].astype(int)
    y_pred = df_compare['y_ai_v10.5'].astype(int)
    
    print("\n" + "="*60)
    print("KẾT QUẢ V10.5 OPTIMIZED")
    print(classification_report(y_true, y_pred, digits=4))
    print(f"Tổng thời gian: {(time.time()-start_time)/60:.2f} phút")
    print("="*60)