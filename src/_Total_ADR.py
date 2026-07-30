
import os, pandas as pd, torch, html, re, time, warnings
from transformers import pipeline
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
warnings.filterwarnings("ignore")

# ====================== CAU HINH TOI UU (GIU NGUYEN) ======================
DEVICE = 0 if torch.cuda.is_available() else -1
BATCH_SIZE = 128
MODEL_NAME = "sileod/deberta-v3-large-tasksource-nli"

# --- MOI: cau hinh sliding-window + max-pooling (thay cho rev[:700]) ---
MAX_LEN = 512   # gioi han token moi window (da tru hao cho template/special tokens ben trong _split_into_windows)
STRIDE = 128    # so token chong lap giua 2 window lien tiep
RULE_OVERRIDE_CONTINUOUS_SCORE = 0.0   # diem AI gia dinh cho review bi rule bo qua (xem docstring o tren)

# --- MOI: file xuat rieng cho SA-ECS fusion (cung dinh dang voi Phan 10 cua ban grid-search) ---
OUTPUT_SA_ECS_FILENAME = "drugCom_ADR_Pscore.csv"

print("KHOI CHAY V10.4 SUPER-FAST - RTX 4070 Ti SUPER OPTIMIZED (+ sliding-window, + P_score_adr)")
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE==0 else 'CPU'}")
print("-" * 85)

# 1. NAP VA TOI UU HOA SIDER — GIU NGUYEN
sider_dict = {}
combined_pattern = None
print("Dang toi uu hoa bo tu dien SIDER/MedDRA...")
try:
    if os.path.exists('meddra_all_se.tsv'):
        df_se = pd.read_csv('meddra_all_se.tsv', sep='\t', header=None, usecols=[0, 5], names=['cid', 'adr'], dtype=str)
        df_se['adr'] = df_se['adr'].str.lower().str.strip()

        if os.path.exists('drug_names.tsv'):
            df_name = pd.read_csv('drug_names.tsv', sep='\t', header=None, names=['cid', 'drug_name'], dtype=str)
            df_sider = pd.merge(df_se, df_name, on='cid')
            temp_dict = df_sider.groupby('drug_name')['adr'].apply(list).to_dict()
            for d, adrs in temp_dict.items():
                filtered_adrs = [re.escape(a) for a in adrs if len(a) > 4]
                if filtered_adrs:
                    sider_dict[d.lower()] = re.compile(r'\b(' + '|'.join(filtered_adrs) + r')\b')
            print("Mode: Specific Drug-ADR (Regex-Optimized)")
        else:
            global_adrs = [re.escape(a) for a in df_se['adr'].unique() if len(a) > 4]
            combined_pattern = re.compile(r'\b(' + '|'.join(global_adrs) + r')\b')
            print("Mode: Global MedDRA (Regex-Optimized)")
except Exception as e:
    print(f"Loi toi uu tu dien: {e}")

# 2. KHOI TAO AI VOI FP16 (Mixed Precision) — GIU NGUYEN
print(f"Dang nap {MODEL_NAME} voi che do FP16...")
zero_shot_pipe = pipeline(
    "zero-shot-classification",
    model=MODEL_NAME,
    device=DEVICE,
    batch_size=BATCH_SIZE,
    model_kwargs={"torch_dtype": torch.float16},
    framework="pt"
)


def _split_into_windows(text: str, tokenizer, max_len: int = MAX_LEN, stride: int = STRIDE) -> list:
    """Chia 1 review dai thanh cac sliding window chong lap theo token, decode lai thanh text.
    Review ngan hon gioi han thi tra ve nguyen van (KHONG cat bot)."""
    safe_max_len = max_len - 32
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= safe_max_len:
        return [text]
    step = safe_max_len - stride
    windows = []
    start = 0
    while start < len(ids):
        end = min(start + safe_max_len, len(ids))
        windows.append(tokenizer.decode(ids[start:end], skip_special_tokens=True))
        if end == len(ids):
            break
        start += step
    return windows


TOKENIZER = zero_shot_pipe.tokenizer

# 3. DOC DU LIEU — GIU NGUYEN
print("Dang nap du lieu reviews...")
df_all = pd.read_csv('drugsCom_ALL_215k.csv', low_memory=False)
df_all['review_clean'] = df_all['review'].fillna("").apply(html.unescape)
reviews = df_all['review_clean'].tolist()
drug_names = df_all['drugName'].str.lower().tolist()

# ====================== XU LY SIEU TOC ======================
results = [0] * len(df_all)
scores_continuous = [RULE_OVERRIDE_CONTINUOUS_SCORE] * len(df_all)   # MOI: P_score_adr lien tuc
candidate_labels = ["experiencing side effects", "no side effects"]
start_time = time.time()

# Gom cac quy tac Regex tinh — GIU NGUYEN
reg_no_side = re.compile(r"no (side effects?|problems?|issues?|complaints?)", re.I)
reg_but = re.compile(r"\b(but|however|although|yet|still)\b", re.I)
reg_stop = re.compile(r"(miss|forget|stop|quit|discontinu)(ed|ing)? (a dose|the (pill|med|drug))", re.I)

for i in tqdm(range(0, len(reviews), BATCH_SIZE), desc="Processing Batch"):
    end_idx = min(i + BATCH_SIZE, len(reviews))
    batch_revs = reviews[i:end_idx]
    batch_drugs = drug_names[i:end_idx]

    # MOI: ai_queue_texts gio la danh sach WINDOW (khong phai 1 text/review), can map nguoc
    # ve "local index" (vi tri trong ai_queue_indices) de max-pool dung ve tung review.
    ai_queue_texts, ai_queue_owner = [], []
    ai_queue_indices, ai_queue_sider = [], []
    for j, (rev, drug) in enumerate(zip(batch_revs, batch_drugs)):
        g_idx = i + j
        rev_low = rev.lower()

        # Rule CPU (Rule logic) — GIU NGUYEN: review khop rule bi bo qua AI, mac dinh nhan = 0
        if reg_no_side.search(rev_low) and not reg_but.search(rev_low):
            continue
        if reg_stop.search(rev_low):
            continue
        # SIDER Check — GIU NGUYEN
        has_sider = False
        if combined_pattern:
            if combined_pattern.search(rev_low): has_sider = True
        elif drug in sider_dict:
            if sider_dict[drug].search(rev_low): has_sider = True

        local_idx = len(ai_queue_indices)
        ai_queue_indices.append(g_idx)
        ai_queue_sider.append(has_sider)

        windows = _split_into_windows(rev, TOKENIZER)
        ai_queue_texts.extend(windows)
        ai_queue_owner.extend([local_idx] * len(windows))

    # Inference Batch — 1 lan goi duy nhat cho toan bo window cua batch nay (GIU dung tinh
    # than "goi 1 lan / batch" cua code goc, chi khac la moi "phan tu" gio la 1 window)
    if ai_queue_texts:
        outputs = zero_shot_pipe(ai_queue_texts, candidate_labels=candidate_labels, hypothesis_template="This is {}.")
        window_scores = []
        for out in outputs:
            score = out['scores'][0] if out['labels'][0] == candidate_labels[0] else out['scores'][1]
            window_scores.append(score)

        # MOI: max-pooling theo local_idx -> dua ve dung review goc (Eq.17)
        n_local = len(ai_queue_indices)
        max_scores = [0.0] * n_local
        has_val = [False] * n_local
        for w_score, owner in zip(window_scores, ai_queue_owner):
            if not has_val[owner] or w_score > max_scores[owner]:
                max_scores[owner] = w_score
                has_val[owner] = True

        for k in range(n_local):
            g_idx = ai_queue_indices[k]
            score = max_scores[k]
            scores_continuous[g_idx] = score
            threshold = 0.42 if ai_queue_sider[k] else 0.50
            results[g_idx] = 1 if score >= threshold else 0

# ====================== DANH GIA VA LUU ======================
df_all['y_ai_v10.4'] = results
df_all['P_score_adr'] = scores_continuous   # MOI: diem AI lien tuc, dung cho SA-ECS
df_all.to_csv('drugsCom_V10.4_FINAL.csv', index=False)

# MOI: xuat rieng file gon nhe dung dinh dang can cho SA-ECS fusion (../src/fusion.py)
_id_col = [c for c in ("review_id", "uniqueID") if c in df_all.columns]
out_cols = _id_col.copy()
if "drugName" in df_all.columns:
    out_cols.append("drugName")
if "condition" in df_all.columns:
    out_cols.append("condition")
out_cols += ["P_score_adr", "y_ai_v10.4"]
out_df = df_all[out_cols].reset_index()
if not _id_col:
    out_df = out_df.rename(columns={"index": "review_id"})
else:
    out_df = out_df.drop(columns=["index"])
out_df = out_df.rename(columns={"y_ai_v10.4": "Y_hybrid"})
out_df.to_csv(OUTPUT_SA_ECS_FILENAME, index=False)
print(f"Da luu {len(out_df):,} dong -> {OUTPUT_SA_ECS_FILENAME} (cot P_score_adr LIEN TUC dung truc tiep cho src/fusion.py)")

# Doi soat 500 mau — GIU NGUYEN
if os.path.exists('Phieu_Khao_Sat_Chuyen_Gia.xlsx'):
    df_expert = pd.read_excel('Phieu_Khao_Sat_Chuyen_Gia.xlsx')
    y_true = df_expert['Nhãn Chuyên Gia (1=Có, 0=Không)'].astype(int)
    y_pred = df_all.loc[df_expert['Mã Đánh Giá'], 'y_ai_v10.4'].astype(int)
    print("\n" + "="*50)
    print("KET QUA DOI SOAT CHUYEN GIA (500 SAMPLES)")
    print(classification_report(y_true, y_pred, digits=4))
    print(f"Tong thoi gian thuc hien: {(time.time()-start_time)/60:.2f} phut")
    print("="*50)
else:
    print("Khong tim thay file Phieu_Khao_Sat_Chuyen_Gia.xlsx de doi soat.")