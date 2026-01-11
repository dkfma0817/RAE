import os
import json
import pandas as pd

# ====== 너 환경에 맞춰 경로만 확인 ======
ROOT = "celeba_data"
IMG_DIR = os.path.join(ROOT, "img_align_celeba", "img_align_celeba")
ATTR_CSV = os.path.join(ROOT, "list_attr_celeba.csv")
SPLIT_CSV = os.path.join(ROOT, "list_eval_partition.csv")

OUT_DIR = os.path.join(ROOT, "t2i_jsonl")
os.makedirs(OUT_DIR, exist_ok=True)

# 우리가 쓸 5개 속성
ATTRS = ["Smiling", "Eyeglasses", "Bangs", "Blond_Hair", "Mustache"]

# 텍스트 템플릿(짧고 일정한 구조)
def attrs_to_prompt(row) -> str:
    tokens = ["a photo of a person"]
    if row["Smiling"] == 1:
        tokens.append("smiling")
    if row["Eyeglasses"] == 1:
        tokens.append("with glasses")
    if row["Bangs"] == 1:
        tokens.append("with bangs")
    if row["Blond_Hair"] == 1:
        tokens.append("with blond hair")
    if row["Mustache"] == 1:
        tokens.append("with a mustache")

    # 문장 안정성을 위해 join 규칙을 단순화
    # "a portrait photo of a person smiling with glasses with bangs ..." 처럼 될 수 있지만
    # 짧고 일관된 텍스트가 오히려 잘 먹는 경우가 많음.
    return " ".join(tokens)

def normalize_attr_values(df, cols):
    # Kaggle CelebA attr는 보통 -1/1 형태. 0/1로 바꿔서 다루기 편하게.
    for c in cols:
        if df[c].dtype != "int64" and df[c].dtype != "int32":
            df[c] = df[c].astype(int)
        # -1 -> 0, 1 -> 1
        df[c] = (df[c] == 1).astype(int)
    return df

def main():
    # 1) 로드
    attr = pd.read_csv(ATTR_CSV)
    split = pd.read_csv(SPLIT_CSV)

    # 컬럼명 확인 (Kaggle 버전에 따라 파일명 컬럼이 다를 수 있어 방어)
    # 보통 "image_id" 또는 "image" 또는 "file_name"
    img_col_attr = None
    for cand in ["image_id", "image", "file_name", "filename"]:
        if cand in attr.columns:
            img_col_attr = cand
            break
    if img_col_attr is None:
        raise ValueError(f"Could not find image id column in attr csv. columns={list(attr.columns)[:10]}...")

    img_col_split = None
    for cand in ["image_id", "image", "file_name", "filename"]:
        if cand in split.columns:
            img_col_split = cand
            break
    if img_col_split is None:
        raise ValueError(f"Could not find image id column in split csv. columns={list(split.columns)[:10]}...")

    # split 컬럼명도 방어 ("partition"이 일반적)
    part_col = None
    for cand in ["partition", "split"]:
        if cand in split.columns:
            part_col = cand
            break
    if part_col is None:
        raise ValueError(f"Could not find partition column in split csv. columns={list(split.columns)[:10]}...")

    # 2) 필요한 5개 속성만 남기기
    missing = [a for a in ATTRS if a not in attr.columns]
    if missing:
        raise ValueError(f"Missing attributes in CSV: {missing}. available={list(attr.columns)[:30]}...")

    attr = attr[[img_col_attr] + ATTRS].copy()
    split = split[[img_col_split, part_col]].copy()

    # 3) merge
    df = attr.merge(split, left_on=img_col_attr, right_on=img_col_split, how="inner")

    # 4) 값 정규화 (-1/1 -> 0/1)
    df = normalize_attr_values(df, ATTRS)

    # 5) prompt 만들기 + 이미지 경로 확인
    records = []
    missing_files = 0
    for _, row in df.iterrows():
        fname = row[img_col_attr]
        img_path = os.path.join(IMG_DIR, fname)
        if not os.path.exists(img_path):
            missing_files += 1
            continue

        prompt = attrs_to_prompt(row)
        rec = {
            "image": img_path,
            "prompt": prompt,
            "attrs": {k: int(row[k]) for k in ATTRS},
            "split": int(row[part_col]),
        }
        records.append(rec)

    print(f"Total merged: {len(df):,}, kept(with files): {len(records):,}, missing_files: {missing_files:,}")

    # 6) split 저장 (CelebA partition: 0=train, 1=val, 2=test)
    def write_jsonl(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    train = [r for r in records if r["split"] == 0]
    val   = [r for r in records if r["split"] == 1]
    test  = [r for r in records if r["split"] == 2]

    write_jsonl(os.path.join(OUT_DIR, "train.jsonl"), train)
    write_jsonl(os.path.join(OUT_DIR, "val.jsonl"), val)
    write_jsonl(os.path.join(OUT_DIR, "test.jsonl"), test)

    print(f"Wrote: train={len(train):,}, val={len(val):,}, test={len(test):,}")
    print("Example:")
    print(train[0])

if __name__ == "__main__":
    main()
