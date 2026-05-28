import pandas as pd
def balls_to_dataframe(balls) -> pd.DataFrame:
    rows = []
    for i, b in enumerate(balls):
        d = b.to_dict()
        d["GB_id"] = i
        rows.append(d)
    cols = ["GB_id","center", "target_mean","radius","members","start_idx","end_idx","slope","trend_purity","reg_purity","size"]
    return pd.DataFrame(rows)[cols]