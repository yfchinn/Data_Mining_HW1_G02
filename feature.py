"""
1. 特徵品質與共線性檢驗 (VIF & Heatmap)：
   - 自動過濾「資料洩漏 (Data Leakage)」變數（如當月高鐵進出站總人次、當月公車營收/人次）。
   - 篩選出真正的「外部/預測特徵」（如 GDP、連假天數、公車供給端特徵），避免預測模型吃到未來的答案。

2. 分析 3 月與 9 月（皆有連假之月份）在「年增率 (YoY %)」上的相關性，並以 8 月作為無連假對照組。

3. 探討每人 GDP 成長對「高鐵 / 公路客運搭乘比率」的影響，驗證民眾是否因所得增加而產生運具升級（轉搭高鐵）。

4. 特徵重要性評估 (Feature Importance)：
   - 使用隨機森林 (Random Forest) 產出 Top 15 關鍵特徵排行榜。

- 產出的 4 張 PNG 圖表會自動存入 `reports/figures/` 資料夾
- 預設保留最後 24 個月作為測試集，特徵分析僅使用訓練集以防資訊洩漏。
- figure2 有超過100%的極端值，是因為疫情解封、運量回彈的影響。
=========================================
"""

import os
import numpy as np
import pandas as pd

# 設定 Matplotlib 為背景繪圖模式 (Agg)，確保在 Server / Remote GPU 環境下執行不報錯
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 引用組員寫好的前處理工具函式 (請確保 preprocessing.py 在同目錄下)
from preprocessing import fit_outlier_bounds, remove_outliers

# 全域繪圖樣式設定 (採用全英文標籤避免 Linux Server 缺字體亂碼)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

# 圖表輸出目錄
FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ==========================================
# 1. 資料載入與特徵過濾 (Data Preparation)
# ==========================================
def load_and_prepare_data(csv_path="data/processed/merged_monthly.csv", test_months=24):
    """
    載入前處理後的 CSV，並劃分訓練集 (Train) 與測試集 (Test)。
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    
    train_df = df.iloc[:-test_months].copy()
    test_df = df.iloc[-test_months:].copy()
    
    print(f"✅ 資料載入成功！總資料筆數: {len(df)} 個月 | 訓練集: {len(train_df)} 個月 | 測試集: {len(test_df)} 個月")
    return df, train_df, test_df


def get_candidate_features(df):
    """
    【重點防漏機制】選取候選特徵，嚴格排除 Data Leakage (資料洩漏) 欄位：
    1. 排除同月份高鐵自身的結果變數 (如進出站總人次、日均人次、列車行駛公里等)。
    2. 排除同月份公車的「營運結果」變數 (如當月公車營收、當月公車總人次)，因為預測未來時無法提前得知。
    3. 保留「外部特徵」(GDP、連假) 與「公車供給端指標」(如營運車輛數、路線公里數)。
    """
    leakage_cols = {
        'date', 'month_sin', 'month_cos', 'days_in_month', 'hsr_passengers',
        'hsr_daily_passengers', 'hsr_passenger_km', 'hsr_avg_trip_km',
        'hsr_occupancy_pct', 'hsr_punctuality_pct', 'hsr_trips', 'hsr_train_km'
    }
    
    candidates = []
    for col in df.columns:
        # 排除 missing 指標欄位
        if col.endswith('_is_missing'):
            continue
        # 排除同月份高鐵各分站進出站人數 (本質等於總運量)
        if col.startswith('hsr_entry_') or col.startswith('hsr_exit_'):
            continue
        # 排除當月公車營收與搭乘人次 (防止結果變數洩漏)
        if col.startswith('bus_') and ('revenue' in col or 'passengers' in col or 'vehicle_km' in col):
            continue
        # 排除其他同源結果變數
        if col in leakage_cols:
            continue
            
        candidates.append(col)
        
    return candidates


# ==========================================
# 2. 模組一：共線性與 VIF 檢驗 (Correlation & VIF)
# ==========================================
def analyze_correlation_and_vif(train_df, feature_cols, target_col='hsr_passengers', top_n=12):
    """
    【模組一目的】檢驗特徵之間的共線性 (Multicollinearity) 與其對高鐵運量的相關性。
    - 繪製 Top N 最強相關特徵的 Heatmap (存為 fig1_correlation_heatmap.png)。
    - 計算 VIF (方差膨脹因子)，若 VIF > 10 代表資訊重複性高，可供建模組員做剔除參考。
    """
    print("\n--- [模組一] 共線性與特徵相關性分析 ---")
    
    corrs = train_df[feature_cols + [target_col]].corr()
    top_features = corrs[target_col].abs().sort_values(ascending=False).head(top_n + 1).index.tolist()
    
    # 畫 Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corrs.loc[top_features, top_features], annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f"Correlation Heatmap (Top {top_n} Features with Target)", fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(FIG_DIR, "fig1_correlation_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📸 圖表已儲存至: {save_path}")
    
    # 計算 VIF
    X_vif = train_df[feature_cols].dropna()
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    
    high_vif = vif_data[vif_data["VIF"] > 10].sort_values(by="VIF", ascending=False)
    print("\n[VIF 提示] 以下特徵 VIF > 10 (存在高度多重共線性，建模時建議篩選剔除):")
    print(high_vif)
    
    return corrs, vif_data


# ==========================================
# 3. 模組二：RQ1 連假效應分析 (March vs Sept Lag)
# ==========================================
def analyze_rq1_holiday_lag(train_df, target_col='hsr_passengers'):
    """
    【模組二目的】回答 RQ1：當年 3 月與 9 月皆有連假時，3 月運量是否對 9 月運量有更強的預測力？
    - 改用「年增率 (YoY %)」進行比對，排除時間趨勢引起的偽相關。
    - 以 8 月 (暑假/無長假) 作為對照組。
    - 圖表存為 fig2_rq1_holiday_lag.png。
    """
    print("\n--- [模組二] RQ1: 連假效應與特定月份 Lag 連動性分析 ---")
    
    # 計算 YoY 成長率
    yoy_df = train_df[[target_col]].pct_change(12).dropna() * 100
    yoy_df['year'] = yoy_df.index.year
    yoy_df['month'] = yoy_df.index.month
    
    yearly_monthly = yoy_df.groupby(['year', 'month'])[target_col].sum().unstack()
    
    if 3 in yearly_monthly.columns and 9 in yearly_monthly.columns:
        march_sept = yearly_monthly[[3, 8, 9]].dropna()
        
        corr_3_9 = march_sept[3].corr(march_sept[9])
        corr_8_9 = march_sept[8].corr(march_sept[9])
        
        print(f"3 月 (連假) 與 9 月 (連假) 運量年增率相關係數: {corr_3_9:.4f}")
        print(f"8 月 (無連假對照) 與 9 月 運量年增率相關係數: {corr_8_9:.4f}")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 實驗組：3月 vs 9月
        sns.regplot(x=march_sept[3], y=march_sept[9], ax=axes[0], color='b')
        axes[0].set_title(f"RQ1: March YoY% vs Sept YoY% (r = {corr_3_9:.2f})")
        axes[0].set_xlabel("March Growth Rate YoY (%)")
        axes[0].set_ylabel("September Growth Rate YoY (%)")
        
        # 對照組：8月 vs 9月
        sns.regplot(x=march_sept[8], y=march_sept[9], ax=axes[1], color='g')
        axes[1].set_title(f"Control: August YoY% vs Sept YoY% (r = {corr_8_9:.2f})")
        axes[1].set_xlabel("August Growth Rate YoY (%)")
        axes[1].set_ylabel("September Growth Rate YoY (%)")
        
        plt.tight_layout()
        save_path = os.path.join(FIG_DIR, "fig2_rq1_holiday_lag.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📸 圖表已儲存至: {save_path}")
    else:
        print("訓練集中缺少足夠的 3 月與 9 月跨年數據。")


# ==========================================
# 4. 模組三：RQ2 GDP (所得) 與運具轉移 (Modal Shift)
# ==========================================
def analyze_rq2_modal_shift(train_df, gdp_col='gdp_per_capita_nominal_twd'):
    """
    【模組三目的】回答 RQ2：GDP (每人所得) 成長是否會顯著促進大眾產生「運具轉移 (Modal Shift)」？
    - 計算「高鐵 / 公路客運」的運量比值 (Ratio)。
    - 驗證隨著 GDP 成長，民眾是否傾向選擇高單價/高速的高鐵，取代傳統公路客運。
    - 圖表存為 fig3_rq2_modal_shift.png (排除 2020-2022 疫情劇烈干擾期)。
    """
    print("\n--- [模組三] RQ2: GDP 成長對運具轉移 (Modal Shift) 之影響分析 ---")
    
    df_rq2 = train_df.copy()
    # 過濾疫情封城特殊年份以保持分析嚴謹度
    df_rq2_clean = df_rq2[~df_rq2.index.year.isin([2020, 2021, 2022])]
    
    if gdp_col in df_rq2.columns and 'bus_total_passengers' in df_rq2.columns and 'hsr_passengers' in df_rq2.columns:
        # 計算運量替代比值
        df_rq2_clean['hsr_to_bus_ratio'] = df_rq2_clean['hsr_passengers'] / df_rq2_clean['bus_total_passengers']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 圖 A：GDP vs 高鐵/客運比值
        sns.regplot(x=gdp_col, y='hsr_to_bus_ratio', data=df_rq2_clean, ax=axes[0], 
                    color='teal', scatter_kws={'alpha':0.6})
        axes[0].set_title("GDP per Capita vs HSR/Bus Traffic Ratio (Excl. COVID)")
        axes[0].set_xlabel("Nominal GDP per Capita (TWD)")
        axes[0].set_ylabel("Traffic Ratio (HSR / Bus)")
        
        # 圖 B：GDP 成長率 vs 高鐵與客運成長率對比
        yoy_df = df_rq2_clean[[gdp_col, 'hsr_passengers', 'bus_total_passengers']].pct_change(12).dropna() * 100
        
        sns.regplot(x=gdp_col, y='hsr_passengers', data=yoy_df, ax=axes[1], 
                    color='red', label='HSR Passengers YoY', scatter_kws={'alpha':0.5})
        sns.regplot(x=gdp_col, y='bus_total_passengers', data=yoy_df, ax=axes[1], 
                    color='blue', label='Bus Passengers YoY', scatter_kws={'alpha':0.5})
        
        axes[1].set_title("GDP Growth Rate vs Transport Growth Rate (YoY %)")
        axes[1].set_xlabel("GDP Growth Rate (%)")
        axes[1].set_ylabel("Passenger Growth Rate (%)")
        axes[1].legend()
        
        plt.tight_layout()
        save_path = os.path.join(FIG_DIR, "fig3_rq2_modal_shift.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📸 圖表已儲存至: {save_path}")
        
        corr_ratio = df_rq2_clean[gdp_col].corr(df_rq2_clean['hsr_to_bus_ratio'])
        print(f"GDP 與『高鐵/客運搭乘比值』相關係數: {corr_ratio:.4f} (顯著正相關代表運具轉移假說成立)")
    else:
        print("資料集中缺少 GDP 或客運相關欄位，無法進行運具轉移分析。")


# ==========================================
# 5. 模組四：特徵重要性評估 (Feature Importance)
# ==========================================
def evaluate_feature_importance(train_df, feature_cols, target_col='hsr_passengers', top_n=15):
    """
    【模組四目的】使用隨機森林 (Random Forest) 評估無洩漏變數下的特徵重要性。
    - 自動調用 preprocessing.py 的 IQR 離群值檢測機制清理訓練集。
    - 印出 Top 15 特徵排行榜 (存為 fig4_feature_importance.png)，提供給建模組員參考。
    """
    print("\n--- [模組四] 特徵重要性排行榜 (Random Forest) ---")
    
    # 套用前處理的離群值過濾
    try:
        bounds = fit_outlier_bounds(train_df, [target_col])
        clean_train, removed = remove_outliers(train_df, bounds)
        print(f"訓練集進行 IQR 離群值檢查: 保留 {len(clean_train)} 列，剔除 {len(removed)} 列極端值")
    except Exception as e:
        print(f"離群值處理跳過，使用原始訓練集。(原因: {e})")
        clean_train = train_df

    X = clean_train[feature_cols].fillna(0)
    y = clean_train[target_col]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    feat_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp.head(top_n), palette='viridis')
    plt.title(f"Top {top_n} Feature Importance Ranking", fontsize=14)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    
    save_path = os.path.join(FIG_DIR, "fig4_feature_importance.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📸 圖表已儲存至: {save_path}")
    
    return feat_imp


# ==========================================
# 主執行流程 (Main Script)
# ==========================================
if __name__ == "__main__":
    # 1. 載入前處理完的 merged_monthly.csv
    df, train_df, test_df = load_and_prepare_data("data/processed/merged_monthly.csv")
    
    # 2. 自動抓取合法的候選特徵 (已排除 Leakage 欄位)
    candidate_features = get_candidate_features(train_df)
    
    # 3. 依序執行四大分析模組
    corr_matrix, vif_table = analyze_correlation_and_vif(train_df, candidate_features)
    analyze_rq1_holiday_lag(train_df)
    analyze_rq2_modal_shift(train_df)
    feature_ranking = evaluate_feature_importance(train_df, candidate_features)
    
    print(f"\n🎉 特徵分析全部順利完成！所有產出圖表均位於 `{FIG_DIR}/` 資料夾中。")
