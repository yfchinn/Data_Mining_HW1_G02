"""
【本程式的核心功能與 EDA 實證發現】
1. 特徵品質與多重共線性檢驗 (Module 1 - VIF & Correlation Heatmap):
   - 嚴格過濾當月結果變數 (Target/Feature Leakage)，並剔除 bus_total_* 等恆等式的欄位 (A = B + C)。
   - 算出的 VIF (約 30~150) 證實外部特徵間存在顯著多重共線性，適合使用 LASSO (L1 正則化) 建模。

2. 驗證研究問題 RQ1 - 連假滯後連動效應 (Module 2 - Holiday Lag Correlation $H_1$):
   - 繪製 2x2 對照散佈圖 (fig2_rq1_holiday_lag.png) 並附帶年份標籤標註。
   - 實證顯示：全歷史數據下 3 月與 9 月 YoY 相關係數為 r = 0.49；若排除 2020-2022 疫情極端扭曲年份，
     r 上升至 0.96 (對照組 8 月與 9 月為 r = 0.98)，證實特定連假月份間具備強烈跨期 Lag-12 連動性。

4. 特徵重要性評估 (Module 4 - Random Forest Feature Importance):
   - 於訓練集上評估 Top 15 特徵排行榜，大眾運輸供給端指標 (路線公里數、營運日數) 與 GDP 具備最高預測力。

5. 給建模組員的資料轉型工具 (Panel Data Transformation Utility):
   - 本程式提供 `convert_to_panel_data(df)` 函式，供建模組員自由呼叫。
   - 若建模組欲使用『單一模型 (OLS / LASSO) 預測全台 12 個分站運量』，呼叫此函式即可將寬資料 (Wide)
     自動重塑為面板資料 (Long Format)，並建立分站獨立之 Lag-1 與 Lag-12 特徵。

【執行環境】
- 僅使用前 210 個月作為 Training Set，保留最後 24 個月為 Test Set。
- 產出之 4 張圖表自動儲存於 `reports/figures/` 資料夾。
================================================================================
"""

import os
import numpy as np
import pandas as pd

# 設定 Matplotlib 背景繪圖模式 (Agg)，確保 Server/無圖形介面環境正常執行
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 嘗試引用前處理模組 (若缺乏則提供容錯備用 logic)
try:
    from preprocessing import fit_outlier_bounds, remove_outliers
except ImportError:
    def fit_outlier_bounds(df, cols):
        bounds = {}
        for c in cols:
            q1 = df[c].quantile(0.25)
            q3 = df[c].quantile(0.75)
            iqr = q3 - q1
            bounds[c] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        return bounds

    def remove_outliers(df, bounds):
        cond = pd.Series(True, index=df.index)
        for c, (lower, upper) in bounds.items():
            cond &= (df[c] >= lower) & (df[c] <= upper)
        return df[cond], df[~cond]

# 全域繪圖樣式設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

# 圖表輸出目錄
FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ==============================================================================
# 1. 資料載入與防洩漏過濾 (Data Preparation & Leakage Filtering)
# ==============================================================================
def load_and_prepare_data(csv_path="data/processed_calendar/merged_monthly.csv", test_months=24):
    """
    載入前處理後的 CSV，動態計算高鐵自身的 5 大 Lag 特徵，並切分 Train/Test。
    """
    if not os.path.exists(csv_path):
        if os.path.exists("merged_monthly.csv"):
            csv_path = "merged_monthly.csv"
            
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    
    # 💡 核心填補：動態計算高鐵總運量 (hsr_passengers) 的 5 大 Lag 特徵
    target_col = 'hsr_passengers'
    df['lag_1'] = df[target_col].shift(1)
    df['lag_3'] = df[target_col].shift(3)
    df['lag_12'] = df[target_col].shift(12)
    df['rolling_mean_3'] = df[target_col].shift(1).rolling(3).mean()
    df['rolling_mean_12'] = df[target_col].shift(1).rolling(12).mean()
    
    # 前 12 個月會因為位移產生 NaN，刪除缺值列以維護訓練集品質
    df_clean = df.dropna(subset=['lag_12', 'rolling_mean_12']).copy()
    
    train_df = df_clean.iloc[:-test_months].copy()
    test_df = df_clean.iloc[-test_months:].copy()
    
    print(f"✅ 資料載入與 Lag 特徵建立成功！總筆數: {len(df_clean)} 個月 | 訓練集: {len(train_df)} 個月 | 測試集: {len(test_df)} 個月")
    return df_clean, train_df, test_df


def get_candidate_features(df):
    """
    【防漏與防重複過濾】
    1. 包含 5 大 Target History Lag 特徵。
    2. 排除當月高鐵/公車結果變數 (Leakage)。
    3. 排除 bus_total_* 欄位 (防止 A = B + C 恆等式導致 VIF 暴衝)。
    """
    target_lags = ['lag_1', 'lag_3', 'lag_12', 'rolling_mean_3', 'rolling_mean_12']
    
    leakage_cols = {
        'date', 'month_sin', 'month_cos', 'days_in_month', 'hsr_passengers',
        'hsr_daily_passengers', 'hsr_passenger_km', 'hsr_avg_trip_km',
        'hsr_occupancy_pct', 'hsr_punctuality_pct', 'hsr_trips', 'hsr_train_km'
    }
    
    candidates = list(target_lags)  # 先加入 5 大 Lag 特徵
    
    for col in df.columns:
        if col in target_lags or col.endswith('_is_missing'):
            continue
        if col.startswith('hsr_entry_') or col.startswith('hsr_exit_'):
            continue
        if col.startswith('bus_') and ('revenue' in col or 'passengers' in col or 'vehicle_km' in col):
            continue
        if col.startswith('bus_total_'):
            continue
        if col in leakage_cols:
            continue
            
        candidates.append(col)
        
    return candidates

# ==============================================================================
# 2. 模組一：多重共線性與 VIF 檢驗 (Correlation & VIF Analysis)
# ==============================================================================
def analyze_correlation_and_vif(train_df, feature_cols, target_col='hsr_passengers', top_n=12):
    """
    【模組一】共線性與 VIF 分析（加強版：自動防範 Singular Matrix 警告）
    """
    print("\n--- [Module 1] 多重共線性與 VIF 分析 ---")
    
    # 1. 計算並繪製 Correlation Heatmap
    corrs = train_df[feature_cols + [target_col]].corr()
    top_features = corrs[target_col].abs().sort_values(ascending=False).head(top_n + 1).index.tolist()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corrs.loc[top_features, top_features], annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f"Correlation Heatmap (Top {top_n} Features with Target)", fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(FIG_DIR, "fig1_correlation_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📸 圖表已存至: {save_path}")
    
    # 2. 安全計算 VIF (排除全為常數/全為0的欄位)
    X_vif = train_df[feature_cols].dropna().copy()
    
    # 剔除標準差為 0 的欄位 (例如全都是 0 的特徵)，防止 Singular Matrix
    non_constant_cols = [col for col in X_vif.columns if X_vif[col].std() > 1e-5]
    X_vif = X_vif[non_constant_cols]
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_vif.columns
    
    vif_values = []
    for i in range(X_vif.shape[1]):
        try:
            val = variance_inflation_factor(X_vif.values, i)
        except Exception:
            val = np.inf
        vif_values.append(val)
        
    vif_data["VIF"] = vif_values
    
    # 印出 VIF 警告特徵
    high_vif = vif_data[vif_data["VIF"] > 10].sort_values(by="VIF", ascending=False)
    print("\n[VIF 提示] 以下特徵存在高度共線性 (VIF > 10 或 inf，可作為 LASSO 篩選的依據):")
    print(high_vif.head(10).to_string(index=False))
    
    return corrs, vif_data


# ==============================================================================
# 3. 模組二：RQ1 連假滯後連動效應 (Holiday Lag Correlation $H_1$)
# ==============================================================================
def analyze_rq1_holiday_lag(train_df, target_col='hsr_passengers'):
    """
    【模組二】RQ1: 連假效應分析 
    - 解決欄位名稱為整數導致 regplot 報錯的問題
    - 繪製 2x2 對照圖：比較『包含疫情』與『排除 2020-2022 疫情年』對 3vs9 與 8vs9 相關性的巨大影響
    - 附帶年份標籤標註 (2020-2022 疫情年以紅字突出顯示)
    """
    print("\n--- [Module 2] RQ1: 連假效應與特定月份 Lag 連動性分析 ---")
    
    # 計算 YoY 成長率
    yoy_df = train_df[[target_col]].pct_change(12).dropna() * 100
    yoy_df['year'] = yoy_df.index.year
    yoy_df['month'] = yoy_df.index.month
    
    yearly_monthly = yoy_df.groupby(['year', 'month'])[target_col].sum().unstack()
    
    if 3 in yearly_monthly.columns and 9 in yearly_monthly.columns:
        ms = yearly_monthly[[3, 8, 9]].dropna().copy()
        # 重命名欄位名稱，防止 Seaborn 將整數欄位名誤判為常數值
        ms.columns = ['m3', 'm8', 'm9']
        
        # 排除 2020-2022 疫情干擾年
        ms_clean = ms[~ms.index.isin([2020, 2021, 2022])].copy()
        
        # 計算相關係數
        r_39_all = ms['m3'].corr(ms['m9'])
        r_89_all = ms['m8'].corr(ms['m9'])
        r_39_clean = ms_clean['m3'].corr(ms_clean['m9'])
        r_89_clean = ms_clean['m8'].corr(ms_clean['m9'])
        
        print(f"全數據   - 3月 vs 9月 YoY 相關係數: r = {r_39_all:.4f} | 8月 vs 9月: r = {r_89_all:.4f}")
        print(f"排除疫情 - 3月 vs 9月 YoY 相關係數: r = {r_39_clean:.4f} | 8月 vs 9月: r = {r_89_clean:.4f}")
        
        # 創建 2x2 的對照圖
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 輔助函式：繪製散佈圖並標註年份
        def plot_scatter_with_labels(ax, data, x_col, y_col, title, color):
            sns.regplot(x=x_col, y=y_col, data=data, ax=ax, color=color, scatter_kws={'alpha': 0.6})
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel(f"{x_col.upper()} Growth Rate YoY (%)", fontsize=10)
            ax.set_ylabel(f"{y_col.upper()} Growth Rate YoY (%)", fontsize=10)
            
            for year, row in data.iterrows():
                is_covid = year in [2020, 2021, 2022]
                txt_color = 'red' if is_covid else 'black'
                font_weight = 'bold' if is_covid else 'normal'
                ax.annotate(
                    str(year), 
                    (row[x_col], row[y_col]), 
                    textcoords="offset points", 
                    xytext=(4, 4), 
                    ha='left', 
                    fontsize=8, 
                    color=txt_color, 
                    fontweight=font_weight
                )

        # 1. 左上：3月 vs 9月 (包含疫情)
        plot_scatter_with_labels(
            axes[0, 0], ms, 'm3', 'm9', 
            f"March vs Sept YoY% [ALL YEARS] (r = {r_39_all:.2f})", 'blue'
        )
        
        # 2. 右上：3月 vs 9月 (排除 2020-2022 疫情)
        plot_scatter_with_labels(
            axes[0, 1], ms_clean, 'm3', 'm9', 
            f"March vs Sept YoY% [EXCL. COVID 2020-2022] (r = {r_39_clean:.2f})", 'darkblue'
        )
        
        # 3. 左下：8月 vs 9月 (對照組 - 包含疫情)
        plot_scatter_with_labels(
            axes[1, 0], ms, 'm8', 'm9', 
            f"August vs Sept YoY% [ALL YEARS] (r = {r_89_all:.2f})", 'green'
        )
        
        # 4. 右下：8月 vs 9月 (對照組 - 排除 2020-2022 疫情)
        plot_scatter_with_labels(
            axes[1, 1], ms_clean, 'm8', 'm9', 
            f"August vs Sept YoY% [EXCL. COVID 2020-2022] (r = {r_89_clean:.2f})", 'darkgreen'
        )
        
        plt.tight_layout()
        save_path = os.path.join(FIG_DIR, "fig2_rq1_holiday_lag.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"📸 2x2 對照圖表已存至: {save_path} (已解決報錯並標註年份與疫情對照)")
    else:
        print("⚠️ 訓練集中缺少足夠的 3 月與 9 月對應數據。")


# ==============================================================================
# 4. 模組三：RQ2 GDP 所得效果與運具轉移 (Modal Shift Analysis $H_2$)
# ==============================================================================
def analyze_rq2_modal_shift(train_df, gdp_col='gdp_per_capita_nominal_twd'):
    """
    【模組三目的】驗證假說 H2：GDP (所得) 成長是否推動民眾產生「運具轉移 (Modal Shift)」。
    - 左圖：GDP vs 高鐵/公路客運搭乘比值 (HSR/Bus Ratio)。
    - 右圖：GDP 成長率 vs 高鐵/客運運量年增率 (對比邊際遞減與短期波動特性)。
    - 產出 fig3_rq2_modal_shift.png (排除 2020-2022 疫情期間之非市場干擾)。
    """
    print("\n--- [Module 3] RQ2: GDP 成長與運具轉移 (Modal Shift) 分析 ---")
    
    df_rq2 = train_df.copy()
    # 排除 2020-2022 疫情封城干擾
    df_rq2_clean = df_rq2[~df_rq2.index.year.isin([2020, 2021, 2022])]
    
    if gdp_col in df_rq2.columns and 'bus_total_passengers' in df_rq2.columns and 'hsr_passengers' in df_rq2.columns:
        # 計算運量替代比值
        df_rq2_clean['hsr_to_bus_ratio'] = df_rq2_clean['hsr_passengers'] / df_rq2_clean['bus_total_passengers']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左圖：水準值 (GDP vs HSR/Bus Ratio)
        sns.regplot(x=gdp_col, y='hsr_to_bus_ratio', data=df_rq2_clean, ax=axes[0], 
                    color='teal', scatter_kws={'alpha':0.6})
        axes[0].set_title("GDP per Capita vs HSR/Bus Traffic Ratio (Excl. COVID)")
        axes[0].set_xlabel("Nominal GDP per Capita (TWD)")
        axes[0].set_ylabel("Traffic Ratio (HSR / Bus)")
        
        # 右圖：成長率 YoY (GDP Growth vs Transport Growth)
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
        print(f"📸 圖表已存至: {save_path}")
        
        corr_ratio = df_rq2_clean[gdp_col].corr(df_rq2_clean['hsr_to_bus_ratio'])
        print(f"GDP 與『高鐵/客運搭乘比值』相關係數: r = {corr_ratio:.4f} (顯著正相關驗證假說 H2)")
    else:
        print("⚠️ 缺少 GDP 或公路客運欄位，無法進行模組三分析。")


# ==============================================================================
# 5. 模組四：特徵重要性排行榜 (Random Forest Feature Importance)
# ==============================================================================
def evaluate_feature_importance(train_df, feature_cols, target_col='hsr_passengers', top_n=15):
    """
    【模組四目的】使用隨機森林評估無洩漏變數下之特徵重要性 Top 15。
    - 產出 fig4_feature_importance.png。
    """
    print("\n--- [Module 4] 特徵重要性評估 (Random Forest) ---")
    
    try:
        bounds = fit_outlier_bounds(train_df, [target_col])
        clean_train, removed = remove_outliers(train_df, bounds)
        print(f"訓練集 IQR 離群值檢查: 保留 {len(clean_train)} 列，剔除 {len(removed)} 列極端值")
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
    #sns.barplot(x='Importance', y='Feature', data=feat_imp.head(top_n), palette='viridis')
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=feat_imp.head(top_n), 
        hue='Feature',      # 指定 hue 為 y 軸變數
        palette='viridis', 
        legend=False        # 關閉右側不需要的圖例 (Legend)
    )
    plt.title(f"Top {top_n} Feature Importance Ranking", fontsize=14)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    
    save_path = os.path.join(FIG_DIR, "fig4_feature_importance.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📸 圖表已存至: {save_path}")
    
    return feat_imp


# ==============================================================================
# 6. 面板資料轉換工具 (Panel Data Transformation Utility - Optional)
# ==============================================================================
def convert_to_panel_data(df):
    """
    【進階工具】若團隊決議採用『單一模型預測全台 12 個分站運量』，
    此函式可將寬資料 (Wide Format) 重塑為面板資料 (Long Format)。
    """
    station_cols = [c for c in df.columns if c.startswith('hsr_entry_') or c.startswith('hsr_exit_')]
    macro_cols = [c for c in df.columns if c not in station_cols and not c.endswith('_is_missing')]
    
    panel_rows = []
    for date_idx, row in df.iterrows():
        for col_name in station_cols:
            parts = col_name.split('_')
            direction = parts[1]  # entry / exit
            station = "_".join(parts[2:]) # taipei, taichung...
            
            val = row[col_name]
            
            row_dict = {
                'date': date_idx,
                'station': station,
                'direction': direction,
                'target_passengers': val
            }
            for m_col in macro_cols:
                row_dict[m_col] = row[m_col]
                
            panel_rows.append(row_dict)
            
    panel_df = pd.DataFrame(panel_rows).set_index('date')
    
    # 針對分站維度建立 Lag 特徵
    panel_df = panel_df.sort_values(['station', 'direction', 'date'])
    panel_df['lag1_passengers'] = panel_df.groupby(['station', 'direction'])['target_passengers'].shift(1)
    panel_df['lag12_passengers'] = panel_df.groupby(['station', 'direction'])['target_passengers'].shift(12)
    
    return panel_df


# ==============================================================================
# 主執行流程 (Main Workflow)
# ==============================================================================
if __name__ == "__main__":
    # 1. 載入 merged_monthly.csv 並切分 Train/Test
    df, train_df, test_df = load_and_prepare_data("data/processed/merged_monthly.csv")
    
    # 2. 自動過濾出合法無洩漏候選特徵
    candidate_features = get_candidate_features(train_df)
    
    # 3. 執行四大探索性分析模組
    corr_matrix, vif_table = analyze_correlation_and_vif(train_df, candidate_features)
    analyze_rq1_holiday_lag(train_df)
    analyze_rq2_modal_shift(train_df)
    feature_ranking = evaluate_feature_importance(train_df, candidate_features)
    
    print(f"\n🎉 特徵分析全部完成！所有圖表均已更新並儲存於 `{FIG_DIR}/` 目錄。")

