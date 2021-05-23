"""
課題：https://signate.jp/competitions/294
借入総額や返済期間、金利、借入目的などの顧客データを使って、債務不履行リスクを予測するモデルを構築
ChargedOffを1、FullyPaidを0として予測
"""

# 評価基準：f1_score
# https: // scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

# import
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectFromModel

# 前処理

# 読み込み
df = pd.read_csv('data/train.csv')
# 欠損値処理
df = df.dropna()
# id列を削除
del df['id']
# 重複部分の削除
df.drop_duplicates(inplace=True)
# 重複部分の削除の後に新しいインデックスを取得
df.reset_index(drop=True, inplace=True)

df["term"] = df["term"].str.replace('years', '')
df["employment_length"] = df["employment_length"].str.replace(
    'years', '').str.replace(
    'year', '')
df["loan_status"] = df["loan_status"].str.replace(
    'FullyPaid', '0').str.replace('ChargedOff', '1')

df["term"] = df["term"].astype(int)
df["employment_length"] = df["employment_length"].astype(int)
df["loan_status"] = df["loan_status"].astype(int)

# データフレームの分離
col_categoric = ["grade", "purpose", "application_type", "loan_status"]
df_numeric = df.drop(col_categoric, axis=1)
df_categoric = df[col_categoric]
# df_categoric内の"loan_status"列と、df_numericの列を横結合する
df_tmp = pd.concat([df_categoric["loan_status"], df_numeric], axis=1)

# print(df)

# 基本統計量
# print(df.describe(include='all'))
"""
                 id      loan_amnt           term  ...  application_type_Joint App  loan_status_ChargedOff  loan_status_FullyPaid
count  2.289710e+05  228971.000000  228971.000000  ...               228971.000000           228971.000000          228971.000000
mean   5.570189e+07    1433.415476       3.448887  ...                    0.022627                0.194544               0.805456
std    4.789707e+07     875.149218       0.834432  ...                    0.148713                0.395850               0.395850
min    5.571600e+04     100.000000       3.000000  ...                    0.000000                0.000000               0.000000
25%    3.345535e+06     780.000000       3.000000  ...                    0.000000                0.000000               1.000000
50%    8.552937e+07    1200.000000       3.000000  ...                    0.000000                0.000000               1.000000
75%    9.230207e+07    2000.000000       3.000000  ...                    0.000000                0.000000               1.000000
max    1.264193e+08    4000.000000       5.000000  ...                    1.000000                1.000000               1.000000
"""

# データ数を取得
counts_loan_status = df["loan_status"].value_counts()

# 棒グラフによる可視化ー質的データ
# counts_loan_status.plot(kind='bar')
"""
graph/loan-status-bar.png
0(FullyPaid)>1(ChargedOff)
"""

# 数量変数のヒストグラムを表示(※figsizeオプションはグラフのサイズを指定）
# df_numeric.hist(figsize=(8, 6))
# グラフのラベルが重ならないようにレイアウトを自動調整
# plt.tight_layout()
"""
graph/numeric-hist.png
全体的に左寄り
"""

"""
仮説
債務履行者（loan_status=0）に比べ、債務不履行者（loan_status=1)のデータでは、正常な検査値の範囲（基準値）から外れるケースが多くなるため、
ヒストグラムを描いた時のピーク値（最頻値）などが、債務履行者と異なるのではないか
"""

# グラフの表示
# plt.figure(figsize=(12, 12))
# for ncol, colname in enumerate(df_numeric.columns):
#     plt.subplot(3, 3, ncol+1)
#     sns.distplot(df_tmp.query("loan_status==0")[colname])
#     sns.distplot(df_tmp.query("loan_status==1")[colname])
#     plt.legend(labels=["FullyPaid", "ChargedOff"], loc='upper right')
"""
graph/hist.png
"""

# heatmapの表示
# plt.figure(figsize=(10, 8))
# sns.heatmap(df.corr(), vmin=-1.0, vmax=1.0, annot=True,
#             cmap='coolwarm', linewidths=0.1)
"""
vmin, vmaxオプションは、最小値と最大値を指定しています。
annot=Trueオプションは、色付きのセルの中に相関係数の数値を表示するためのものです。
cmap, linewidthsオプションは、見た目を調整するために、それぞれ塗り色のパターンとセル間の線の太さを指定しています。
"""
"""
graph/heatmap.png
"""

# # カテゴリ列を除外（数量変数のデータに絞る）
# X_target = df.drop(col_categoric, axis=1)
# # 多項式・交互作用特徴量の生成
# polynomial = PolynomialFeatures(degree=2, include_bias=False)
# polynomial_arr = polynomial.fit_transform(X_target)
# # polynomial_arrのデータフレーム化 （※カラムはshape[1]でpolynomial_arrの列数分だけ出力）
# X_polynomial = pd.DataFrame(polynomial_arr, columns=[
#                             "poly" + str(x) for x in range(polynomial_arr.shape[1])])
# # 組み込み法に使うモデルの指定
# # オプションのpenalty='l1'は、正則化と呼ばれる 過学習を防止するための制約条件
# fs_model = LogisticRegression(random_state=0)
# # 閾値の指定
# fs_threshold = "mean"
# selector = SelectFromModel(fs_model, threshold=fs_threshold)
# selector.fit(X_polynomial, df["loan_status"])
# # 学習させたモデルにget_support関数を使うことで、特徴量選択の結果が重要な変数ならTrue、重要でない変数ならFalseのbool型のリストを返す
# mask = selector.get_support()
# # 選択された特徴量だけのサンプル取得
# X_polynomial_masked = X_polynomial.loc[:, mask]
# # print(X_polynomial_masked.head())
"""
     poly6  poly13   poly16
0   6000.0  3500.0  11403.0
1  10000.0  3350.0  14726.6
2   3000.0  2130.0   6098.9
3   4500.0  2040.0   9513.2
4    930.0  2370.0   5806.5
"""

# ダミー変数化
df = pd.get_dummies(
    df, columns=["grade", "purpose", "application_type"])

# 目的変数のデータフレーム
y = df["loan_status"]
# 説明変数のデータフレーム
X = df.drop(["loan_status"], axis=1)

# X_polied = pd.concat([X, X_polynomial_masked], axis=1)
# X_poliedで学習した場合のF1スコア: 0.38907081868407833

# # 等間隔のbin分割
# bins_credit_score = [659.0, 697.0, 734.0, 845.0]
# X_cut, bin_indice = pd.cut(
#     X["credit_score"], bins=bins_credit_score, retbins=True)
# # bin分割した結果の表示
# # print(X_cut.value_counts())
# """
# (659.0, 697.0]    121010
# (697.0, 734.0]     58279
# (734.0, 845.0]     28284
# """
# # bin分割した結果をダミー変数化 (prefix=X_Cut.nameは、列名の接頭語を指定している)
# X_dummies = pd.get_dummies(X_cut, prefix=X_cut.name)
# # 元の説明変数のデータフレーム(X)と、ダミー変数化の結果(X_dummies)を横連結
# X_binned = pd.concat([X, X_dummies], axis=1)
# # 結果の確認
# print(X_binned.head())

# データの分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)

# X_train, X_test, y_train, y_test = train_test_split(
#     X_polied, y, test_size=0.4, random_state=0)

# ダウンサンプリングhttps://blog.amedama.jp/entry/imbalanced-data
# sampler = RandomUnderSampler(random_state=42, sampling_strategy={
#                              0: y_train[y_train == 0], 1: y_train[y_train == 1]})
sampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
# print(sum(y_resampled))  # 25694
# print(len(y_resampled))  # 51388
# print(y_resampled)

df["loan_status"] = df["loan_status"].replace(
    True, 0).replace(False, 1)
df["loan_status"] = df["loan_status"].astype(int)

# データサイズの確認
# print(X_train.shape)
# (160279, 56)
# print(X_test.shape)
# (68692, 56)
# print(y_train.shape)
# (160279,)

# モデルの学習
# lr = LogisticRegression(class_weight=df['credit_rate'])
# print(X_resampled, y_resampled)
lr = LogisticRegression()
lr.fit(X_resampled, y_resampled)

# print(X_test)

# 予測
y_pred = lr.predict(X_test)
# print("y_predの結果（2値判定の結果）: ", y_pred)
# print("1(ChargedOff)の数: ", sum(y_pred))

# 予測（判定確率の算出）
result = lr.predict_proba(X_test)
# print("予測結果（最初の5サンプルを表示）")
# print(result[:5])

# １となる確率
result_1 = lr.predict_proba(X_test)[:, 1]
# print("1(ChargedOff)の確率を抽出した結果（最初の5サンプルだけ表示）")
# print(result_1[:5])

# 混同行列の作成
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
# print(cm)

# 混同行列をデータフレーム化
df_cm = pd.DataFrame(np.rot90(cm, 2), index=["実際のChargedOff", "実際のFullyPaid"], columns=[
                     "ChargedOffの予測", "FullyPaidの予測"])
print(df_cm)
"""
               ChargedOffの予測  FullyPaidの予測
実際のChargedOff            518         16657
実際のFullyPaid             433         65422
"""
"""
               ChargedOffの予測  FullyPaidの予測
実際のChargedOff          11977          5198
実際のFullyPaid           30787         35068
"""

# heatmapによる混同行列の可視化
sns.heatmap(df_cm, annot=True, fmt="2g", cmap='Blues')
plt.yticks(va='center')
"""
graph/result-hreatmap.png
graph/result-hreatmap_1.png
"""

# f1_score(y_true, y_pred)
f1 = f1_score(y_test, y_pred)

print("f1 is:", f1)
# 0.0571554672845636
# 0.3996396336275213

plt.show()
