# 提出用の処理をメモ

# 提出用
df_test = pd.read_csv('data/test.csv')
# 欠損値処理
df_test = df_test.dropna()
# 重複部分の削除
df_test.drop_duplicates(inplace=True)
# 重複部分の削除の後に新しいインデックスを取得
df_test.reset_index(drop=True, inplace=True)
# idを分離
test_id = df_test["id"]
# id列を削除
del df_test['id']
# 前処理
df_test["term"] = df_test["term"].str.replace('years', '')
df_test["employment_length"] = df_test["employment_length"].str.replace(
    'years', '').str.replace(
    'year', '')
df_test["term"] = df_test["term"].astype(int)
df_test["employment_length"] = df_test["employment_length"].astype(int)
# ダミー変数化
df_test = pd.get_dummies(
    df_test, columns=["grade", "purpose", "application_type"])
y_test_pred = lr.predict(df_test)
# print("結果：", y_test_pred)

print(len(test_id))
print(len(np.array(y_test_pred)))

# 提出
df_submit = pd.DataFrame({"id": test_id, "loan_status": np.array(y_test_pred)})
df_submit.to_csv('result/submit.csv', index=None, header=None)
"""
There is no prediction or the format is wrong for ID "1067656" in the submitted file. Please check the file.
→SIGNATEに提出する場合は欠損値を列ごと削除するのはNG
"""
