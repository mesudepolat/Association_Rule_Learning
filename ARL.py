############################################
# ASSOCIATION_RULE_LEARNING
############################################

# Veri ön işleme
import pandas as pd
from helpers.helpers import check_df, crm_data_prep
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.info()
df.head()
check_df(df)

df = crm_data_prep(df)
check_df(df)

df_ger = df[df["Country"] == "Germany"]
# ürünleri değerlendirebilmek için "postage" çıkardım
df_ger = df[(df['Country'] == "Germany") & (df['Description'] != "POSTAGE")]
check_df(df_ger)

df_ger.groupby(['Invoice', 'StockCode']).agg({"Quantity": "sum"}).head(100)

df_ger.groupby(['Invoice', 'StockCode']).agg({"Quantity": "sum"}).unstack().iloc[14:20, 14:20]

# kontrol işlemi
df[(df["StockCode"] == 16235) & (df["Invoice"] ==538174)]

# quantityleri nan olanları 0 ile doldurma işlemi
df_ger.groupby(['Invoice', 'StockCode']).\
    agg({"Quantity": "sum"}).\
    unstack().fillna(0).iloc[14:20, 14:20]

# quantityleri nan olmayanları 1 ile doldurma işlemi
df_ger.groupby(['Invoice', 'StockCode']).\
    agg({"Quantity": "sum"}).\
    unstack().fillna(0).\
    applymap(lambda x:1 if x > 0 else 0).iloc[14:20, 14:20]


# işlemleri fonksiyonlaştırma
def create_invoice_product_df(dataframe):
    return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0).\
        applymap(lambda x: 1 if x > 0 else 0)

ger_inv_pro_df = create_invoice_product_df(df_ger)

# birliktelik kurallarının çıkarılması
frequent_itemsets = apriori(ger_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()
rules.sort_values("lift", ascending=False).head()