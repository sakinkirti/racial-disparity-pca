import pandas as pd

path = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/dataset/racial_disparity_FINAL_batch1_anonymized.csv'
save = '/Users/sakinkirti/Programming/Python/CCIPD/racial-disparity-pca/dataset/racial_disparity_filtered.csv'

df = pd.read_csv(path)
df = df[df['GGG (1-5)'].notnull()]

df.to_csv(save)