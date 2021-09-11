import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import resample
cnn_folder = 'CNN/'

def get_alpha_f1s(df_gen: pd.DataFrame, df_cnn: pd.DataFrame) -> float:
    case_ids_cnn = df_gen['Case_Ids'].values
    case_ids_gen = df_cnn['Case_Ids'].values
    case_ids = np.intersect1d(case_ids_cnn, case_ids_gen)
    cnn_df_only = df_cnn.loc[df_cnn['Case_Ids'].isin(case_ids)]
    gen_df_only = df_gen.loc[df_gen['Case_Ids'].isin(case_ids)]
    real = gen_df_only['Real'].values
    case_ids_df = gen_df_only['Case_Ids'].values

    data = pd.DataFrame()
    data['Case_Ids'] = case_ids_df
    data['CNN Prob LUAD'] = cnn_df_only['Prob LUAD'].values
    data['CNN Prob HLT'] = cnn_df_only['Prob HLT'].values
    data['CNN Prob LUSC'] = cnn_df_only['Prob LUSC'].values
    data['Gen Prob LUAD'] = gen_df_only['Prob LUAD'].values
    data['Gen Prob HLT'] = gen_df_only['Prob HLT'].values
    data['Gen Prob LUSC'] = gen_df_only['Prob LUSC'].values
    data['CNN Pred'] = data[['CNN Prob LUAD', 'CNN Prob HLT', 'CNN Prob LUSC']].idxmax(axis=1).values
    data['CNN Pred'].replace({'CNN Prob LUAD': 0, 'CNN Prob HLT': 1, 'CNN Prob LUSC': 2}, inplace=True)
    data['Gen Pred'] = data[['Gen Prob LUAD', 'Gen Prob HLT', 'Gen Prob LUSC']].idxmax(axis=1).values
    data['Gen Pred'].replace({'Gen Prob LUAD': 0, 'Gen Prob HLT': 1, 'Gen Prob LUSC': 2}, inplace=True)
    data['Real'] = real

    accs_cnn = []
    accs_gen = []
    len_samples = int(data.shape[0]*0.9)
    for i in range(10):
        resample_data = resample(data, n_samples=len_samples, replace=False, stratify=real,
                random_state=42*i)

        real = resample_data['Real']
        
        acc_gen = f1_score(real, resample_data['Gen Pred'].values,average='weighted')
        acc_cnn = f1_score(real, resample_data['CNN Pred'].values,average='weighted')
        accs_cnn.append(acc_cnn)
        accs_gen.append(acc_gen)

    alpha_gen = np.mean(acc_gen) / (np.mean(acc_gen) + np.mean(acc_cnn))
    alpha_cnn = np.mean(acc_cnn) / (np.mean(acc_gen) + np.mean(acc_cnn))
    
    return alpha_cnn, alpha_gen

def get_test_df(df_gen: pd.DataFrame, df_cnn: pd.DataFrame, alpha: float) -> float:
    case_ids_cnn = df_cnn['Case_Ids'].values
    case_ids_gen = df_gen['Case_Ids'].values
    case_ids = np.intersect1d(case_ids_cnn, case_ids_gen)
    cnn_df_only = df_cnn.loc[df_cnn['Case_Ids'].isin(case_ids)]
    gen_df_only = df_gen.loc[df_gen['Case_Ids'].isin(case_ids)]
    real = gen_df_only['Real'].values
    case_ids_df = gen_df_only['Case_Ids'].values

    data = pd.DataFrame()
    data['Case_Ids'] = case_ids_df
    data['CNN Prob LUAD'] = cnn_df_only['Prob LUAD'].values
    data['CNN Prob HLT'] = cnn_df_only['Prob HLT'].values
    data['CNN Prob LUSC'] = cnn_df_only['Prob LUSC'].values
    data['Gen Prob LUAD'] = gen_df_only['Prob LUAD'].values
    data['Gen Prob HLT'] = gen_df_only['Prob HLT'].values
    data['Gen Prob LUSC'] = gen_df_only['Prob LUSC'].values
    data['CNN Pred'] = data[['CNN Prob LUAD', 'CNN Prob HLT', 'CNN Prob LUSC']].idxmax(axis=1).values
    data['CNN Pred'].replace({'CNN Prob LUAD': 0, 'CNN Prob HLT': 1, 'CNN Prob LUSC': 2}, inplace=True)
    data['Gen Pred'] = data[['Gen Prob LUAD', 'Gen Prob HLT', 'Gen Prob LUSC']].idxmax(axis=1).values
    data['Gen Pred'].replace({'Gen Prob LUAD': 0, 'Gen Prob HLT': 1, 'Gen Prob LUSC': 2}, inplace=True)
    
    luad_prob = data['CNN Prob LUAD']*alpha[0]+alpha[1]*data['Gen Prob LUAD']
    hlt_prob = data['CNN Prob HLT']*alpha[0]+alpha[1]*data['Gen Prob HLT']
    lusc_prob = data['CNN Prob LUSC']*alpha[0]+alpha[1]*data['Gen Prob LUSC']

    data['Combined Prob LUAD'] = luad_prob
    data['Combined Prob HLT'] = hlt_prob
    data['Combined Prob LUSC'] = lusc_prob
    data['Combined Pred'] = data[['Combined Prob LUAD', 'Combined Prob HLT', 'Combined Prob LUSC']].idxmax(axis=1).values
    data['Combined Pred'].replace({'Combined Prob LUAD': 0, 'Combined Prob HLT': 1, 'Combined Prob LUSC': 2}, inplace=True)

    data['Real'] = real
    acc = accuracy_score(real, data['Combined Pred'].values)
    print('{} Test acc {}'.format(alpha, acc))
    return acc, data

for n_genes in [3,6,10]:
    gen_folder = str(n_genes)+'/'
    dataset = ['train', 'test']
    if not os.path.isdir('prob_alpha_'+str(n_genes)+'_range'):
        os.mkdir('prob_alpha_'+str(n_genes)+'_range')
    
    accs = []
    d = 'test'
    alphas_cnn = []
    alphas_gen = []
    writer = pd.ExcelWriter('prob_alpha_'+str(n_genes)+'_range/'+d+'_prob_range_mean.xlsx', engine='openpyxl') 
    accs_alphas = {}
    for i in tqdm(range(10)):
        name = 'split_'+str(i)
        d = 'train'
        cnn_df = pd.read_excel('CNN/resnet_10cv_split'+str(i)+'_'+d+'.xlsx')
        gen_df = pd.read_excel(gen_folder+'genes_splits_rangegenes_split'+str(i)+'_'+str(n_genes)+'gen_'+d+'_mrmr.xlsx')

        alpha1, alpha2 = get_alpha_f1s(gen_df, cnn_df)
        alpha1 = round(alpha1, 2)
        alpha2 = round(alpha2, 2)
        alphas_cnn.append(alpha1)
        alphas_gen.append(alpha2)

        d = 'test'
        cnn_df_test = pd.read_excel('CNN/resnet_10cv_split'+str(i)+'_'+d+'.xlsx')
        gen_df_test = pd.read_excel(gen_folder+'genes_splits_rangegenes_split'+str(i)+'_'+str(n_genes)+'gen_'+d+'_mrmr.xlsx')
 
        acc_test, data_test = get_test_df(gen_df_test, cnn_df_test, [alpha1,alpha2])
        accs.append(acc_test) 
        data_test.to_excel(writer, sheet_name=name)
    
    writer.save()
