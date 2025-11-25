#!/usr/bin/env python3
"""
Enhanced Single Cell Immune Aging Neural Network Training Script
支持智能的skip_training模式，可以独立进行数据分析
支持多个数据集和多个细胞类型的缓存和分析
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
import umap
import pickle
import os
import argparse
from rich import print
from tqdm import tqdm
import warnings
import scanpy as sc
import hashlib
import json
from collections import defaultdict
# import flashinfer
import re
import random


# 设置matplotlib参数
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(42)

# ===============================
# 数据加载和预处理
# ===============================

def load_h5ad_data_selective(data_dir, datasets):
    """只加载指定的h5ad数据文件"""
    h5ad_files = {
        'AIDA': os.path.join(data_dir, 'AIDA_protein.h5ad'),
        'eQTL': os.path.join(data_dir, 'eQTL_protein.h5ad'),
        'HCA': os.path.join(data_dir, 'HCA_protein.h5ad'),
        'siAge': os.path.join(data_dir, 'siAge_protein.h5ad')
    }
    
    adata_dict = {}
    for dataset in datasets:  # 只加载指定的数据集
        if dataset in h5ad_files:
            file_path = h5ad_files[dataset]
            if os.path.exists(file_path):
                print(f"Loading {dataset} from {file_path}...")
                adata_dict[dataset] = sc.read_h5ad(file_path)
                print(f"Loaded {dataset}: {adata_dict[dataset].shape}")
            else:
                print(f"Warning: {file_path} not found")
        else:
            print(f"Warning: Unknown dataset {dataset}")
    
    return adata_dict

def load_scimilarity_annotations_optimized(data_dir, datasets=None, celltypes=None):
    """加载SCimilarity注释数据（深度优化版本）"""
    scimilarity_file = os.path.join(data_dir, 'sc_merge4_with_scGPT_cell_emb.h5ad')
    if os.path.exists(scimilarity_file):
        print(f"Loading SCimilarity annotations from {scimilarity_file}...")
        adata_merged = sc.read_h5ad(scimilarity_file)
        
        total_obs = adata_merged.obs.copy()
        total_obs['dataset'] = total_obs['batch'].str.split('_').str[0]
        
        # 如果指定了数据集，先过滤
        if datasets is not None:
            print(f"Filtering for datasets: {datasets}")
            if 'AIDA' in datasets:
                # AIDA特殊处理
                dataset_mask = total_obs['batch'].isin(['AIDA_1', 'AIDA_2'])
                for dataset in datasets:
                    if dataset != 'AIDA':
                        dataset_mask |= (total_obs['batch'] == dataset)
            else:
                dataset_mask = total_obs['batch'].isin(datasets)
            
            total_obs = total_obs[dataset_mask]
            print(f"Filtered to {len(total_obs)} cells from specified datasets")
        
        # 如果指定了细胞类型，进一步过滤
        if celltypes is not None:
            celltype_map = {
                'CD4T': 'CD4-positive, alpha-beta T cell',
                'CD8T': 'CD8-positive, alpha-beta T cell',
                'macrophage': 'macrophage',
                'monocyte': 'monocyte',
                'NK': 'natural killer cell'
            }
            target_celltypes = [celltype_map.get(ct, ct) for ct in celltypes]
            
            print(f"Filtering for celltypes: {target_celltypes}")
            celltype_mask = total_obs['celltype_hint'].isin(target_celltypes)
            total_obs = total_obs[celltype_mask]
            print(f"Filtered to {len(total_obs)} cells of specified celltypes")
        
        celltype_mapping = {}
        
        # 如果没有指定datasets，则加载所有数据集（保持向后兼容）
        if datasets is None:
            datasets_to_process = ['AIDA', 'eQTL', 'HCA', 'siAge']
        else:
            datasets_to_process = datasets
        
        print(f"Building celltype mapping for datasets: {datasets_to_process}")
        
        for dataset in datasets_to_process:
            if dataset == 'AIDA':
                dataset_obs = total_obs[total_obs['batch'].isin(['AIDA_1', 'AIDA_2'])]
            else:
                dataset_obs = total_obs[total_obs['batch'] == dataset]
            
            if len(dataset_obs) > 0:
                dataset_obs.index = dataset_obs.index.str.replace('-[^-]+$', '', regex=True)
                celltype_mapping[dataset] = dataset_obs['celltype_hint']
                print(f"  {dataset}: {len(dataset_obs)} cells with celltype annotations")
            else:
                print(f"  Warning: No cells found for dataset {dataset}")
        
        print(f"Celltype mapping built for {len(celltype_mapping)} datasets")
        return celltype_mapping
    else:
        print(f"Warning: SCimilarity annotation file not found: {scimilarity_file}")
        return {}

def create_bulk_data_for_visualization(cell_data, ages, gene_names, donor_dataset_mapping):
    """创建用于可视化的bulk数据"""
    bulk_data = pd.DataFrame(index=list(cell_data.keys()), columns=gene_names)
    
    for donor_id, cells in cell_data.items():
        # 计算每个donor的平均表达
        mean_expr = np.mean(cells, axis=0)
        bulk_data.loc[donor_id] = mean_expr
    
    # 添加年龄和数据集信息
    bulk_data['age'] = [ages[donor_id] for donor_id in bulk_data.index]
    bulk_data['dataset'] = [donor_dataset_mapping.get(donor_id, 'Unknown') for donor_id in bulk_data.index]
    
    # 创建年龄组
    bins = [0] + list(range(20, 81, 5)) + [100]
    labels = ["<20"] + [f"{i}-{i+5}" for i in range(20, 80, 5)] + ["≥80"]
    bulk_data['age_group'] = pd.cut(
        bulk_data['age'], bins=bins, labels=labels, right=False, include_lowest=True
    )
    
    return bulk_data

# ===============================
# 参考模型验证功能
# ===============================

def validate_with_reference_lasso(cell_data, ages, train_ids, test_ids, celltypes_str, save_path=None):
    """使用参考代码中的LASSO模型验证数据划分一致性"""
    print(f"\n=== Reference LASSO Validation - {celltypes_str} ===")
    
    # 创建bulk数据（平均表达）
    bulk_data = {}
    for donor_id, cells in cell_data.items():
        bulk_data[donor_id] = np.mean(cells, axis=0)
    
    # 准备训练和测试数据
    train_X = np.array([bulk_data[donor_id] for donor_id in train_ids if donor_id in bulk_data])
    train_y = np.array([ages[donor_id] for donor_id in train_ids if donor_id in bulk_data])
    test_X = np.array([bulk_data[donor_id] for donor_id in test_ids if donor_id in bulk_data])
    test_y = np.array([ages[donor_id] for donor_id in test_ids if donor_id in bulk_data])
    
    print(f"Training data: {train_X.shape}")
    print(f"Test data: {test_X.shape}")
    
    # 训练LASSO模型
    lasso = LassoCV(cv=5, random_state=114514, max_iter=2000)
    lasso.fit(train_X, train_y)
    
    # 预测
    train_pred = lasso.predict(train_X)
    test_pred = lasso.predict(test_X)
    
    # 计算指标
    train_corr, train_p = pearsonr(train_y, train_pred)
    test_corr, test_p = pearsonr(test_y, test_pred)
    
    print(f"LASSO Results:")
    print(f"  Training Correlation: {train_corr:.4f} (p={train_p:.2e})")
    print(f"  Test Correlation: {test_corr:.4f} (p={test_p:.2e})")
    print(f"  Test RMSE: {np.sqrt(mean_squared_error(test_y, test_pred)):.4f}")
    
    # 可视化
    if save_path:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 训练集
        axes[0].scatter(train_y, train_pred, alpha=0.6, color='blue', s=30)
        axes[0].plot([train_y.min(), train_y.max()], [train_y.min(), train_y.max()], 'k--', alpha=0.8)
        axes[0].set_xlabel('True Age')
        axes[0].set_ylabel('Predicted Age')
        axes[0].set_title(f'LASSO Training Set\nR = {train_corr:.3f}')
        axes[0].grid(True, alpha=0.3)
        
        # 测试集
        axes[1].scatter(test_y, test_pred, alpha=0.6, color='red', s=30)
        axes[1].plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', alpha=0.8)
        axes[1].set_xlabel('True Age')
        axes[1].set_ylabel('Predicted Age')
        axes[1].set_title(f'LASSO Test Set\nR = {test_corr:.3f}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    return {
        'train_correlation': train_corr,
        'test_correlation': test_corr,
        'train_rmse': np.sqrt(mean_squared_error(train_y, train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(test_y, test_pred)),
        'lasso_model': lasso
    }

# ===============================
# 优化的数据缓存机制
# ===============================

def get_data_cache_key(datasets, celltypes):
    """生成数据缓存的键（支持多个细胞类型）"""
    key_data = {
        'datasets': sorted(datasets),
        'celltypes': sorted(celltypes),  # 支持多个细胞类型
        'version': '2.0'  # 更新版本
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

def save_extracted_data_cache(cache_key, cell_data, ages, gene_names, donor_dataset_mapping, donor_celltype_counts, cache_dir):
    """保存提取的数据缓存（包含细胞类型计数）"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"extracted_data_{cache_key}.pkl")
    
    cache_data = {
        'cell_data': cell_data,
        'ages': ages,
        'gene_names': gene_names,
        'donor_dataset_mapping': donor_dataset_mapping,
        'donor_celltype_counts': donor_celltype_counts  # 新增
    }
    
    print(f"Saving extracted data cache to: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"Extracted data cache saved successfully!")

def load_extracted_data_cache(cache_key, cache_dir):
    """加载提取的数据缓存"""
    cache_file = os.path.join(cache_dir, f"extracted_data_{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        print(f"Loading extracted data cache from: {cache_file}")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        print(f"Extracted data cache loaded successfully!")
        return (cache_data['cell_data'], 
                cache_data['ages'], 
                cache_data['gene_names'], 
                cache_data['donor_dataset_mapping'],
                cache_data.get('donor_celltype_counts', {}))  # 向后兼容
    
    return None, None, None, None, None

def get_embedding_cache_key(datasets, celltypes, embedding_type):
    """生成嵌入数据缓存的键"""
    key_data = {
        'datasets': sorted(datasets),
        'celltypes': sorted(celltypes),
        'embedding_type': embedding_type,
        'version': '1.0'
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

def save_embedding_cache(cache_key, cell_data, ages, feature_names, donor_dataset_mapping, 
                        donor_celltype_counts, cache_dir, embedding_type):
    """保存嵌入数据缓存"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"embedding_{embedding_type}_{cache_key}.pkl")
    
    cache_data = {
        'cell_data': cell_data,
        'ages': ages,
        'feature_names': feature_names,
        'donor_dataset_mapping': donor_dataset_mapping,
        'donor_celltype_counts': donor_celltype_counts,
        'embedding_type': embedding_type
    }
    
    print(f"Saving {embedding_type} embedding cache to: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"{embedding_type} embedding cache saved successfully!")

def load_embedding_cache(cache_key, cache_dir, embedding_type):
    """加载嵌入数据缓存"""
    cache_file = os.path.join(cache_dir, f"embedding_{embedding_type}_{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        print(f"Loading {embedding_type} embedding cache from: {cache_file}")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        print(f"{embedding_type} embedding cache loaded successfully!")
        return (cache_data['cell_data'], 
                cache_data['ages'], 
                cache_data['feature_names'], 
                cache_data['donor_dataset_mapping'],
                cache_data.get('donor_celltype_counts', {}))
    
    return None, None, None, None, None

def load_scgpt_embeddings_with_cache(data_dir, datasets, celltypes, cache_dir):
    """加载scGPT嵌入数据（带缓存）"""
    
    # 检查缓存
    cache_key = get_embedding_cache_key(datasets, celltypes, 'scgpt')
    cell_data, ages, feature_names, donor_dataset_mapping, donor_celltype_counts = load_embedding_cache(
        cache_key, cache_dir, 'scgpt'
    )
    
    if cell_data is not None:
        print("✅ Using cached scGPT embeddings!")
        return cell_data, ages, feature_names, donor_dataset_mapping, donor_celltype_counts
    
    print("❌ No scGPT embedding cache found, loading from h5ad...")
    
    scgpt_file = os.path.join(data_dir, 'sc_merge4_with_scGPT_cell_emb.h5ad')
    
    if not os.path.exists(scgpt_file):
        raise FileNotFoundError(f"scGPT embedding file not found: {scgpt_file}")
    
    print(f"Loading scGPT embeddings from {scgpt_file}...")
    adata = sc.read_h5ad(scgpt_file)
    
    # 提取数据
    cell_data, ages, donor_dataset_mapping, donor_celltype_counts = extract_embedding_data_with_celltypes(
        adata, 'X_scGPT', celltypes, datasets
    )
    
    # scGPT嵌入是512维
    feature_names = [f"scgpt_emb_{i:03d}" for i in range(512)]
    
    # 保存缓存
    save_embedding_cache(cache_key, cell_data, ages, feature_names, donor_dataset_mapping, 
                        donor_celltype_counts, cache_dir, 'scgpt')
    
    print(f"Extracted scGPT embeddings for {len(cell_data)} donors")
    return cell_data, ages, feature_names, donor_dataset_mapping, donor_celltype_counts

def load_scimilarity_embeddings_with_cache(data_dir, datasets, celltypes, cache_dir):
    """加载SCimilarity嵌入数据（带缓存）"""
    
    # 检查缓存
    cache_key = get_embedding_cache_key(datasets, celltypes, 'scimilarity')
    cell_data, ages, feature_names, donor_dataset_mapping, donor_celltype_counts = load_embedding_cache(
        cache_key, cache_dir, 'scimilarity'
    )
    
    if cell_data is not None:
        print("✅ Using cached SCimilarity embeddings!")
        return cell_data, ages, feature_names, donor_dataset_mapping, donor_celltype_counts
    
    print("❌ No SCimilarity embedding cache found, loading from h5ad...")
    
    scimilarity_file = os.path.join(data_dir, 'sc_merge4_with_scGPT_cell_emb.h5ad')
    
    if not os.path.exists(scimilarity_file):
        raise FileNotFoundError(f"SCimilarity embedding file not found: {scimilarity_file}")
    
    print(f"Loading SCimilarity embeddings from {scimilarity_file}...")
    adata = sc.read_h5ad(scimilarity_file)
    
    # 检查是否有SCimilarity嵌入
    if 'X_scimilarity' not in adata.obsm:
        raise ValueError("SCimilarity embeddings not found in the file")
    
    # 提取数据
    cell_data, ages, donor_dataset_mapping, donor_celltype_counts = extract_embedding_data_with_celltypes(
        adata, 'X_scimilarity', celltypes, datasets
    )
    
    # SCimilarity嵌入维度
    emb_dim = list(cell_data.values())[0].shape[1] if cell_data else 0
    feature_names = [f"scimilarity_emb_{i:03d}" for i in range(emb_dim)]
    
    # 保存缓存
    save_embedding_cache(cache_key, cell_data, ages, feature_names, donor_dataset_mapping, 
                        donor_celltype_counts, cache_dir, 'scimilarity')
    
    print(f"Extracted SCimilarity embeddings for {len(cell_data)} donors, dim={emb_dim}")
    return cell_data, ages, feature_names, donor_dataset_mapping, donor_celltype_counts

def load_data_unified(data_dir, celltypes, datasets, embedding_type, cache_dir='./data_cache', force_reload=False):
    """统一的数据加载函数，支持所有数据类型和缓存"""
    
    # 如果强制重新加载，删除相关缓存
    if force_reload:
        if embedding_type in ['scgpt', 'scimilarity']:
            cache_key = get_embedding_cache_key(datasets, celltypes, embedding_type)
            cache_file = os.path.join(cache_dir, f"embedding_{embedding_type}_{cache_key}.pkl")
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"Removed existing {embedding_type} embedding cache: {cache_file}")
        else:
            # 原始基因表达数据的缓存
            cache_key = get_data_cache_key(datasets, celltypes)
            cache_file = os.path.join(cache_dir, f"extracted_data_{cache_key}.pkl")
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"Removed existing gene expression cache: {cache_file}")
    
    # 根据嵌入类型加载数据
    if embedding_type == 'scgpt':
        return load_scgpt_embeddings_with_cache(data_dir, datasets, celltypes, cache_dir)
    elif embedding_type == 'scimilarity':
        return load_scimilarity_embeddings_with_cache(data_dir, datasets, celltypes, cache_dir)
    else:
        # 原始基因表达数据
        return load_data_with_cache(data_dir, celltypes, datasets, cache_dir)

def parse_age_from_development_stage(development_stage):
    """统一的年龄解析函数，处理各种年龄格式"""
    if pd.isna(development_stage):
        return None
    
    age_str = str(development_stage).strip()
    original_age_str = age_str  # 保存原始格式用于错误信息
    age_str_lower = age_str.lower()
    
    # 1. 处理 "60-year-old stage" 格式
    year_old_match = re.search(r'(\d+)-year-old', age_str_lower)
    if year_old_match:
        return float(year_old_match.group(1))
    
    # 2. 处理纯数字格式
    if re.match(r'^\d+$', age_str.strip()):
        return float(age_str)
    
    # 3. 处理数字-数字格式 (如 "25-30")，但排除包含特殊关键词的情况
    if '-' in age_str and not any(keyword in age_str_lower for keyword in ['stage', 'year-old', 'decade']):
        try:
            return float(age_str.split('-')[0])
        except ValueError:
            pass
    
    # 4. 处理描述性年龄格式 (如 "eighth decade stage")
    decade_mapping = {
        'first decade stage': 10,
        'second decade stage': 20,
        'third decade stage': 30,
        'fourth decade stage': 40,
        'fifth decade stage': 50,
        'sixth decade stage': 60,
        'seventh decade stage': 70,
        'eighth decade stage': 80,
        'ninth decade stage': 90,
        'tenth decade stage': 100,
        # 也支持不带stage的格式
        'first decade': 10,
        'second decade': 20,
        'third decade': 30,
        'fourth decade': 40,
        'fifth decade': 50,
        'sixth decade': 60,
        'seventh decade': 70,
        'eighth decade': 80,
        'ninth decade': 90,
        'tenth decade': 100,
    }
    
    if age_str_lower in decade_mapping:
        return float(decade_mapping[age_str_lower])
    
    # 5. 最后尝试提取任何数字作为年龄
    numbers = re.findall(r'\d+', age_str)
    if numbers:
        try:
            # 取第一个数字，通常是年龄
            return float(numbers[0])
        except ValueError:
            pass
    
    # 如果都无法解析，返回None
    print(f"Warning: Could not parse age format: '{original_age_str}'")
    return None

def extract_multiple_celltypes_data(adata_dict, celltype_mapping, celltypes, datasets=['AIDA', 'eQTL']):
    """提取多个细胞类型的数据（优化年龄处理）"""
    
    celltype_map = {
        'CD4T': 'CD4-positive, alpha-beta T cell',
        'CD8T': 'CD8-positive, alpha-beta T cell',
        'macrophage': 'macrophage',
        'monocyte': 'monocyte',
        'NK': 'natural killer cell'
    }
    
    # 将输入的细胞类型转换为标准名称
    target_celltypes = [celltype_map.get(ct, ct) for ct in celltypes]
    
    # 获取所有数据集的共同基因
    gene_intersections = []
    for dataset in datasets:
        if dataset in adata_dict:
            gene_intersections.append(adata_dict[dataset].var_names)
    
    if not gene_intersections:
        raise ValueError(f"No valid datasets found in {datasets}")
    
    common_genes = gene_intersections[0]
    for genes in gene_intersections[1:]:
        common_genes = common_genes.intersection(genes)
    
    print(f"Common genes across datasets: {len(common_genes)}")
    
    # 存储结果
    cell_data = {}
    ages = {}
    donor_dataset_mapping = {}
    donor_celltype_counts = {}
    
    for dataset in datasets:
        if dataset not in adata_dict:
            continue
            
        adata = adata_dict[dataset]
        print(f"Processing {dataset} dataset...")
        
        if dataset in celltype_mapping:
            celltype_series = celltype_mapping[dataset]
            
            # 处理 Categorical 类型的 Series
            if hasattr(celltype_series, 'cat'):
                if 'unknown' not in celltype_series.cat.categories:
                    celltype_series = celltype_series.cat.add_categories(['unknown'])
                celltype_series = celltype_series.reindex(adata.obs_names, fill_value='unknown')
            else:
                celltype_series = celltype_series.reindex(adata.obs_names, fill_value='unknown')
            
            adata.obs['celltype_scimilarity'] = celltype_series
        else:
            print(f"Warning: No celltype mapping found for {dataset}")
            continue
        
        # 筛选目标细胞类型
        target_cells_mask = adata.obs['celltype_scimilarity'].isin(target_celltypes)
        if target_cells_mask.sum() == 0:
            print(f"Warning: No target cells found in {dataset}")
            continue
        
        print(f"Found {target_cells_mask.sum()} target cells in {dataset}")
        adata_subset = adata[target_cells_mask, common_genes]
        
        # 处理年龄信息（优化版本）
        if 'age' in adata.obs.columns:
            age_col = 'age'
        elif 'development_stage' in adata.obs.columns:
            print(f"Processing development_stage for {dataset}...")
            # 使用新的年龄解析函数
            parsed_ages = []
            failed_count = 0
            
            for stage in adata.obs['development_stage']:
                parsed_age = parse_age_from_development_stage(stage)
                if parsed_age is not None:
                    parsed_ages.append(parsed_age)
                else:
                    parsed_ages.append(np.nan)
                    failed_count += 1
            
            if failed_count > 0:
                print(f"Warning: Failed to parse {failed_count} age values in {dataset}")
            
            adata.obs['age'] = parsed_ages
            age_col = 'age'
        else:
            print(f"Warning: No age information found in {dataset}")
            continue
        
        # 按donor处理
        donor_ids = adata_subset.obs['donor_id'].unique()
        print(f"Processing {len(donor_ids)} donors in {dataset}...")
        
        for donor_id in tqdm(donor_ids, desc=f"Processing {dataset} donors"):
            if donor_id == 'MH8919227':  # 跳过特定donor
                continue
                
            donor_cells = adata_subset[adata_subset.obs['donor_id'] == donor_id]
            if donor_cells.shape[0] == 0:
                continue
            
            # 检查年龄是否有效
            donor_age = donor_cells.obs[age_col].iloc[0]
            if pd.isna(donor_age):
                print(f"Warning: No valid age for donor {donor_id}, skipping")
                continue
            
            # 统计每个donor的细胞类型数量
            celltype_counts = donor_cells.obs['celltype_scimilarity'].value_counts().to_dict()
            
            # 获取细胞表达数据
            if hasattr(donor_cells.X, 'toarray'):
                cell_expressions = donor_cells.X.toarray()
            else:
                cell_expressions = donor_cells.X
            
            # 存储数据
            if donor_id in cell_data:
                # 如果已经有这个donor的数据，合并
                cell_data[donor_id] = np.vstack([cell_data[donor_id], cell_expressions.astype(np.float32)])
                # 合并细胞类型计数
                for ct, count in celltype_counts.items():
                    donor_celltype_counts[donor_id][ct] = donor_celltype_counts[donor_id].get(ct, 0) + count
            else:
                # 新donor
                cell_data[donor_id] = cell_expressions.astype(np.float32)
                ages[donor_id] = float(donor_age)
                donor_dataset_mapping[donor_id] = dataset
                donor_celltype_counts[donor_id] = celltype_counts
    
    print(f"Extracted data for {len(cell_data)} donors")
    if ages:
        print(f"Age range: {min(ages.values()):.1f} - {max(ages.values()):.1f}")
    
    return cell_data, ages, list(common_genes), donor_dataset_mapping, donor_celltype_counts

def extract_embedding_data_with_celltypes(adata, embedding_key, celltypes, datasets):
    """从adata中提取嵌入数据，同时统计细胞类型（优化年龄处理）"""
    
    # 过滤数据集
    adata.obs['dataset'] = adata.obs['batch'].str.split('_').str[0]
    
    if 'AIDA' in datasets:
        dataset_mask = adata.obs['batch'].isin(['AIDA_1', 'AIDA_2'])
        for dataset in datasets:
            if dataset != 'AIDA':
                dataset_mask |= (adata.obs['batch'] == dataset)
    else:
        dataset_mask = adata.obs['batch'].isin(datasets)
    
    adata = adata[dataset_mask]
    
    # 细胞类型过滤
    celltype_map = {
        'CD4T': 'CD4-positive, alpha-beta T cell',
        'CD8T': 'CD8-positive, alpha-beta T cell',
        'macrophage': 'macrophage',
        'monocyte': 'monocyte',
        'NK': 'natural killer cell'
    }
    target_celltypes = [celltype_map.get(ct, ct) for ct in celltypes]
    celltype_mask = adata.obs['celltype_hint'].isin(target_celltypes)
    adata = adata[celltype_mask]
    
    # 排除特定donor
    adata = adata[adata.obs['donor_id'] != 'MH8919227']
    
    print(f"Filtered to {adata.shape[0]} cells")
    
    # 处理年龄信息（优化版本）
    if 'age' in adata.obs.columns:
        age_col = 'age'
    elif 'development_stage' in adata.obs.columns:
        print("Processing development_stage...")
        # 使用新的年龄解析函数
        parsed_ages = []
        failed_count = 0
        
        for stage in adata.obs['development_stage']:
            parsed_age = parse_age_from_development_stage(stage)
            if parsed_age is not None:
                parsed_ages.append(parsed_age)
            else:
                parsed_ages.append(np.nan)
                failed_count += 1
        
        if failed_count > 0:
            print(f"Warning: Failed to parse {failed_count} age values")
        
        adata.obs['age'] = parsed_ages
        age_col = 'age'
    else:
        print(f"Warning: No age information found")
        raise ValueError("No age column found")
    
    # 提取嵌入数据
    embeddings = adata.obsm[embedding_key]
    
    # 按donor聚合
    cell_data = {}
    ages = {}
    donor_dataset_mapping = {}
    donor_celltype_counts = {}
    
    for donor_id in tqdm(adata.obs['donor_id'].unique(), desc="Processing donors"):
        donor_mask = adata.obs['donor_id'] == donor_id
        donor_cells = adata[donor_mask]
        
        # 检查年龄是否有效
        donor_age = donor_cells.obs[age_col].iloc[0]
        if pd.isna(donor_age):
            print(f"Warning: No valid age for donor {donor_id}, skipping")
            continue
        
        # 获取嵌入
        donor_embeddings = embeddings[donor_mask.values]
        cell_data[donor_id] = donor_embeddings.astype(np.float32)
        
        # 获取年龄
        ages[donor_id] = float(donor_age)
        
        # 获取数据集
        dataset = donor_cells.obs['dataset'].iloc[0]
        donor_dataset_mapping[donor_id] = dataset
        
        # 统计细胞类型
        celltype_counts = donor_cells.obs['celltype_hint'].value_counts().to_dict()
        donor_celltype_counts[donor_id] = celltype_counts
    
    return cell_data, ages, donor_dataset_mapping, donor_celltype_counts

def load_data_with_cache(data_dir, celltypes, datasets, cache_dir='./data_cache'):
    """使用缓存机制加载数据（优化年龄处理）"""
    
    # 检查是否有提取数据的缓存
    cache_key = get_data_cache_key(datasets, celltypes)
    cell_data, ages, gene_names, donor_dataset_mapping, donor_celltype_counts = load_extracted_data_cache(cache_key, cache_dir)
    
    if cell_data is not None:
        print("✅ Using cached extracted data, skipping h5ad loading!")
        return cell_data, ages, gene_names, donor_dataset_mapping, donor_celltype_counts
    
    print("❌ No extracted data cache found, loading from h5ad files...")
    
    # 只加载需要的数据集
    print(f"Loading h5ad data for datasets: {datasets}")
    adata_dict = load_h5ad_data_selective(data_dir, datasets)
    
    # 深度优化：只为需要的数据集和细胞类型加载SCimilarity注释
    print("Loading SCimilarity annotations...")
    celltype_mapping = load_scimilarity_annotations_optimized(data_dir, datasets, celltypes)
    
    print(f"Extracting {celltypes} data...")
    cell_data, ages, gene_names, donor_dataset_mapping, donor_celltype_counts = extract_multiple_celltypes_data(
        adata_dict, celltype_mapping, celltypes, datasets
    )
    
    # 保存提取数据的缓存
    save_extracted_data_cache(cache_key, cell_data, ages, gene_names, donor_dataset_mapping, donor_celltype_counts, cache_dir)
    
    return cell_data, ages, gene_names, donor_dataset_mapping, donor_celltype_counts

# ===============================
# 数据划分缓存机制
# ===============================

def get_data_split_cache_key(datasets, celltypes, test_size, val_size, random_state):
    """生成数据划分的缓存键"""
    key_data = {
        'datasets': sorted(datasets),
        'celltypes': sorted(celltypes),
        'test_size': test_size,
        'val_size': val_size,
        'random_state': random_state
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

def save_data_split_cache(cache_key, train_ids, val_ids, test_ids, cache_dir):
    """保存数据划分缓存"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"data_split_{cache_key}.pkl")
    
    cache_data = {
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"Data split cache saved to: {cache_file}")

def load_data_split_cache(cache_key, cache_dir):
    """加载数据划分缓存"""
    cache_file = os.path.join(cache_dir, f"data_split_{cache_key}.pkl")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        print(f"Data split cache loaded from: {cache_file}")
        return cache_data['train_ids'], cache_data['val_ids'], cache_data['test_ids']
    
    return None, None, None

def create_data_split(cell_data, datasets, celltypes, test_size=0.2, val_size=0.1, 
                     random_state=114514, cache_dir='./data_cache'):
    """创建或加载数据划分"""
    
    # 生成缓存键
    cache_key = get_data_split_cache_key(datasets, celltypes, test_size, val_size, random_state)
    
    # 尝试加载缓存
    train_ids, val_ids, test_ids = load_data_split_cache(cache_key, cache_dir)
    
    if train_ids is None:
        # 如果没有缓存，创建新的数据划分
        donor_ids = list(cell_data.keys())
        print(f"Creating new data split for {len(donor_ids)} donors")
        
        # 使用与机器学习代码相同的划分方式
        train_val_ids, test_ids = train_test_split(
            donor_ids, test_size=test_size, random_state=random_state
        )
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=val_size/(1-test_size), random_state=random_state
        )
        
        # 保存缓存
        save_data_split_cache(cache_key, train_ids, val_ids, test_ids, cache_dir)
    
    print(f"Data split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    return train_ids, val_ids, test_ids

# ===============================
# 神经网络模型定义
# ===============================

class DeepSetsAgePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128, dropout=0.2, max_cells=1000):
        super().__init__()
        
        self.max_cells = max_cells
        self.output_dim = output_dim
        self.input_dim = input_dim  # 添加这个用于后续分析

        self.cell_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # 使用完整的max_cells * output_dim维度
        self.donor_aggregator = nn.Sequential(
            nn.Linear(max_cells * output_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, cells, mask=None):
        batch_size, batch_max_cells, gene_dim = cells.shape
        
        # 编码所有细胞
        cells_flat = cells.view(-1, gene_dim)
        cell_features = self.cell_encoder(cells_flat)
        cell_features = cell_features.view(batch_size, batch_max_cells, -1)

        # 应用掩码
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(cell_features)
            cell_features = cell_features * mask_expanded

        # 关键部分：处理不同的batch_max_cells
        # 需要填充到模型的max_cells大小
        padding_size = self.max_cells - batch_max_cells
        
        # 零填充
        if padding_size > 0:
            padding = torch.zeros(batch_size, padding_size, self.output_dim, 
                                device=cell_features.device, dtype=cell_features.dtype)
            cell_features_padded = torch.cat([cell_features, padding], dim=1)
        else:
            cell_features_padded = cell_features
        
        # 展平并通过聚合器
        donor_features = cell_features_padded.view(batch_size, -1)  # [B, max_cells * output_dim]
        age_pred = self.donor_aggregator(donor_features)
        return age_pred.view(-1)
    
    def get_cell_contributions(self, cells, mask=None, method='gradient'):
        """
        获取每个细胞对年龄预测的贡献
        
        Args:
            cells: [batch_size, n_cells, n_genes] 
            mask: [batch_size, n_cells] 
            method: 'gradient' | 'activation' | 'integrated_gradient'
            
        Returns:
            dict: {
                'cell_contributions': [batch_size, n_cells],  # 每个细胞的贡献分数
                'cell_features': [batch_size, n_cells, output_dim],  # 细胞编码特征
                'age_pred': [batch_size],  # 年龄预测
            }
        """
        self.eval()
        batch_size, batch_max_cells, gene_dim = cells.shape
        
        # 需要梯度
        cells.requires_grad_(True)
        
        with torch.enable_grad():
            # Forward pass获取中间特征
            cells_flat = cells.view(-1, gene_dim)
            cell_features = self.cell_encoder(cells_flat)
            cell_features = cell_features.view(batch_size, batch_max_cells, -1)
            
            # 应用掩码
            original_cell_features = cell_features.clone()
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand_as(cell_features)
                cell_features = cell_features * mask_expanded
            
            # 填充和聚合
            padding_size = self.max_cells - batch_max_cells
            if padding_size > 0:
                padding = torch.zeros(batch_size, padding_size, self.output_dim, 
                                    device=cell_features.device, dtype=cell_features.dtype)
                cell_features_padded = torch.cat([cell_features, padding], dim=1)
            else:
                cell_features_padded = cell_features
            
            donor_features = cell_features_padded.view(batch_size, -1)
            age_pred = self.donor_aggregator(donor_features).view(-1)
            
            if method == 'gradient':
                # 计算梯度贡献
                cell_contributions = []
                for i in range(batch_size):
                    # 对每个样本计算梯度
                    grad_outputs = torch.zeros_like(age_pred)
                    grad_outputs[i] = 1.0
                    
                    # 计算对cell_features的梯度
                    grads = torch.autograd.grad(
                        outputs=age_pred[i], 
                        inputs=original_cell_features,
                        create_graph=False,
                        retain_graph=True,
                        only_inputs=True
                    )[0]
                    
                    # 计算每个细胞的总贡献 (L2 norm of gradients)
                    cell_grad_norms = torch.norm(grads[i], dim=1)  # [n_cells]
                    
                    # 应用mask
                    if mask is not None:
                        cell_grad_norms = cell_grad_norms * mask[i]
                    
                    cell_contributions.append(cell_grad_norms)
                
                cell_contributions = torch.stack(cell_contributions)  # [batch_size, n_cells]
                
            elif method == 'activation':
                # 使用激活值幅度作为贡献
                cell_contributions = torch.norm(original_cell_features, dim=2)  # [batch_size, n_cells]
                if mask is not None:
                    cell_contributions = cell_contributions * mask
            
            else:
                raise ValueError(f"Unsupported method: {method}")
        
        return {
            'cell_contributions': cell_contributions.detach().cpu().numpy(),
            'cell_features': original_cell_features.detach().cpu().numpy(),
            'age_pred': age_pred.detach().cpu().numpy(),
        }
    
    def get_gene_contributions(self, cells, mask=None, method='gradient'):
        """
        获取每个基因对年龄预测的贡献
        
        Args:
            cells: [batch_size, n_cells, n_genes]
            mask: [batch_size, n_cells]
            method: 'gradient' | 'integrated_gradient'
            
        Returns:
            dict: {
                'gene_contributions': [batch_size, n_genes],  # 每个基因的平均贡献
                'cell_gene_contributions': [batch_size, n_cells, n_genes],  # 每个细胞中每个基因的贡献
                'age_pred': [batch_size],
            }
        """
        self.eval()
        batch_size, n_cells, n_genes = cells.shape
        
        cells.requires_grad_(True)
        
        with torch.enable_grad():
            age_pred = self(cells, mask)
            
            # 计算输入梯度
            cell_gene_contributions = []
            
            for i in range(batch_size):
                # 对每个样本计算关于输入的梯度
                grads = torch.autograd.grad(
                    outputs=age_pred[i],
                    inputs=cells,
                    create_graph=False,
                    retain_graph=True,
                    only_inputs=True
                )[0]
                
                # 取绝对值作为贡献
                sample_contrib = torch.abs(grads[i])  # [n_cells, n_genes]
                
                # 应用mask
                if mask is not None:
                    mask_expanded = mask[i].unsqueeze(-1).expand_as(sample_contrib)
                    sample_contrib = sample_contrib * mask_expanded
                
                cell_gene_contributions.append(sample_contrib)
            
            cell_gene_contributions = torch.stack(cell_gene_contributions)  # [batch_size, n_cells, n_genes]
            
            # 计算每个基因的总贡献（跨所有细胞求和）
            gene_contributions = cell_gene_contributions.sum(dim=1)  # [batch_size, n_genes]
        
        return {
            'gene_contributions': gene_contributions.detach().cpu().numpy(),
            'cell_gene_contributions': cell_gene_contributions.detach().cpu().numpy(),
            'age_pred': age_pred.detach().cpu().numpy(),
        }

class SimpleSetTransformerAgePredictor(nn.Module):
    def __init__(self, input_dim, d_model=256, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()

        if d_model % n_heads != 0:
            d_model = ((d_model // n_heads) + 1) * n_heads

        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

        self.pool_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.age_predictor = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.LayerNorm(d_model//2),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, cells, mask=None):
        batch_size = cells.shape[0]
        
        x = self.input_proj(cells)
        
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask.bool()

        for attn_layer, layer_norm in zip(self.self_attention_layers, self.layer_norms):
            residual = x
            attn_output, _ = attn_layer(x, x, x, key_padding_mask=attn_mask)
            x = layer_norm(attn_output + residual)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        pooled_output, _ = self.pool_attention(cls_token, x, x, key_padding_mask=attn_mask)
        pooled_output = pooled_output.squeeze(1)

        age_pred = self.age_predictor(pooled_output)
        return age_pred.view(-1)

def get_indptr_from_mask(mask, query):
    indptr = torch.zeros(mask.shape[0] + 1, device=query.device, dtype=torch.int32)
    indptr[0] = 0
    row_counts = mask.sum(dim=1).flatten()
    indptr[1:] = torch.cumsum(row_counts, dim=0)
    return indptr

def get_indices_from_mask(mask, query):
    nonzero_indices = torch.nonzero(mask)
    indices = nonzero_indices[:, 1].to(dtype=torch.int32, device=query.device)
    return indices

def manual_sparse_attention_simple(query, key, value, sparse_mask, block_size=32):
    """简化的手动稀疏注意力实现"""
    seq_len, num_heads, head_dim = query.shape
    num_blocks = seq_len // block_size
    
    # 重塑为块格式
    q_blocks = query.view(num_blocks, block_size, num_heads, head_dim)
    k_blocks = key.view(num_blocks, block_size, num_heads, head_dim)
    v_blocks = value.view(num_blocks, block_size, num_heads, head_dim)
    
    # 初始化输出
    output = torch.zeros_like(query)
    output_blocks = output.view(num_blocks, block_size, num_heads, head_dim)
    
    # 计算有效连接
    active_connections = torch.nonzero(sparse_mask, as_tuple=False)
    print(f"Manual sparse: {len(active_connections)} connections out of {num_blocks * num_blocks}")
    
    for connection in active_connections:
        i, j = connection[0].item(), connection[1].item()
        
        # 获取块并转换为标准注意力格式
        q_block = q_blocks[i].transpose(0, 1)  # [num_heads, block_size, head_dim]
        k_block = k_blocks[j].transpose(0, 1)  # [num_heads, block_size, head_dim]
        v_block = v_blocks[j].transpose(0, 1)  # [num_heads, block_size, head_dim]
        
        # 计算块内注意力
        attn_block = torch.nn.functional.scaled_dot_product_attention(q_block, k_block, v_block)
        
        # 转换回原格式并累加
        attn_block = attn_block.transpose(0, 1)  # [block_size, num_heads, head_dim]
        output_blocks[i] += attn_block
    
    return output

def debug_flashinfer_formats_fixed(query, key, value, sparse_mask, block_size=32):
    """修复后的FlashInfer格式调试"""
    seq_len, num_heads, head_dim = query.shape
    device = query.device
    
    print(f"\n=== FlashInfer Format Debug (Fixed) ===")
    print(f"Input: seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")
    print(f"Input dtype: {query.dtype}")
    print(f"Block size: {block_size}")
    
    # 生成稀疏结构
    indptr = get_indptr_from_mask(sparse_mask, query)
    indices = get_indices_from_mask(sparse_mask, query)
    
    print(f"Sparse structure: indptr={indptr.shape}, indices={indices.shape}")
    
    if len(indices) == 0:
        print("No sparse connections, using dense attention")
        return torch.nn.functional.scaled_dot_product_attention(
            query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        ).transpose(0, 1)
    
    # 创建workspace
    workspace_buffer = torch.empty(256 * 1024 * 1024, device=device, dtype=torch.uint8)
    
    # 尝试不同的数据类型
    dtypes_to_try = [torch.float16, torch.bfloat16, torch.float32]
    
    for dtype in dtypes_to_try:
        print(f"\n--- Testing dtype: {dtype} ---")
        
        try:
            # 转换数据类型
            q_test = query.to(dtype)
            k_test = key.to(dtype)
            v_test = value.to(dtype)
            
            print(f"Converted shapes: Q={q_test.shape}, K={k_test.shape}, V={v_test.shape}")
            
            # 创建wrapper
            wrapper = flashinfer.BlockSparseAttentionWrapper(workspace_buffer)
            
            # Plan阶段
            wrapper.plan(
                indptr=indptr,
                indices=indices,
                M=seq_len,
                N=seq_len,
                R=block_size,
                C=block_size,
                num_qo_heads=num_heads,
                num_kv_heads=num_heads,
                head_dim=head_dim,
            )
            print(f"✅ Plan succeeded with {dtype}")
            
            # 尝试正确的3D张量格式
            tensor_formats = [
                ("3D_seq_heads_dim", q_test, k_test, v_test),
                ("3D_heads_seq_dim", q_test.transpose(0, 1), k_test.transpose(0, 1), v_test.transpose(0, 1)),
            ]
            
            for format_name, q_fmt, k_fmt, v_fmt in tensor_formats:
                try:
                    print(f"  Trying format {format_name}: Q={q_fmt.shape}, K={k_fmt.shape}, V={v_fmt.shape}")
                    
                    # 确保是3D张量
                    assert q_fmt.dim() == 3, f"Q must be 3D, got {q_fmt.dim()}D"
                    assert k_fmt.dim() == 3, f"K must be 3D, got {k_fmt.dim()}D"
                    assert v_fmt.dim() == 3, f"V must be 3D, got {v_fmt.dim()}D"
                    
                    output = wrapper.run(q_fmt, k_fmt, v_fmt)
                    print(f"  ✅ Raw output shape: {output.shape}")
                    
                    # 转换回原始格式 [seq_len, num_heads, head_dim]
                    if format_name == "3D_heads_seq_dim":
                        output = output.transpose(0, 1)
                    
                    # 转换回原始数据类型
                    output = output.to(query.dtype)
                    
                    print(f"✅ SUCCESS with {dtype} and {format_name}")
                    print(f"Final output shape: {output.shape}")
                    return output
                    
                except Exception as e:
                    print(f"  ❌ Format {format_name} failed: {str(e)[:150]}...")
                    continue
            
        except Exception as e:
            print(f"❌ dtype {dtype} failed completely: {str(e)[:150]}...")
            continue
    
    print("❌ All FlashInfer attempts failed")
    return None

def flashinfer_with_correct_format(query, key, value, sparse_mask, block_size=32):
    """使用正确格式的FlashInfer实现"""
    orig_seqlen, num_heads, head_dim = query.shape
    original_dtype = query.dtype
    
    print(f"FlashInfer input check: Q={query.shape}, dtype={query.dtype}")
    
    # 填充到块大小的倍数
    padding_len = 0
    if orig_seqlen % block_size != 0:
        padded_len = ((orig_seqlen + block_size - 1) // block_size) * block_size
        padding_len = padded_len - orig_seqlen
        
        print(f"Padding from {orig_seqlen} to {padded_len}")
        
        padding = torch.zeros(padding_len, num_heads, head_dim, device=query.device, dtype=query.dtype)
        final_query = torch.cat([query, padding], dim=0)
        final_key = torch.cat([key, padding], dim=0)
        final_value = torch.cat([value, padding], dim=0)
        
        # 更新稀疏掩码大小
        new_num_blocks = padded_len // block_size
        old_num_blocks = sparse_mask.shape[0]
        
        if new_num_blocks > old_num_blocks:
            # 扩展稀疏掩码
            new_mask = torch.zeros(new_num_blocks, new_num_blocks, device=sparse_mask.device, dtype=sparse_mask.dtype)
            new_mask[:old_num_blocks, :old_num_blocks] = sparse_mask
            # 新增的块只连接自己（对角线）
            for i in range(old_num_blocks, new_num_blocks):
                new_mask[i, i] = True
            sparse_mask = new_mask
    else:
        final_query, final_key, final_value = query, key, value
    
    print(f"Final tensor shapes: Q={final_query.shape}, K={final_key.shape}, V={final_value.shape}")
    print(f"Sparse mask shape: {sparse_mask.shape}")
    
    # 调试FlashInfer格式
    output = debug_flashinfer_formats_fixed(final_query, final_key, final_value, sparse_mask, block_size)
    
    if output is not None:
        # 移除填充
        if padding_len > 0:
            output = output[:orig_seqlen]
        return output
    
    # 如果FlashInfer失败，使用手动稀疏注意力
    print("🔄 Falling back to manual sparse attention")
    manual_output = manual_sparse_attention_simple(final_query, final_key, final_value, sparse_mask, block_size)
    
    if padding_len > 0:
        manual_output = manual_output[:orig_seqlen]
    
    return manual_output

def create_better_sparse_mask(seq_len, block_size, sparsity=0.1, device='cuda'):
    """创建更好的随机稀疏掩码"""
    num_blocks = seq_len // block_size
    mask = torch.zeros(num_blocks, num_blocks, device=device, dtype=torch.bool)
    
    # 1. 确保对角线连接（自注意力）
    for i in range(num_blocks):
        mask[i, i] = True
    
    # 2. 添加一些局部连接（相邻块）
    for i in range(num_blocks - 1):
        if torch.rand(1).item() < 0.5:  # 50%概率连接相邻块
            mask[i, i + 1] = True
            mask[i + 1, i] = True
    
    # 3. 添加随机长距离连接
    total_possible = num_blocks * num_blocks
    diagonal_connections = num_blocks
    current_connections = mask.sum().item()
    
    target_connections = int(total_possible * sparsity)
    remaining_connections = max(0, target_connections - current_connections)
    
    if remaining_connections > 0:
        # 获取所有未连接的位置
        unconnected_positions = []
        for i in range(num_blocks):
            for j in range(num_blocks):
                if not mask[i, j]:
                    unconnected_positions.append((i, j))
        
        if len(unconnected_positions) > 0:
            # 随机选择连接
            num_to_add = min(remaining_connections, len(unconnected_positions))
            selected_indices = torch.randperm(len(unconnected_positions))[:num_to_add]
            
            for idx in selected_indices:
                i, j = unconnected_positions[idx]
                mask[i, j] = True
    
    actual_sparsity = 1 - mask.sum().float() / mask.numel()
    print(f"Created sparse mask: {num_blocks}x{num_blocks} blocks, sparsity={actual_sparsity:.3f}")
    
    return mask

def pad_batch_to_sparse_compatible(cells, masks, target_block_size=32):
    """将整个batch填充到稀疏注意力兼容的长度"""
    batch_size, max_cells, input_dim = cells.shape
    
    # 找到batch中的最大有效长度
    if masks is not None:
        max_valid_length = int(masks.sum(dim=1).max().item())
    else:
        max_valid_length = max_cells
    
    # 调整到块大小的倍数，并确保足够大以使用稀疏注意力
    min_sparse_length = target_block_size * 2  # 至少2个块
    target_length = max(max_valid_length, min_sparse_length)
    target_length = ((target_length + target_block_size - 1) // target_block_size) * target_block_size
    
    print(f"Batch padding: max_valid={max_valid_length}, target={target_length}, block_size={target_block_size}")
    
    # 如果需要填充
    if target_length > max_cells:
        padding_length = target_length - max_cells
        
        # 填充cells
        padding_cells = torch.zeros(batch_size, padding_length, input_dim, 
                                   device=cells.device, dtype=cells.dtype)
        padded_cells = torch.cat([cells, padding_cells], dim=1)
        
        # 填充masks
        if masks is not None:
            padding_masks = torch.zeros(batch_size, padding_length, 
                                       device=masks.device, dtype=masks.dtype)
            padded_masks = torch.cat([masks, padding_masks], dim=1)
        else:
            padded_masks = torch.ones(batch_size, target_length, 
                                    device=cells.device, dtype=torch.float32)
    else:
        # 截断到目标长度
        padded_cells = cells[:, :target_length]
        if masks is not None:
            padded_masks = masks[:, :target_length]
        else:
            padded_masks = torch.ones(batch_size, target_length, 
                                    device=cells.device, dtype=torch.float32)
    
    return padded_cells, padded_masks, target_length

class BatchSparseAttentionLayer(nn.Module):
    """支持整个batch稀疏注意力的层"""
    
    def __init__(self, d_model, n_heads, dropout, sparse_config):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.sparse_config = sparse_config
        
        assert d_model % n_heads == 0
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 缓存掩码
        self.mask_cache = {}
        
    def get_sparse_mask(self, seq_len, device):
        """获取稀疏掩码"""
        block_size = self.sparse_config.get('block_size', 32)
        sparsity = self.sparse_config.get('sparsity', 0.1)
        
        cache_key = (seq_len, block_size, sparsity)
        if cache_key not in self.mask_cache:
            self.mask_cache[cache_key] = create_better_sparse_mask(
                seq_len, block_size, sparsity, device
            )
        
        return self.mask_cache[cache_key]
    
    def apply_sparse_attention_to_sample(self, query, key, value, attention_mask=None):
        """对单个样本应用稀疏注意力"""
        seq_len, num_heads, head_dim = query.shape
        device = query.device
        block_size = self.sparse_config.get('block_size', 32)
        
        # 检查是否使用稀疏注意力
        use_sparse = (
            self.sparse_config.get('sparse_type', 'dense') != 'dense' and
            seq_len >= block_size and
            seq_len % block_size == 0
        )
        
        if use_sparse:
            # 获取稀疏掩码
            sparse_mask = self.get_sparse_mask(seq_len, device)
            
            # 使用FlashInfer稀疏注意力
            output = flashinfer_with_correct_format(query, key, value, sparse_mask, block_size)
            
            if output is not None:
                # 应用attention mask（如果有的话）
                if attention_mask is not None:
                    # attention_mask: [seq_len]
                    mask_expanded = attention_mask.unsqueeze(-1).unsqueeze(-1)  # [seq_len, 1, 1]
                    output = output * mask_expanded
                
                return output
        
        # 回退到密集注意力
        if attention_mask is not None:
            # 创建causal mask
            causal_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
            causal_mask = causal_mask.expand(num_heads, seq_len, seq_len)
            causal_mask = causal_mask.unsqueeze(0)  # [1, num_heads, seq_len, seq_len]
        else:
            causal_mask = None
        
        output = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),  # [1, num_heads, seq_len, head_dim]
            key.transpose(0, 1).unsqueeze(0),    # [1, num_heads, seq_len, head_dim]
            value.transpose(0, 1).unsqueeze(0),  # [1, num_heads, seq_len, head_dim]
            attn_mask=causal_mask
        )
        
        return output.squeeze(0).transpose(0, 1)  # [seq_len, num_heads, head_dim]
        
    def forward(self, query, key, value, attention_mask=None):
        """
        Args:
            query: [batch_size, seq_len, d_model] 或 [seq_len, d_model] (单样本)
            key: [batch_size, seq_len, d_model] 或 [seq_len, d_model] (单样本)
            value: [batch_size, seq_len, d_model] 或 [seq_len, d_model] (单样本)
            attention_mask: [batch_size, seq_len] 或 [seq_len] (单样本)
        """
        
        # 处理单样本情况
        if query.dim() == 2:
            return self._forward_single(query, key, value, attention_mask)
        
        # 批处理情况
        batch_size, seq_len, _ = query.shape
        device = query.device
        
        print(f"BatchSparseAttention input: batch_size={batch_size}, seq_len={seq_len}")
        
        # 线性投影
        Q = self.q_proj(query).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # 处理每个batch样本
        outputs = []
        for b in range(batch_size):
            # 获取单个样本 [seq_len, num_heads, head_dim]
            q_sample = Q[b]
            k_sample = K[b]
            v_sample = V[b]
            
            # 获取attention mask
            if attention_mask is not None:
                mask_sample = attention_mask[b]  # [seq_len]
            else:
                mask_sample = None
            
            # 应用稀疏注意力
            output_sample = self.apply_sparse_attention_to_sample(
                q_sample, k_sample, v_sample, mask_sample
            )
            
            outputs.append(output_sample)
        
        # 合并batch结果
        batch_output = torch.stack(outputs, dim=0)  # [batch_size, seq_len, num_heads, head_dim]
        
        # 重塑并投影
        batch_output = batch_output.reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(batch_output)
        output = self.dropout(output)
        
        # 残差连接
        output = self.layer_norm(output + query)
        
        return output
    
    def _forward_single(self, query, key, value, attention_mask=None):
        """处理单样本的前向传播"""
        seq_len = query.shape[0]
        
        # 线性投影
        Q = self.q_proj(query).view(seq_len, self.n_heads, self.head_dim)
        K = self.k_proj(key).view(seq_len, self.n_heads, self.head_dim)
        V = self.v_proj(value).view(seq_len, self.n_heads, self.head_dim)
        
        # 应用稀疏注意力
        attn_output = self.apply_sparse_attention_to_sample(Q, K, V, attention_mask)
        
        # 重塑并投影
        attn_output = attn_output.reshape(seq_len, self.d_model)
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        # 残差连接
        output = self.layer_norm(output + query)
        
        return output

class UnifiedSparseTransformerAgePredictor(nn.Module):
    """统一填充的稀疏Transformer年龄预测器"""
    
    def __init__(self, input_dim, d_model=256, n_heads=8, n_layers=2, 
                 dropout=0.1, max_cells=1000, sparse_config=None):
        super().__init__()
        
        if d_model % n_heads != 0:
            d_model = ((d_model // n_heads) + 1) * n_heads
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_cells = max_cells
        
        # 稀疏配置
        self.sparse_config = sparse_config or {
            'sparse_type': 'random',
            'block_size': 32,
            'sparsity': 0.1,
            'model_type': 'single_cell'
        }
        
        print(f"Unified Sparse Transformer config: {self.sparse_config}")
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_cells, d_model) * 0.02)
        
        # 使用批处理稀疏注意力层
        self.attention_layers = nn.ModuleList([
            BatchSparseAttentionLayer(d_model, n_heads, dropout, self.sparse_config)
            for _ in range(n_layers)
        ])
        
        # 前馈网络层
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model)
            ) for _ in range(n_layers)
        ])
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 年龄预测头
        self.age_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1)
        )
    
    def forward(self, cells, mask=None):
        batch_size, max_cells, input_dim = cells.shape
        block_size = self.sparse_config.get('block_size', 32)
        
        print(f"UnifiedSparseTransformer input: batch_size={batch_size}, max_cells={max_cells}")
        
        # 统一填充到稀疏注意力兼容的长度
        padded_cells, padded_masks, target_length = pad_batch_to_sparse_compatible(
            cells, mask, block_size
        )
        
        print(f"After padding: target_length={target_length}")
        
        # 输入投影
        x = self.input_proj(padded_cells)  # [batch_size, target_length, d_model]
        
        # 添加位置编码
        if target_length <= self.max_cells:
            pos_enc = self.pos_encoding[:target_length].unsqueeze(0)  # [1, target_length, d_model]
            x = x + pos_enc
        
        # 通过注意力层和前馈网络
        for attn_layer, ffn_layer in zip(self.attention_layers, self.ffn_layers):
            # 稀疏注意力（整个batch统一处理）
            x_attn = attn_layer(x, x, x, padded_masks)
            
            # 前馈网络（带残差连接）
            x = ffn_layer(x_attn) + x_attn
        
        # 应用mask进行池化
        if padded_masks is not None:
            # 将padding部分设为0
            mask_expanded = padded_masks.unsqueeze(-1)  # [batch_size, target_length, 1]
            x = x * mask_expanded
            
            # 计算有效长度的平均值
            valid_lengths = padded_masks.sum(dim=1, keepdim=True)  # [batch_size, 1]
            x_pooled = x.sum(dim=1) / (valid_lengths + 1e-8)  # [batch_size, d_model]
        else:
            # 简单平均池化
            x_pooled = x.mean(dim=1)  # [batch_size, d_model]
        
        print(f"After pooling: x_pooled.shape={x_pooled.shape}")
        
        # 年龄预测
        age_pred = self.age_predictor(x_pooled)
        return age_pred.view(-1)

# ===============================
# 数据集类
# ===============================

class SingleCellDataset(Dataset):
    def __init__(self, cell_data, ages, donor_ids, scaler=None, max_cells=None):
        self.cell_data = cell_data
        self.ages = ages
        self.scaler = scaler
        
        self.valid_donor_ids = []
        for donor_id in donor_ids:
            if donor_id in cell_data and donor_id in ages:
                data = cell_data[donor_id]
                if data is not None and data.shape[0] > 0:
                    self.valid_donor_ids.append(donor_id)

        self.donor_ids = self.valid_donor_ids
        print(f"Dataset initialized with {len(self.donor_ids)} valid donors")

        if max_cells is None:
            cell_counts = [self.cell_data[donor_id].shape[0] for donor_id in self.donor_ids]
            self.max_cells = max(cell_counts) if cell_counts else 1000
        else:
            self.max_cells = max_cells

        if self.donor_ids:
            self.n_genes = self.cell_data[self.donor_ids[0]].shape[1]
        else:
            self.n_genes = 0

        print(f"Max cells: {self.max_cells}, Genes: {self.n_genes}")

    def __len__(self):
        return len(self.donor_ids)

    def __getitem__(self, idx):
        donor_id = self.donor_ids[idx]
        
        try:
            cells = self.cell_data[donor_id].astype(np.float32)
            age = float(self.ages[donor_id])

            if np.isnan(cells).any():
                cells = np.nan_to_num(cells, nan=0.0)
            if not np.isfinite(cells).all():
                cells = np.nan_to_num(cells, nan=0.0, posinf=0.0, neginf=0.0)

            if self.scaler is not None:
                try:
                    cells = self.scaler.transform(cells)
                except Exception as e:
                    print(f"Warning: Scaler transform failed for donor {donor_id}: {e}")

            return {
                'cells': torch.FloatTensor(cells),
                'age': torch.FloatTensor([age]),
                'donor_id': donor_id,
                'n_cells': cells.shape[0]
            }

        except Exception as e:
            print(f"Error loading donor {donor_id}: {e}")
            return {
                'cells': torch.zeros(1, self.n_genes),
                'age': torch.FloatTensor([0.0]),
                'donor_id': donor_id,
                'n_cells': 1
            }

def collate_fn(batch):
    """处理变长输入的collate function"""
    max_cells = max([item['n_cells'] for item in batch])
    batch_size = len(batch)
    n_genes = batch[0]['cells'].shape[1]
    
    padded_cells = torch.zeros(batch_size, max_cells, n_genes)
    masks = torch.zeros(batch_size, max_cells)
    ages = []
    donor_ids = []
    
    for i, item in enumerate(batch):
        cells = item['cells']
        n_cells = item['n_cells']
        
        padded_cells[i, :n_cells] = cells
        masks[i, :n_cells] = 1
        ages.append(item['age'])
        donor_ids.append(item['donor_id'])
    
    return {
        'cells': padded_cells,
        'masks': masks,
        'ages': torch.cat(ages),
        'donor_ids': donor_ids
    }

def prepare_data_loaders(cell_data, ages, train_ids, val_ids, test_ids, batch_size=8):
    """准备数据加载器"""
    
    # 准备标准化器
    print("Preparing scaler...")
    train_cells_list = []
    for donor_id in train_ids[:min(len(train_ids), 50)]:  # 限制样本数量
        if donor_id in cell_data:
            train_cells_list.append(cell_data[donor_id])
    
    if train_cells_list:
        train_cells = np.vstack(train_cells_list)
        scaler = StandardScaler()
        scaler.fit(train_cells)
        print("StandardScaler fitted successfully!")
    else:
        print("Warning: No training data found, using identity scaler")
        class IdentityScaler:
            def fit(self, X): return self
            def transform(self, X): return X.astype(np.float32)
        scaler = IdentityScaler()
    
    # 计算最大细胞数
    all_donor_ids = list(cell_data.keys())
    max_cells = max([cell_data[donor_id].shape[0] for donor_id in all_donor_ids])
    print(f"Maximum cells per donor: {max_cells}")
    
    # 创建数据集
    train_dataset = SingleCellDataset(cell_data, ages, train_ids, scaler, max_cells)
    val_dataset = SingleCellDataset(cell_data, ages, val_ids, scaler, max_cells)
    test_dataset = SingleCellDataset(cell_data, ages, test_ids, scaler, max_cells)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0)
    
    return train_loader, val_loader, test_loader, scaler, max_cells

# ===============================
# 训练和评估函数
# ===============================

def train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-3, 
                device='cuda', patience=15, save_path=None):
    """训练模型 - 使用val_corr作为early stopping指标"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=patience//3, factor=0.5, verbose=True  # mode='max' 因为我们要最大化相关性
    )

    best_val_corr = -1.0  # 初始化为-1，因为相关性范围是[-1, 1]
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_correlations = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_losses = []

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in train_pbar:
            cells, ages, masks = batch['cells'].to(device), batch['ages'].to(device), batch['masks'].to(device)

            optimizer.zero_grad()
            age_pred = model(cells, masks)
            loss = F.mse_loss(age_pred, ages)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_losses.append(loss.item())
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation
        model.eval()
        epoch_val_losses = []
        val_predictions = []
        val_true_ages = []

        with torch.no_grad():
            for batch in val_loader:
                cells, ages, masks = batch['cells'].to(device), batch['ages'].to(device), batch['masks'].to(device)
                age_pred = model(cells, masks)
                val_loss = F.mse_loss(age_pred, ages)

                epoch_val_losses.append(val_loss.item())

                pred_numpy = age_pred.cpu().numpy()
                ages_numpy = ages.cpu().numpy()

                if pred_numpy.ndim == 0:
                    pred_numpy = np.array([pred_numpy])
                if ages_numpy.ndim == 0:
                    ages_numpy = np.array([ages_numpy])

                val_predictions.extend(pred_numpy.tolist())
                val_true_ages.extend(ages_numpy.tolist())

        # 计算验证集相关性
        if len(val_predictions) > 1:
            val_corr, _ = pearsonr(val_predictions, val_true_ages)
            if np.isnan(val_corr):
                val_corr = 0.0
        else:
            val_corr = 0.0

        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_correlations.append(val_corr)

        # 使用相关性作为调度器的指标
        scheduler.step(val_corr)

        # print(f'Epoch {epoch+1}/{num_epochs}:')
        # print(f'  Train Loss: {avg_train_loss:.4f}')
        # print(f'  Val Loss: {avg_val_loss:.4f}')
        # print(f'  Val Correlation: {val_corr:.4f}')
        # print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Early stopping基于val_corr
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f'  Model saved to {save_path} (best val_corr: {best_val_corr:.4f})')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            print(f'Best validation correlation: {best_val_corr:.4f}')
            break

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_correlations': val_correlations,
        'best_val_corr': best_val_corr  # 返回最佳相关性而不是最佳损失
    }

# def evaluate_model(model, test_loader, device='cuda'):
#     """评估模型"""
#     model.eval()
#     predictions = []
#     true_ages = []
#     donor_ids = []

#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc='Testing'):
#             cells, ages, masks = batch['cells'].to(device), batch['ages'].to(device), batch['masks'].to(device)
#             age_pred = model(cells, masks)

#             pred_numpy = age_pred.cpu().numpy()
#             ages_numpy = ages.cpu().numpy()

#             if pred_numpy.ndim == 0:
#                 pred_numpy = np.array([pred_numpy])
#             if ages_numpy.ndim == 0:
#                 ages_numpy = np.array([ages_numpy])

#             predictions.extend(pred_numpy.tolist())
#             true_ages.extend(ages_numpy.tolist())
#             donor_ids.extend(batch['donor_ids'])

#     mse = mean_squared_error(true_ages, predictions)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(true_ages, predictions)
#     corr, p_value = pearsonr(true_ages, predictions)

#     results = {
#         'predictions': predictions,
#         'true_ages': true_ages,
#         'donor_ids': donor_ids,
#         'mse': mse,
#         'rmse': rmse,
#         'r2': r2,
#         'correlation': corr,
#         'p_value': p_value
#     }

#     print(f'Test Results:')
#     print(f'  MSE: {mse:.4f}')
#     print(f'  RMSE: {rmse:.4f}')
#     print(f'  R²: {r2:.4f}')
#     print(f'  Correlation: {corr:.4f} (p={p_value:.2e})')

#     return results

def evaluate_model(model, test_loader, device='cuda'):
    """评估模型 - 包含完整的统计分析"""
    model.eval()
    predictions = []
    true_ages = []
    donor_ids = []

    # 数据收集阶段
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            cells, ages, masks = batch['cells'].to(device), batch['ages'].to(device), batch['masks'].to(device)
            age_pred = model(cells, masks)

            pred_numpy = age_pred.cpu().numpy()
            ages_numpy = ages.cpu().numpy()

            if pred_numpy.ndim == 0:
                pred_numpy = np.array([pred_numpy])
            if ages_numpy.ndim == 0:
                ages_numpy = np.array([ages_numpy])

            predictions.extend(pred_numpy.tolist())
            true_ages.extend(ages_numpy.tolist())
            donor_ids.extend(batch['donor_ids'])

    # 转换为numpy数组以便计算
    predictions = np.array(predictions)
    true_ages = np.array(true_ages)

    # --- 1. 基础误差指标 ---
    mse = mean_squared_error(true_ages, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_ages, predictions)

    # --- 2. 相对误差指标 ---
    # RAE (相对绝对误差)
    numerator = np.sum(np.abs(true_ages - predictions))
    denominator = np.sum(np.abs(true_ages - np.mean(true_ages)))
    rae = numerator / denominator if denominator != 0 else float('inf')

    # --- 3. 相关性指标 ---
    # 皮尔逊相关系数
    pearson_corr, p_value_pearson = pearsonr(true_ages, predictions)
    # 斯皮尔曼相关系数
    spearman_corr, p_value_spearman = spearmanr(true_ages, predictions)
    # R² 决定系数
    r2 = r2_score(true_ages, predictions)

    # --- 4. 对数比率分析（偏差分析）---
    # 避免除零错误
    valid_indices = true_ages > 0
    if np.any(valid_indices):
        ratio = predictions[valid_indices] / true_ages[valid_indices]
        log2_ratio = np.log2(ratio)
        mean_log2_ratio = np.mean(log2_ratio)
        std_log2_ratio = np.std(log2_ratio)
    else:
        mean_log2_ratio = float('nan')
        std_log2_ratio = float('nan')

    # 汇总所有结果
    results = {
        # 原始数据
        'predictions': predictions.tolist(),
        'true_ages': true_ages.tolist(),
        'donor_ids': donor_ids,

        # 误差指标
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'rae': rae,

        # 相关性指标
        'pearson_corr': pearson_corr,
        'p_value_pearson': p_value_pearson,
        'spearman_corr': spearman_corr,
        'p_value_spearman': p_value_spearman,
        'r2': r2,

        # 偏差分析
        'mean_log2_ratio': mean_log2_ratio,
        'std_log2_ratio': std_log2_ratio
    }

    # 格式化输出结果
    print('\n' + '='*60)
    print('模型评估结果')
    print('='*60)

    print('\n--- 误差指标 ---')
    print(f'  MSE (均方误差):        {mse:.4f}')
    print(f'  RMSE (均方根误差):     {rmse:.4f} 岁')
    print(f'  MAE (平均绝对误差):    {mae:.4f} 岁')
    print(f'  RAE (相对绝对误差):    {rae:.4f}')

    print('\n--- 相关性分析 ---')
    print(f'  R² (决定系数):         {r2:.4f}')
    print(f'  皮尔逊相关系数:       {pearson_corr:.4f} (p={p_value_pearson:.2e})')
    print(f'  斯皮尔曼相关系数:     {spearman_corr:.4f} (p={p_value_spearman:.2e})')

    print('\n--- 偏差分析 ---')
    print(f'  Log2(预测/实际) 均值:  {mean_log2_ratio:.4f}')
    print(f'  Log2(预测/实际) 标准差: {std_log2_ratio:.4f}')

    # 解释性说明
    print('\n--- 指标解释 ---')
    if rae < 1:
        print(f'  ✓ RAE < 1: 模型优于均值预测')
    else:
        print(f'  ✗ RAE >= 1: 模型表现不佳')

    if abs(mean_log2_ratio) < 0.1:
        print(f'  ✓ 偏差较小: 模型预测较为准确')
    elif mean_log2_ratio > 0:
        print(f'  ⚠ 系统性高估: 模型倾向于预测更高的年龄')
    else:
        print(f'  ⚠ 系统性低估: 模型倾向于预测更低的年龄')

    print('='*60 + '\n')

    return results

# ===============================
# 跨数据集评估功能
# ===============================

def evaluate_cross_dataset(model, cell_data, ages, donor_dataset_mapping, 
                          train_datasets, test_datasets, scaler, device='cuda'):
    """跨数据集评估模型性能"""
    print(f"\n=== Cross-Dataset Evaluation ===")
    print(f"Training datasets: {train_datasets}")
    print(f"Test datasets: {test_datasets}")
    
    results = {}
    
    for test_dataset in test_datasets:
        print(f"\nEvaluating on {test_dataset}...")
        
        # 获取测试数据集的donors
        test_donors = [donor_id for donor_id, dataset in donor_dataset_mapping.items() 
                      if dataset == test_dataset and donor_id in cell_data and donor_id in ages]
        
        if not test_donors:
            print(f"No donors found for dataset {test_dataset}")
            continue
        
        print(f"Found {len(test_donors)} donors in {test_dataset}")
        
        # 创建测试数据集
        test_dataset_obj = SingleCellDataset(cell_data, ages, test_donors, scaler)
        test_loader = DataLoader(test_dataset_obj, batch_size=8, shuffle=False, 
                               collate_fn=collate_fn, num_workers=0)
        
        # 评估
        dataset_results = evaluate_model(model, test_loader, device)
        dataset_results['dataset'] = test_dataset
        dataset_results['n_donors'] = len(test_donors)
        
        results[test_dataset] = dataset_results
        
        print(f"{test_dataset} Results:")
        print(f"  Correlation: {dataset_results['correlation']:.4f}")
        print(f"  RMSE: {dataset_results['rmse']:.4f}")
        print(f"  R²: {dataset_results['r2']:.4f}")
    
    return results

def plot_cross_dataset_results(cross_dataset_results, celltypes_str, model_type, embedding_type, save_path=None):
    """绘制跨数据集评估结果"""
    if not cross_dataset_results:
        print("No cross-dataset results to plot")
        return
    
    datasets = list(cross_dataset_results.keys())
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(2, n_datasets, figsize=(6*n_datasets, 12))
    if n_datasets == 1:
        axes = axes.reshape(2, 1)
    
    colors = ['#374E55', '#DF8F44', '#B24745', '#6A6599']
    
    for i, dataset in enumerate(datasets):
        results = cross_dataset_results[dataset]
        true_ages = np.array(results['true_ages'])
        predictions = np.array(results['predictions'])
        
        color = colors[i % len(colors)]
        
        # 散点图
        axes[0, i].scatter(true_ages, predictions, alpha=0.6, s=30, color=color)
        axes[0, i].plot([true_ages.min(), true_ages.max()], 
                       [true_ages.min(), true_ages.max()], 
                       'k--', alpha=0.8, linewidth=2)
        
        axes[0, i].set_xlabel('True Age')
        axes[0, i].set_ylabel('Predicted Age')
        axes[0, i].set_title(f'{model_type} - {dataset}\n{celltypes_str} ({embedding_type})')
        
        # 添加统计信息
        text = f"R = {results['correlation']:.3f}\nP = {results['p_value']:.3e}\nRMSE = {results['rmse']:.2f}"
        axes[0, i].text(0.05, 0.95, text, transform=axes[0, i].transAxes, 
                       fontsize=10, verticalalignment="top",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        axes[0, i].grid(True, alpha=0.3)
        
        # 残差图
        residuals = predictions - true_ages
        axes[1, i].scatter(true_ages, residuals, alpha=0.6, s=30, color=color)
        axes[1, i].axhline(y=0, color='red', linestyle='--', alpha=0.8)
        axes[1, i].set_xlabel('True Age')
        axes[1, i].set_ylabel('Residuals')
        axes[1, i].set_title(f'Residuals - {dataset}')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

# ===============================
# 仅评估模式功能
# ===============================

def load_trained_model(model_path, model_type, input_dim, hidden_dim=256, max_cells=1000):
    """加载训练好的模型"""
    print(f"Loading trained model from: {model_path}")
    
    # 创建模型
    if model_type == 'deepsets':
        model = DeepSetsAgePredictor(input_dim=input_dim, hidden_dim=hidden_dim, max_cells=max_cells)
    elif model_type == 'transformer':
        model = SimpleSetTransformerAgePredictor(input_dim=input_dim, d_model=hidden_dim)
    elif model_type == 'sparse_transformer':
        sparse_config = {
            'sparse_type': 'random',
            'block_size': 32,
            'sparsity': 0.1,
            'model_type': 'single_cell'
        }
        model = UnifiedSparseTransformerAgePredictor(
            input_dim=input_dim, d_model=hidden_dim, max_cells=max_cells, sparse_config=sparse_config
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"Model loaded successfully!")
    
    return model

def evaluate_only_mode(args):
    """仅评估模式的主函数"""
    print(f"\n=== Evaluation Only Mode ===")
    print(f"Model path: {args.model_path}")
    print(f"Model type: {args.model_type}")
    print(f"Embedding type: {args.embedding_type}")
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    print(f"Using device: {device}")
    
    # 加载数据
    cell_data, ages, feature_names, donor_dataset_mapping, donor_celltype_counts = load_data_unified(
        args.data_dir, args.celltypes, args.datasets, args.embedding_type, args.cache_dir, args.force_reload
    )
    
    input_dim = len(feature_names)
    
    # 创建数据划分
    train_ids, val_ids, test_ids = create_data_split(
        cell_data, args.datasets, args.celltypes, 
        random_state=args.random_state, cache_dir=args.cache_dir
    )
    
    # 准备数据加载器
    train_loader, val_loader, test_loader, scaler, max_cells = prepare_data_loaders(
        cell_data, ages, train_ids, val_ids, test_ids, batch_size=args.batch_size
    )
    
    # 加载模型
    model = load_trained_model(args.model_path, args.model_type, input_dim, 
                              args.hidden_dim, max_cells)
    model = model.to(device)
    
    # 评估测试集
    print("Evaluating on test set...")
    test_results = evaluate_model(model, test_loader, device)
    
    # 跨数据集评估
    all_datasets = ['AIDA', 'eQTL', 'HCA', 'siAge']
    available_datasets = list(set(donor_dataset_mapping.values()))
    train_datasets = args.datasets
    test_datasets = [d for d in all_datasets if d in available_datasets]
    
    cross_results = evaluate_cross_dataset(
        model, cell_data, ages, donor_dataset_mapping,
        train_datasets, test_datasets, scaler, device
    )
    
    # 生成文件名前缀
    datasets_str = '+'.join(args.datasets)
    celltypes_str = '+'.join(args.celltypes)
    
    # 可视化结果
    plot_cross_dataset_results(
        cross_results, celltypes_str, args.model_type, args.embedding_type,
        save_path=os.path.join(args.output_dir, f'{celltypes_str}_{datasets_str}_{args.model_type}_{args.embedding_type}_cross_dataset_eval.png')
    )
    
    # 保存结果
    eval_results = {
        'test_results': test_results,
        'cross_dataset_results': cross_results,
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids,
        'donor_celltype_counts': donor_celltype_counts,
        'args': vars(args)
    }
    
    results_path = os.path.join(args.output_dir, f'{celltypes_str}_{datasets_str}_{args.model_type}_{args.embedding_type}_evaluation_only.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(eval_results, f)
    
    print(f"Evaluation results saved to: {results_path}")
    
    return eval_results

# ===============================
# 可视化函数
# ===============================

def plot_celltype_statistics(donor_celltype_counts, celltypes, datasets, save_path=None):
    """绘制每个donor的细胞类型统计"""
    
    # 准备数据
    donors = list(donor_celltype_counts.keys())
    celltype_data = []
    
    for donor_id in donors:
        counts = donor_celltype_counts[donor_id]
        total_cells = sum(counts.values())
        
        for celltype, count in counts.items():
            celltype_data.append({
                'donor_id': donor_id,
                'celltype': celltype,
                'count': count,
                'total_cells': total_cells,
                'percentage': count / total_cells * 100
            })
    
    df = pd.DataFrame(celltype_data)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 每个细胞类型的总数量
    celltype_totals = df.groupby('celltype')['count'].sum().sort_values(ascending=False)
    axes[0, 0].bar(range(len(celltype_totals)), celltype_totals.values, color='skyblue')
    axes[0, 0].set_xticks(range(len(celltype_totals)))
    axes[0, 0].set_xticklabels([ct.replace('CD', 'CD') for ct in celltype_totals.index], rotation=45)
    axes[0, 0].set_ylabel('Total Cell Count')
    axes[0, 0].set_title('Total Cell Count by Cell Type')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 每个donor的细胞总数分布
    donor_totals = df.groupby('donor_id')['total_cells'].first()
    axes[0, 1].hist(donor_totals, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('Total Cells per Donor')
    axes[0, 1].set_ylabel('Number of Donors')
    axes[0, 1].set_title('Distribution of Total Cells per Donor')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 细胞类型比例的箱线图
    celltype_percentages = df.pivot(index='donor_id', columns='celltype', values='percentage').fillna(0)
    celltype_percentages.boxplot(ax=axes[1, 0])
    axes[1, 0].set_ylabel('Percentage (%)')
    axes[1, 0].set_title('Cell Type Percentage Distribution')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 堆叠条形图显示每个donor的细胞类型组成
    # 选择前20个donor进行可视化
    top_donors = donor_totals.nlargest(20).index
    subset_df = df[df['donor_id'].isin(top_donors)]
    
    pivot_data = subset_df.pivot(index='donor_id', columns='celltype', values='count').fillna(0)
    pivot_data.plot(kind='bar', stacked=True, ax=axes[1, 1], colormap='Set3')
    axes[1, 1].set_xlabel('Donor ID')
    axes[1, 1].set_ylabel('Cell Count')
    axes[1, 1].set_title('Cell Type Composition (Top 20 Donors)')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print("\n=== Cell Type Statistics ===")
    print(f"Total donors: {len(donors)}")
    print(f"Cell types found: {list(celltype_totals.index)}")
    print("\nCell type totals:")
    for ct, count in celltype_totals.items():
        print(f"  {ct}: {count:,} cells")
    
    print(f"\nDonor cell count statistics:")
    print(f"  Mean: {donor_totals.mean():.1f}")
    print(f"  Median: {donor_totals.median():.1f}")
    print(f"  Range: {donor_totals.min()} - {donor_totals.max()}")
    
    return df

def plot_data_overview(bulk_data, celltypes_str, datasets, save_path=None):
    """绘制数据概览"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 年龄分布
    axes[0, 0].hist(bulk_data['age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'Age Distribution - {celltypes_str}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 数据集分布
    unique_datasets = bulk_data['dataset'].unique()
    if len(unique_datasets) > 1:
        dataset_counts = bulk_data['dataset'].value_counts()
        axes[0, 1].pie(dataset_counts.values, labels=dataset_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Dataset Distribution')
    else:
        axes[0, 1].text(0.5, 0.5, f'Single Dataset: {unique_datasets[0] if len(unique_datasets) > 0 else "Unknown"}', 
                       ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=14)
        axes[0, 1].set_title('Dataset Information')
    
    # 年龄组分布
    age_group_counts = bulk_data['age_group'].value_counts().sort_index()
    axes[1, 0].bar(range(len(age_group_counts)), age_group_counts.values, 
                   color='lightcoral', alpha=0.7)
    axes[1, 0].set_xticks(range(len(age_group_counts)))
    axes[1, 0].set_xticklabels(age_group_counts.index, rotation=45)
    axes[1, 0].set_xlabel('Age Group')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Age Group Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 数据集vs年龄
    if len(unique_datasets) > 1:
        for i, dataset in enumerate(unique_datasets):
            subset = bulk_data[bulk_data['dataset'] == dataset]
            axes[1, 1].scatter(subset['age'], [i] * len(subset), 
                             alpha=0.6, s=30, label=dataset)
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('Dataset')
        axes[1, 1].set_yticks(range(len(unique_datasets)))
        axes[1, 1].set_yticklabels(unique_datasets)
        axes[1, 1].set_title('Age Distribution by Dataset')
        axes[1, 1].legend()
    else:
        axes[1, 1].scatter(bulk_data['age'], np.random.normal(0, 0.1, len(bulk_data)), 
                          alpha=0.6, s=30, color='green')
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('Random Jitter')
        axes[1, 1].set_title(f'Age Distribution - {unique_datasets[0] if len(unique_datasets) > 0 else "Unknown"}')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_dimensionality_reduction(bulk_data, celltypes_str, save_path=None):
    """绘制降维可视化（PCA和UMAP）"""
    # 准备数据
    gene_cols = [col for col in bulk_data.columns if col not in ['age', 'dataset', 'age_group']]
    X = bulk_data[gene_cols].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # UMAP
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # PCA - 按年龄着色
    scatter1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=bulk_data['age'], 
                                 cmap='RdBu_r', s=30, alpha=0.7)
    axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[0, 0].set_title(f'PCA - {celltypes_str} (colored by age)')
    plt.colorbar(scatter1, ax=axes[0, 0], label='Age')
    
    # PCA - 按年龄组着色
    age_groups = bulk_data['age_group'].astype(str)
    unique_groups = age_groups.unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))
    
    for i, group in enumerate(unique_groups):
        mask = age_groups == group
        axes[0, 1].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          c=[colors[i]], label=group, s=30, alpha=0.7)
    axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[0, 1].set_title(f'PCA - {celltypes_str} (colored by age group)')
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # UMAP - 按年龄着色
    scatter2 = axes[1, 0].scatter(X_umap[:, 0], X_umap[:, 1], c=bulk_data['age'], 
                                 cmap='RdBu_r', s=30, alpha=0.7)
    axes[1, 0].set_xlabel('UMAP1')
    axes[1, 0].set_ylabel('UMAP2')
    axes[1, 0].set_title(f'UMAP - {celltypes_str} (colored by age)')
    plt.colorbar(scatter2, ax=axes[1, 0], label='Age')
    
    # UMAP - 按年龄组着色
    for i, group in enumerate(unique_groups):
        mask = age_groups == group
        axes[1, 1].scatter(X_umap[mask, 0], X_umap[mask, 1], 
                          c=[colors[i]], label=group, s=30, alpha=0.7)
    axes[1, 1].set_xlabel('UMAP1')
    axes[1, 1].set_ylabel('UMAP2')
    axes[1, 1].set_title(f'UMAP - {celltypes_str} (colored by age group)')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def compute_feature_age_correlation(bulk_data):
    """计算特征与年龄的相关性"""
    gene_cols = [col for col in bulk_data.columns if col not in ['age', 'dataset', 'age_group']]
    X = bulk_data[gene_cols]
    age = bulk_data['age']
    
    pearson_r = {}
    pearson_p = {}
    spearman_r = {}
    spearman_p = {}

    for feature in X.columns:
        # 皮尔逊相关
        r_pearson, p_pearson = pearsonr(X[feature], age)
        pearson_r[feature] = r_pearson
        pearson_p[feature] = p_pearson

        # 斯皮尔曼相关
        r_spearman, p_spearman = spearmanr(X[feature], age)
        spearman_r[feature] = r_spearman
        spearman_p[feature] = p_spearman

    # FDR 多重校正
    _, pval_pearson_fdr, _, _ = multipletests(list(pearson_p.values()), method='fdr_bh')
    _, pval_spearman_fdr, _, _ = multipletests(list(spearman_p.values()), method='fdr_bh')

    result = pd.DataFrame({
        'pearson_r': pd.Series(pearson_r),
        'pearson_r_abs': np.abs(pd.Series(pearson_r)),
        'pearson_p': pd.Series(pearson_p),
        'pearson_fdr': pval_pearson_fdr,
        'spearman_r': pd.Series(spearman_r),
        'spearman_r_abs': np.abs(pd.Series(spearman_r)),
        'spearman_p': pd.Series(spearman_p),
        'spearman_fdr': pval_spearman_fdr,
    })

    return result

def plot_feature_correlation_analysis(bulk_data, celltypes_str, save_path=None):
    """绘制特征相关性分析"""
    corr_results = compute_feature_age_correlation(bulk_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 皮尔逊相关性分布
    axes[0, 0].hist(corr_results['pearson_r'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_xlabel('Pearson Correlation')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'Distribution of Pearson Correlations - {celltypes_str}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 斯皮尔曼相关性分布
    axes[0, 1].hist(corr_results['spearman_r'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('Spearman Correlation')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Distribution of Spearman Correlations - {celltypes_str}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 相关性散点图
    axes[1, 0].scatter(corr_results['pearson_r'], corr_results['spearman_r'], 
                      alpha=0.6, s=20, color='green')
    axes[1, 0].plot([-1, 1], [-1, 1], 'r--', alpha=0.7)
    axes[1, 0].set_xlabel('Pearson Correlation')
    axes[1, 0].set_ylabel('Spearman Correlation')
    axes[1, 0].set_title('Pearson vs Spearman Correlation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 显著性分析
    significant_pearson = (corr_results['pearson_fdr'] < 0.05).sum()
    significant_spearman = (corr_results['spearman_fdr'] < 0.05).sum()
    total_features = len(corr_results)
    
    categories = ['Pearson\n(FDR<0.05)', 'Spearman\n(FDR<0.05)', 'Total Features']
    values = [significant_pearson, significant_spearman, total_features]
    colors = ['skyblue', 'lightcoral', 'lightgray']
    
    bars = axes[1, 1].bar(categories, values, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('Number of Features')
    axes[1, 1].set_title('Significant Correlations')
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(values),
                       f'{value}', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr_results

def plot_top_correlated_genes(bulk_data, corr_results, celltypes_str, n_genes=6, save_path=None):
    """绘制与年龄相关性最高的基因表达模式"""
    # 选择相关性最高的基因
    top_genes = corr_results.nlargest(n_genes, 'spearman_r_abs').index.tolist()
    
    # 创建年龄分组的平均表达
    age_list = sorted(bulk_data['age'].unique())
    age_samples = {}
    for age in age_list:
        age_samples[age] = bulk_data[bulk_data['age'] == age].index.tolist()
    
    # 绘图
    n_cols = 3
    n_rows = (n_genes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, gene in enumerate(top_genes):
        row = i // n_cols
        col = i % n_cols
        
        if n_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        
        # 散点图
        ax.scatter(bulk_data['age'], bulk_data[gene], alpha=0.5, s=20, color='lightblue')
        
        # 计算每个年龄的平均表达
        avg_expr = []
        for age in age_list:
            samples = age_samples[age]
            avg_data = bulk_data.loc[samples, gene].mean()
            avg_expr.append(avg_data)
        
        # 绘制平均表达趋势线
        ax.plot(age_list, avg_expr, 'r-', linewidth=2, alpha=0.8, label='Mean expression')
        
        # 添加相关性信息
        corr_val = corr_results.loc[gene, 'spearman_r']
        p_val = corr_results.loc[gene, 'spearman_p']
        
        gene_name = gene.split('_')[0] if '_' in gene else gene
        ax.set_title(f'{gene_name}\nSpearman r = {corr_val:.3f}, p = {p_val:.2e}')
        ax.set_xlabel('Age')
        ax.set_ylabel('Mean Expression')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 隐藏多余的子图
    for i in range(n_genes, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1:
            axes[col].set_visible(False)
        else:
            axes[row, col].set_visible(False)
    
    plt.suptitle(f'Top {n_genes} Age-Correlated Genes - {celltypes_str}', fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_data_split_overview(bulk_data, train_ids, val_ids, test_ids, celltypes_str, save_path=None):
    """绘制数据划分概览"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 创建划分标签
    split_labels = []
    for donor_id in bulk_data.index:
        if donor_id in train_ids:
            split_labels.append('Train')
        elif donor_id in val_ids:
            split_labels.append('Validation')
        elif donor_id in test_ids:
            split_labels.append('Test')
        else:
            split_labels.append('Unknown')
    
    bulk_data_with_split = bulk_data.copy()
    bulk_data_with_split['split'] = split_labels
    
    # 1. 划分比例饼图
    split_counts = bulk_data_with_split['split'].value_counts()
    axes[0, 0].pie(split_counts.values, labels=split_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Data Split Distribution')
    
    # 2. 各划分的年龄分布
    for split in ['Train', 'Validation', 'Test']:
        if split in split_counts.index:
            subset = bulk_data_with_split[bulk_data_with_split['split'] == split]
            axes[0, 1].hist(subset['age'], alpha=0.6, label=split, bins=15)
    axes[0, 1].set_xlabel('Age')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Age Distribution by Split')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 年龄vs划分散点图
    colors = {'Train': 'blue', 'Validation': 'orange', 'Test': 'green', 'Unknown': 'red'}
    for split in split_counts.index:
        subset = bulk_data_with_split[bulk_data_with_split['split'] == split]
        y_pos = list(colors.keys()).index(split)
        axes[1, 0].scatter(subset['age'], [y_pos] * len(subset), 
                          alpha=0.6, s=30, color=colors[split], label=split)
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Split')
    axes[1, 0].set_yticks(range(len(colors)))
    axes[1, 0].set_yticklabels(list(colors.keys()))
    axes[1, 0].set_title('Age Distribution by Split (Scatter)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 年龄组在各划分中的分布
    age_split_crosstab = pd.crosstab(bulk_data_with_split['age_group'], bulk_data_with_split['split'])
    age_split_crosstab.plot(kind='bar', ax=axes[1, 1], alpha=0.7)
    axes[1, 1].set_xlabel('Age Group')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Age Group Distribution by Split')
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Data Split Overview - {celltypes_str}', fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_history(history, save_path=None):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # 损失曲线
    axes[0].plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 验证相关性
    axes[1].plot(epochs, history['val_correlations'], 'g-', label='Val Correlation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Correlation')
    axes[1].set_title('Validation Correlation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 对数尺度损失
    axes[2].semilogy(epochs, history['train_losses'], 'b-', label='Train Loss (log)', linewidth=2)
    axes[2].semilogy(epochs, history['val_losses'], 'r-', label='Val Loss (log)', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss (log scale)')
    axes[2].set_title('Loss (Log Scale)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# def plot_prediction_results_comprehensive(results, bulk_data, celltypes_str, model_type, 
#                                         train_ids, val_ids, test_ids, save_path=None):
#     """绘制全面的预测结果分析"""
#     true_ages = np.array(results['true_ages'])
#     predictions = np.array(results['predictions'])
#     test_donors = results['donor_ids']
#     corr = results['correlation']
#     p_value = results['p_value']
    
#     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
#     # 1. 基本预测散点图
#     axes[0, 0].scatter(true_ages, predictions, alpha=0.6, s=30, color='#1F77B4')
    
#     # 添加回归线
#     z = np.polyfit(true_ages, predictions, 1)
#     p = np.poly1d(z)
#     axes[0, 0].plot(true_ages, p(true_ages), "r--", alpha=0.8, linewidth=2)
    
#     # 完美预测线
#     axes[0, 0].plot([true_ages.min(), true_ages.max()], 
#                    [true_ages.min(), true_ages.max()], 
#                    "k--", alpha=0.6, linewidth=2, label="Perfect prediction")
    
#     axes[0, 0].set_xlabel("True Age")
#     axes[0, 0].set_ylabel("Predicted Age")
#     axes[0, 0].set_title(f"{model_type} - {celltypes_str}")
    
#     # 添加统计信息
#     text = f"R = {corr:.3f}\nP = {p_value:.3e}\nRMSE = {results['rmse']:.2f}"
#     axes[0, 0].text(0.05, 0.95, text, transform=axes[0, 0].transAxes, 
#                    fontsize=12, verticalalignment="top",
#                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
#     axes[0, 0].legend()
#     axes[0, 0].grid(True, alpha=0.3)
    
#     # 2. 残差图
#     residuals = predictions - true_ages
#     axes[0, 1].scatter(true_ages, residuals, alpha=0.6, s=30, color='orange')
#     axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.8)
#     axes[0, 1].set_xlabel("True Age")
#     axes[0, 1].set_ylabel("Residuals (Predicted - True)")
#     axes[0, 1].set_title("Residual Plot")
#     axes[0, 1].grid(True, alpha=0.3)
    
#     # 3. 残差分布
#     axes[0, 2].hist(residuals, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
#     axes[0, 2].axvline(0, color='red', linestyle='--', alpha=0.8)
#     axes[0, 2].set_xlabel("Residuals")
#     axes[0, 2].set_ylabel("Frequency")
#     axes[0, 2].set_title(f"Residual Distribution\nMean: {np.mean(residuals):.2f}, Std: {np.std(residuals):.2f}")
#     axes[0, 2].grid(True, alpha=0.3)
    
#     # 4. 年龄组分析
#     test_bulk_data = bulk_data.loc[test_donors]
#     age_groups = test_bulk_data['age_group'].astype(str)
#     unique_groups = sorted(age_groups.unique())
    
#     group_true = []
#     group_pred = []
#     group_labels = []
    
#     for group in unique_groups:
#         mask = age_groups == group
#         if mask.sum() > 0:
#             group_true.extend(true_ages[mask])
#             group_pred.extend(predictions[mask])
#             group_labels.extend([group] * mask.sum())
    
#     # 按年龄组着色的散点图
#     colors = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))
#     for i, group in enumerate(unique_groups):
#         mask = np.array(group_labels) == group
#         if mask.sum() > 0:
#             axes[1, 0].scatter(np.array(group_true)[mask], np.array(group_pred)[mask], 
#                              c=[colors[i]], label=group, s=30, alpha=0.7)
    
#     axes[1, 0].plot([min(group_true), max(group_true)], 
#                    [min(group_true), max(group_true)], 
#                    "k--", alpha=0.6, linewidth=2)
#     axes[1, 0].set_xlabel("True Age")
#     axes[1, 0].set_ylabel("Predicted Age")
#     axes[1, 0].set_title("Predictions by Age Group")
#     axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     axes[1, 0].grid(True, alpha=0.3)
    
#     # 5. 年龄组平均预测
#     group_stats = []
#     for group in unique_groups:
#         mask = np.array(group_labels) == group
#         if mask.sum() >= 3:  # 至少3个样本
#             group_stats.append({
#                 'group': group,
#                 'true_mean': np.mean(np.array(group_true)[mask]),
#                 'pred_mean': np.mean(np.array(group_pred)[mask]),
#                 'count': mask.sum()
#             })
    
#     if group_stats:
#         group_df = pd.DataFrame(group_stats)
#         axes[1, 1].scatter(group_df['true_mean'], group_df['pred_mean'], 
#                           s=group_df['count']*3, alpha=0.7, color='purple')
        
#         # 添加标签
#         for _, row in group_df.iterrows():
#             axes[1, 1].annotate(f"{row['group']}\n(n={row['count']})", 
#                                (row['true_mean'], row['pred_mean']),
#                                xytext=(5, 5), textcoords='offset points', fontsize=8)
        
#         # 计算年龄组相关性
#         if len(group_df) > 2:
#             group_corr, group_p = pearsonr(group_df['true_mean'], group_df['pred_mean'])
#             axes[1, 1].set_title(f'Age Group Means\nR = {group_corr:.3f}, P = {group_p:.3e}')
#         else:
#             axes[1, 1].set_title('Age Group Means')
        
#         axes[1, 1].plot([group_df['true_mean'].min(), group_df['true_mean'].max()], 
#                        [group_df['true_mean'].min(), group_df['true_mean'].max()], 
#                        "k--", alpha=0.6, linewidth=2)
#         axes[1, 1].set_xlabel("True Age (Group Mean)")
#         axes[1, 1].set_ylabel("Predicted Age (Group Mean)")
#         axes[1, 1].grid(True, alpha=0.3)
    
#     # 6. 数据集分析（如果有多个数据集）
#     if 'dataset' in test_bulk_data.columns and len(test_bulk_data['dataset'].unique()) > 1:
#         datasets = test_bulk_data['dataset'].unique()
#         dataset_colors = plt.cm.Set1(np.linspace(0, 1, len(datasets)))
        
#         for i, dataset in enumerate(datasets):
#             mask = test_bulk_data['dataset'] == dataset
#             dataset_true = true_ages[mask]
#             dataset_pred = predictions[mask]
            
#             axes[1, 2].scatter(dataset_true, dataset_pred, 
#                              c=[dataset_colors[i]], label=dataset, s=30, alpha=0.7)
        
#         axes[1, 2].plot([true_ages.min(), true_ages.max()], 
#                        [true_ages.min(), true_ages.max()], 
#                        "k--", alpha=0.6, linewidth=2)
#         axes[1, 2].set_xlabel("True Age")
#         axes[1, 2].set_ylabel("Predicted Age")
#         axes[1, 2].set_title("Predictions by Dataset")
#         axes[1, 2].legend()
#         axes[1, 2].grid(True, alpha=0.3)
#     else:
#         # 如果只有一个数据集，显示预测误差的分布
#         abs_errors = np.abs(residuals)
#         axes[1, 2].scatter(true_ages, abs_errors, alpha=0.6, s=30, color='red')
#         axes[1, 2].set_xlabel("True Age")
#         axes[1, 2].set_ylabel("Absolute Error")
#         axes[1, 2].set_title(f"Absolute Prediction Error\nMean: {np.mean(abs_errors):.2f}")
#         axes[1, 2].grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()

def plot_prediction_results_comprehensive(results, bulk_data, celltypes_str, model_type, 
                                        train_ids, val_ids, test_ids, save_path=None):
    """绘制全面的预测结果分析 - 增强版"""

    # 提取基础数据
    true_ages = np.array(results['true_ages'])
    predictions = np.array(results['predictions'])
    test_donors = results['donor_ids']

    # 计算所有统计指标
    # 相关性指标
    pearson_corr = results.get('pearson_corr', results.get('correlation'))
    p_value_pearson = results.get('p_value_pearson', results.get('p_value'))
    spearman_corr = results.get('spearman_corr', spearmanr(true_ages, predictions)[0])
    p_value_spearman = results.get('p_value_spearman', spearmanr(true_ages, predictions)[1])

    # 误差指标
    rmse = results['rmse']
    mae = results.get('mae', np.mean(np.abs(predictions - true_ages)))
    r2 = results.get('r2')
    rae = results.get('rae', np.sum(np.abs(true_ages - predictions)) / 
                      np.sum(np.abs(true_ages - np.mean(true_ages))))

    # 对数比率分析
    valid_idx = true_ages > 0
    if np.any(valid_idx):
        log2_ratio = np.log2(predictions[valid_idx] / true_ages[valid_idx])
        mean_log2_ratio = results.get('mean_log2_ratio', np.mean(log2_ratio))
        std_log2_ratio = results.get('std_log2_ratio', np.std(log2_ratio))
    else:
        log2_ratio = np.array([])
        mean_log2_ratio = 0
        std_log2_ratio = 0

    # 创建图表布局
    fig = plt.figure(figsize=(30, 21))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.35)

    # ==================== 第一行 ====================

    # 1. 主预测散点图（增强统计信息）
    ax1 = fig.add_subplot(gs[0, 0])

    # 绘制散点图，使用渐变色表示密度
    scatter = ax1.scatter(true_ages, predictions, alpha=0.6, s=30, 
                         c=np.abs(predictions - true_ages), cmap='coolwarm',
                         vmin=0, vmax=np.percentile(np.abs(predictions - true_ages), 95))
    plt.colorbar(scatter, ax=ax1, label='|Error|')

    # 添加回归线
    z = np.polyfit(true_ages, predictions, 1)
    p = np.poly1d(z)
    ax1.plot(true_ages, p(true_ages), "b--", alpha=0.8, linewidth=2, 
             label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

    # 完美预测线
    ax1.plot([true_ages.min(), true_ages.max()], 
            [true_ages.min(), true_ages.max()], 
            "k--", alpha=0.6, linewidth=2, label="Perfect prediction")

    # 添加置信区间
    ax1.fill_between([true_ages.min(), true_ages.max()],
                     [true_ages.min() - mae, true_ages.max() - mae],
                     [true_ages.min() + mae, true_ages.max() + mae],
                     alpha=0.2, color='gray', label=f'±MAE ({mae:.2f})')

    ax1.set_xlabel("True Age (years)", fontsize=11)
    ax1.set_ylabel("Predicted Age (years)", fontsize=11)
    ax1.set_title(f"{model_type} - {celltypes_str}", fontsize=12, fontweight='bold')

    # 增强的统计信息框
    stats_text = (f"Pearson R = {pearson_corr:.3f} (p={p_value_pearson:.2e})\n"
                 f"Spearman ρ = {spearman_corr:.3f} (p={p_value_spearman:.2e})\n"
                 f"R² = {r2:.3f}\n"
                 f"RMSE = {rmse:.2f} years\n"
                 f"MAE = {mae:.2f} years\n"
                 f"RAE = {rae:.3f}")

    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                     edgecolor='gray', alpha=0.9))

    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. 残差图（增强版）
    ax2 = fig.add_subplot(gs[0, 1])
    residuals = predictions - true_ages

    # 使用颜色编码年龄范围
    scatter2 = ax2.scatter(true_ages, residuals, alpha=0.6, s=30, 
                          c=true_ages, cmap='viridis')
    plt.colorbar(scatter2, ax=ax2, label='True Age')

    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax2.axhline(y=np.mean(residuals), color='blue', linestyle=':', 
               alpha=0.8, label=f'Mean={np.mean(residuals):.2f}')

    # 添加标准差线
    ax2.axhline(y=np.std(residuals), color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(y=-np.std(residuals), color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel("True Age (years)", fontsize=11)
    ax2.set_ylabel("Residuals (Predicted - True)", fontsize=11)
    ax2.set_title("Residual Analysis", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. 残差分布直方图 + 正态性检验
    ax3 = fig.add_subplot(gs[0, 2])

    # 绘制直方图
    n, bins, patches = ax3.hist(residuals, bins=25, alpha=0.7, 
                                color='lightgreen', edgecolor='black', density=True)

    # 添加核密度估计
    from scipy import stats
    kde = stats.gaussian_kde(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 100)
    ax3.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    # 添加正态分布参考线
    normal_dist = stats.norm(loc=np.mean(residuals), scale=np.std(residuals))
    ax3.plot(x_range, normal_dist.pdf(x_range), 'b--', linewidth=2, 
            label='Normal', alpha=0.7)

    ax3.axvline(0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax3.axvline(np.mean(residuals), color='green', linestyle='-', 
               alpha=0.8, linewidth=2, label=f'Mean={np.mean(residuals):.2f}')

    # Shapiro-Wilk正态性检验
    if len(residuals) <= 5000:  # Shapiro-Wilk适用于小样本
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        normality_text = f"Shapiro-Wilk p={shapiro_p:.3f}"
    else:
        normality_text = ""

    ax3.set_xlabel("Residuals (years)", fontsize=11)
    ax3.set_ylabel("Density", fontsize=11)
    ax3.set_title(f"Residual Distribution\n{normality_text}", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ==================== 第二行 ====================

    # 4. 年龄组分析（保持原有逻辑，增强显示）
    ax4 = fig.add_subplot(gs[1, 0])
    test_bulk_data = bulk_data.loc[test_donors]
    age_groups = test_bulk_data['age_group'].astype(str)
    unique_groups = sorted(age_groups.unique())

    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))
    group_errors = []

    for i, group in enumerate(unique_groups):
        mask = age_groups == group
        if mask.sum() > 0:
            group_true = true_ages[mask]
            group_pred = predictions[mask]
            ax4.scatter(group_true, group_pred, c=[colors[i]], 
                       label=f'{group} (n={mask.sum()})', s=40, alpha=0.7)

            # 计算每组的MAE
            group_mae = np.mean(np.abs(group_pred - group_true))
            group_errors.append({'group': group, 'mae': group_mae, 'n': mask.sum()})

    ax4.plot([true_ages.min(), true_ages.max()], 
            [true_ages.min(), true_ages.max()], 
            "k--", alpha=0.6, linewidth=2)

    ax4.set_xlabel("True Age (years)", fontsize=11)
    ax4.set_ylabel("Predicted Age (years)", fontsize=11)
    ax4.set_title("Predictions by Age Group", fontsize=12, fontweight='bold')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. Log2比率分析
    ax5 = fig.add_subplot(gs[1, 1])

    if len(log2_ratio) > 0:
        # 绘制直方图
        n, bins, patches = ax5.hist(log2_ratio, bins=25, alpha=0.7, 
                                    color='skyblue', edgecolor='black')

        # 根据值着色
        for i, patch in enumerate(patches):
            if bins[i] < -0.1:
                patch.set_facecolor('blue')  # 低估
            elif bins[i] > 0.1:
                patch.set_facecolor('red')    # 高估
            else:
                patch.set_facecolor('green')  # 准确

        ax5.axvline(x=0, color='black', linestyle='--', linewidth=2, 
                   label='Perfect (ratio=1)')
        ax5.axvline(x=mean_log2_ratio, color='red', linestyle='-', linewidth=2,
                   label=f'Mean={mean_log2_ratio:.3f}')

        # 添加标准差范围
        ax5.axvspan(mean_log2_ratio - std_log2_ratio, 
                   mean_log2_ratio + std_log2_ratio,
                   alpha=0.2, color='gray', label=f'±1 SD ({std_log2_ratio:.3f})')

        # 添加偏差解释
        if abs(mean_log2_ratio) < 0.1:
            bias_text = "Low bias ✓"
            bias_color = 'green'
        elif mean_log2_ratio > 0:
            bias_text = f"Overestim. ↑ ({2**mean_log2_ratio:.2f}x)"
            bias_color = 'red'
        else:
            bias_text = f"Underestim. ↓ ({2**mean_log2_ratio:.2f}x)"
            bias_color = 'blue'

        ax5.text(0.95, 0.95, bias_text, transform=ax5.transAxes,
                fontsize=11, color=bias_color, fontweight='bold',
                ha='right', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                         edgecolor=bias_color, alpha=0.9))

        ax5.set_xlabel("Log₂(Predicted/True)", fontsize=11)
        ax5.set_ylabel("Frequency", fontsize=11)
        ax5.set_title("Prediction Bias Analysis", fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)

    # 6. 绝对误差热图（按年龄分组）
    ax6 = fig.add_subplot(gs[1, 2])

    # 创建年龄区间
    age_bins = np.linspace(true_ages.min(), true_ages.max(), 11)
    age_labels = [f'{int(age_bins[i])}-{int(age_bins[i+1])}' 
                  for i in range(len(age_bins)-1)]

    # 计算每个区间的误差统计
    abs_errors = np.abs(predictions - true_ages)
    error_matrix = []

    for i in range(len(age_bins)-1):
        mask = (true_ages >= age_bins[i]) & (true_ages < age_bins[i+1])
        if mask.sum() > 0:
            errors_in_bin = abs_errors[mask]
            error_matrix.append([
                np.mean(errors_in_bin),
                np.median(errors_in_bin),
                np.std(errors_in_bin),
                mask.sum()
            ])
        else:
            error_matrix.append([np.nan, np.nan, np.nan, 0])

    error_df = pd.DataFrame(error_matrix, 
                            columns=['Mean', 'Median', 'Std', 'Count'],
                            index=age_labels)

    # 绘制热图
    sns.heatmap(error_df[['Mean', 'Median', 'Std']].T, 
                annot=True, fmt='.2f', cmap='YlOrRd',
                cbar_kws={'label': 'Error (years)'}, ax=ax6)

    ax6.set_xlabel("Age Range", fontsize=11)
    ax6.set_ylabel("Error Metric", fontsize=11)
    ax6.set_title("Absolute Error by Age Range", fontsize=12, fontweight='bold')

    # ==================== 第三行 ====================

    # 7. 年龄组平均值分析
    ax7 = fig.add_subplot(gs[2, 0])

    group_stats = []
    for group in unique_groups:
        mask = age_groups == group
        if mask.sum() >= 3:
            group_stats.append({
                'group': group,
                'true_mean': np.mean(true_ages[mask]),
                'pred_mean': np.mean(predictions[mask]),
                'true_std': np.std(true_ages[mask]),
                'pred_std': np.std(predictions[mask]),
                'count': mask.sum()
            })

    if group_stats:
        group_df = pd.DataFrame(group_stats)

        # 绘制带误差棒的散点图
        ax7.errorbar(group_df['true_mean'], group_df['pred_mean'],
                    xerr=group_df['true_std'], yerr=group_df['pred_std'],
                    fmt='o', alpha=0.6, capsize=5)

        # 点的大小表示样本数
        scatter = ax7.scatter(group_df['true_mean'], group_df['pred_mean'],
                            s=group_df['count']*5, alpha=0.7, 
                            c=range(len(group_df)), cmap='viridis')

        # 添加标签
        for _, row in group_df.iterrows():
            ax7.annotate(f"{row['group']}\n(n={row['count']})",
                        (row['true_mean'], row['pred_mean']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        # 计算组间相关性
        if len(group_df) > 2:
            group_corr, group_p = pearsonr(group_df['true_mean'], group_df['pred_mean'])
            ax7.set_title(f'Age Group Statistics\nR = {group_corr:.3f}, P = {group_p:.3e}',
                         fontsize=12, fontweight='bold')

        ax7.plot([group_df['true_mean'].min(), group_df['true_mean'].max()],
                [group_df['true_mean'].min(), group_df['true_mean'].max()],
                "k--", alpha=0.6, linewidth=2)

        ax7.set_xlabel("True Age (Group Mean ± SD)", fontsize=11)
        ax7.set_ylabel("Predicted Age (Group Mean ± SD)", fontsize=11)
        ax7.grid(True, alpha=0.3)

    # 8. Bland-Altman图
    ax8 = fig.add_subplot(gs[2, 1])

    mean_ages = (true_ages + predictions) / 2
    diff_ages = predictions - true_ages

    ax8.scatter(mean_ages, diff_ages, alpha=0.6, s=30)

    # 平均差异线
    mean_diff = np.mean(diff_ages)
    ax8.axhline(mean_diff, color='red', linestyle='-', linewidth=2,
               label=f'Mean diff = {mean_diff:.2f}')

    # 一致性界限 (±1.96 SD)
    sd_diff = np.std(diff_ages)
    ax8.axhline(mean_diff + 1.96*sd_diff, color='red', linestyle='--',
               label=f'±1.96 SD = [{mean_diff - 1.96*sd_diff:.2f}, {mean_diff + 1.96*sd_diff:.2f}]')
    ax8.axhline(mean_diff - 1.96*sd_diff, color='red', linestyle='--')

    # 零线
    ax8.axhline(0, color='black', linestyle=':', alpha=0.5)

    ax8.set_xlabel("Mean of True and Predicted Age", fontsize=11)
    ax8.set_ylabel("Difference (Predicted - True)", fontsize=11)
    ax8.set_title("Bland-Altman Plot", fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

    # 9. 性能指标汇总
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    # 创建性能指标表格
    metrics_data = [
        ['Metric', 'Val', 'Mark'],
        ['', '', ''],
        # ['相关性分析', '', ''],
        ['Pearson R', f'{pearson_corr:.4f}', '✓' if pearson_corr > 0.7 else '⚠'],
        ['Spearman ρ', f'{spearman_corr:.4f}', '✓' if spearman_corr > 0.7 else '⚠'],
        ['R²', f'{r2:.4f}', '✓' if r2 > 0.5 else '⚠'],
        ['', '', ''],
        # ['误差分析', '', ''],
        ['RMSE', f'{rmse:.2f}', '✓' if rmse < 10 else '⚠'],
        ['MAE', f'{mae:.2f}', '✓' if mae < 8 else '⚠'],
        ['RAE', f'{rae:.4f}', '✓' if rae < 1 else '✗'],
        ['', '', ''],
        # ['偏差分析', '', ''],
        ['Log₂ Mean', f'{mean_log2_ratio:.4f}', '✓' if abs(mean_log2_ratio) < 0.1 else '⚠'],
        ['Log₂ SD', f'{std_log2_ratio:.4f}', '✓' if std_log2_ratio < 0.5 else '⚠'],
        ['', '', ''],
        # ['数据集信息', '', ''],
        ['Train', f'{len(train_ids)} samples', ''],
        ['Valid', f'{len(val_ids)} samples', ''],
        ['Test', f'{len(test_ids)} samples', ''],
    ]

    # 创建表格
    table = ax9.table(cellText=metrics_data,
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.4, 0.35, 0.25])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # 设置表格样式
    for i in range(len(metrics_data)):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # 标题行
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            # elif metrics_data[i][0] in ['相关性分析', '误差分析', '偏差分析', '数据集信息']:
            #     cell.set_facecolor('#E0E0E0')
            #     cell.set_text_props(weight='bold')
            elif j == 2:  # 评价列
                if metrics_data[i][2] == '✓':
                    cell.set_text_props(color='green', weight='bold')
                elif metrics_data[i][2] == '⚠':
                    cell.set_text_props(color='orange', weight='bold')
                elif metrics_data[i][2] == '✗':
                    cell.set_text_props(color='red', weight='bold')

    ax9.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)

    # 添加总标题
    fig.suptitle(f'Comprehensive Model Evaluation: {model_type} - {celltypes_str}',
                fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'图表已保存至: {save_path}')

    plt.show()

    # # 返回详细的分组分析结果
    # return {
    #     'group_errors': group_errors if 'group_errors' in locals() else None,
    #     'error_by_age': error_df if 'error_df' in locals() else None,
    #     'group_statistics': group_df if 'group_df' in locals() else None
    # }

def create_data_analysis_report(bulk_data, corr_results, celltypes_str, datasets, 
                               train_ids, val_ids, test_ids, args, save_path=None):
    """创建数据分析报告（不包含模型结果）"""
    report = []
    report.append(f"# Data Analysis Report - {celltypes_str}")
    report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 数据集信息
    report.append("## Dataset Information")
    report.append(f"- Cell Types: {celltypes_str}")
    report.append(f"- Datasets: {', '.join(datasets)}")
    report.append(f"- Total Donors: {len(bulk_data)}")
    report.append(f"- Age Range: {bulk_data['age'].min():.1f} - {bulk_data['age'].max():.1f}")
    report.append(f"- Mean Age: {bulk_data['age'].mean():.1f} ± {bulk_data['age'].std():.1f}")
    report.append("")
    
    # 数据划分信息
    report.append("## Data Split Information")
    report.append(f"- Training Set: {len(train_ids)} donors")
    report.append(f"- Validation Set: {len(val_ids)} donors")
    report.append(f"- Test Set: {len(test_ids)} donors")
    report.append(f"- Random State: {args.random_state}")
    report.append("")
    
    # 特征分析
    report.append("## Feature Analysis")
    significant_pearson = (corr_results['pearson_fdr'] < 0.05).sum()
    significant_spearman = (corr_results['spearman_fdr'] < 0.05).sum()
    report.append(f"- Total Features: {len(corr_results)}")
    report.append(f"- Significant Pearson Correlations (FDR < 0.05): {significant_pearson}")
    report.append(f"- Significant Spearman Correlations (FDR < 0.05): {significant_spearman}")
    report.append("")
    
    # 顶级相关基因
    report.append("### Top 10 Age-Correlated Genes (by Spearman)")
    top_genes = corr_results.nlargest(10, 'spearman_r_abs')
    for i, (gene, row) in enumerate(top_genes.iterrows(), 1):
        gene_name = gene.split('_')[0] if '_' in gene else gene
        report.append(f"{i}. {gene_name}: r = {row['spearman_r']:.3f}, p = {row['spearman_p']:.2e}")
    report.append("")
    
    # 年龄组分析
    if 'age_group' in bulk_data.columns:
        report.append("## Age Group Distribution")
        age_group_counts = bulk_data['age_group'].value_counts().sort_index()
        for group, count in age_group_counts.items():
            report.append(f"- {group}: {count} donors")
        report.append("")
    
    # 数据集分析
    if 'dataset' in bulk_data.columns:
        report.append("## Dataset Distribution")
        dataset_counts = bulk_data['dataset'].value_counts()
        for dataset, count in dataset_counts.items():
            report.append(f"- {dataset}: {count} donors")
        report.append("")
    
    # 保存报告
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        print(f"Data analysis report saved to: {save_path}")
    
    return '\n'.join(report)

def create_comprehensive_report(results, history, bulk_data, 
                               celltypes_str, model_type, datasets, train_ids, val_ids, test_ids, args, save_path=None):
    """创建综合报告（包含模型结果）"""
    report = []
    report.append(f"# {model_type.upper()} Model Training Report - {celltypes_str}")
    report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 数据集信息
    report.append("## Dataset Information")
    report.append(f"- Cell Types: {celltypes_str}")
    report.append(f"- Datasets: {', '.join(datasets)}")
    report.append(f"- Total Donors: {len(bulk_data)}")
    report.append(f"- Age Range: {bulk_data['age'].min():.1f} - {bulk_data['age'].max():.1f}")
    report.append(f"- Mean Age: {bulk_data['age'].mean():.1f} ± {bulk_data['age'].std():.1f}")
    report.append("")
    
    # 数据划分信息
    report.append("## Data Split Information")
    report.append(f"- Training Set: {len(train_ids)} donors")
    report.append(f"- Validation Set: {len(val_ids)} donors")
    report.append(f"- Test Set: {len(test_ids)} donors")
    report.append(f"- Random State: {args.random_state}")
    report.append("")
    
    # 模型配置
    report.append("## Model Configuration")
    report.append(f"- Model Type: {model_type}")
    report.append(f"- Hidden Dimension: {args.hidden_dim}")
    report.append(f"- Batch Size: {args.batch_size}")
    report.append(f"- Learning Rate: {args.lr}")
    report.append(f"- Max Epochs: {args.epochs}")
    report.append("")
    
    # 训练结果
    report.append("## Training Results")
    report.append(f"- Final Training Loss: {history['train_losses'][-1]:.4f}")
    report.append(f"- Final Validation Loss: {history['val_losses'][-1]:.4f}")
    report.append(f"- Best Validation Correlation: {history['best_val_corr']:.4f}")
    report.append(f"- Final Validation Correlation: {history['val_correlations'][-1]:.4f}")
    report.append(f"- Training Epochs: {len(history['train_losses'])}")
    report.append("")
    
    # 测试结果
    report.append("## Test Results")
    report.append(f"- Test Correlation: {results['correlation']:.4f} (p = {results['p_value']:.2e})")
    report.append(f"- Test RMSE: {results['rmse']:.4f}")
    report.append(f"- Test R²: {results['r2']:.4f}")
    report.append(f"- Test MSE: {results['mse']:.4f}")
    report.append("")
    
    # 保存报告
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        print(f"Comprehensive report saved to: {save_path}")
    
    return '\n'.join(report)

def analyze_cell_distribution(cell_data):
    """详细分析细胞数量分布，包括更小的block_size"""
    cell_counts = [data.shape[0] for data in cell_data.values()]
    
    print("=== Detailed Cell Count Distribution Analysis ===")
    print(f"Total donors: {len(cell_counts)}")
    print(f"Min cells: {min(cell_counts)}")
    print(f"Max cells: {max(cell_counts)}")
    print(f"Mean cells: {np.mean(cell_counts):.1f}")
    print(f"Median cells: {np.median(cell_counts):.1f}")
    print(f"Std cells: {np.std(cell_counts):.1f}")
    
    # 分析更小的block_size
    block_sizes = [4, 8, 16, 32, 64, 128, 256]
    print("\n=== Extended Block Size Coverage Analysis ===")
    print("Block Size | Divisible | Sparse Eligible | Avg Blocks | Memory Factor")
    print("-" * 70)
    
    for block_size in block_sizes:
        divisible_count = sum(1 for count in cell_counts if count % block_size == 0)
        coverage = divisible_count / len(cell_counts) * 100
        
        # 计算可以使用稀疏注意力的比例
        sparse_eligible = sum(1 for count in cell_counts if count >= block_size and count % block_size == 0)
        sparse_coverage = sparse_eligible / len(cell_counts) * 100
        
        # 计算平均块数量
        avg_blocks = np.mean([count // block_size for count in cell_counts if count >= block_size])
        
        # 内存因子：block_size越小，块数越多，但每块内存占用越小
        memory_factor = block_size * block_size  # 每个块的attention矩阵大小
        
        print(f"{block_size:9d} | {coverage:8.1f}% | {sparse_coverage:14.1f}% | {avg_blocks:9.1f} | {memory_factor:12d}")
    
    return cell_counts

# ===============================
# 主函数（智能skip_training版本）
# ===============================
def main():
    parser = argparse.ArgumentParser(description='Enhanced Single Cell Age Prediction Neural Networks')
    parser.add_argument('--data_dir', type=str, default='/personal/ImmAge/processed_h5ad',
                       help='Directory containing h5ad files')
    parser.add_argument('--celltypes', nargs='+', default=['CD4T'], 
                       choices=['CD4T', 'CD8T', 'macrophage', 'monocyte', 'NK'],
                       help='Cell types to train on (can specify multiple)')
    parser.add_argument('--datasets', nargs='+', default=['AIDA', 'eQTL'],
                       choices=['AIDA', 'eQTL', 'HCA', 'siAge'],
                       help='Datasets to use for training')
    parser.add_argument('--model_type', type=str, default='deepsets', 
                       choices=['deepsets', 'transformer', 'sparse_transformer'],
                       help='Type of neural network model')
    parser.add_argument('--embedding_type', type=str, default='gene_expression',
                       choices=['gene_expression', 'scgpt', 'scimilarity'],
                       help='Type of input features')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--output_dir', type=str, default='./neural_network_results_enhanced',
                       help='Output directory for results')
    parser.add_argument('--cache_dir', type=str, default='./data_cache',
                       help='Directory for data cache')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--random_state', type=int, default=114514, help='Random state for reproducibility')
    
    # 新增选项
    parser.add_argument('--skip_training', action='store_true', help='Skip training and only do data analysis')
    parser.add_argument('--evaluate_only', action='store_true', help='Only evaluate a trained model')
    parser.add_argument('--model_path', type=str, help='Path to trained model for evaluation')
    parser.add_argument('--validate_reference', action='store_true', help='Validate data split with reference LASSO')
    parser.add_argument('--cross_dataset_eval', action='store_true', help='Perform cross-dataset evaluation')
    parser.add_argument('--force_reload', action='store_true', help='Force reload data')

    args = parser.parse_args()
    
    # 验证参数
    if args.evaluate_only and not args.model_path:
        parser.error("--evaluate_only requires --model_path")
  
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Cell types: {args.celltypes}")
    print(f"Datasets: {args.datasets}")
    print(f"Embedding type: {args.embedding_type}")

    if args.force_reload:
        print("Force reload mode: Will ignore existing data cache")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # 仅评估模式
    if args.evaluate_only:
        return evaluate_only_mode(args)
    
    # 加载数据（统一接口，支持缓存）
    print("Loading data...")
    cell_data, ages, feature_names, donor_dataset_mapping, donor_celltype_counts = load_data_unified(
        args.data_dir, args.celltypes, args.datasets, args.embedding_type, args.cache_dir, args.force_reload
    )
    
    # 创建数据划分
    train_ids, val_ids, test_ids = create_data_split(
        cell_data, args.datasets, args.celltypes, 
        random_state=args.random_state, cache_dir=args.cache_dir
    )
    
    # 数据统计
    n_donors = len(cell_data)
    n_genes = len(feature_names)
    cell_counts = analyze_cell_distribution(cell_data)
    
    print(f"Data loaded: {n_donors} donors, {n_genes} genes")
    print(f"Cells per donor: {np.mean(cell_counts):.1f} ± {np.std(cell_counts):.1f}")
    print(f"Age range: {min(ages.values()):.1f} - {max(ages.values()):.1f}")
    
    # 生成文件名前缀
    datasets_str = '+'.join(args.datasets)
    celltypes_str = '+'.join(args.celltypes)
    
    # 参考模型验证
    if args.validate_reference:
        reference_results = validate_with_reference_lasso(
            cell_data, ages, train_ids, test_ids, celltypes_str,
            save_path=os.path.join(args.output_dir, f'{celltypes_str}_{datasets_str}_{args.embedding_type}_reference_lasso.png')
        )
        
        print(f"Reference LASSO validation completed!")
        print(f"This helps verify that data split is consistent with reference code.")

    if args.skip_training:
        print("Data analysis mode - creating visualizations...")
        
        # 创建bulk数据用于可视化
        bulk_data = create_bulk_data_for_visualization(cell_data, ages, feature_names, donor_dataset_mapping)

        if donor_celltype_counts:
            # 添加细胞类型统计可视化
            celltype_stats_df = plot_celltype_statistics(
                donor_celltype_counts, args.celltypes, args.datasets,
                save_path=os.path.join(args.output_dir, f'{celltypes_str}_{datasets_str}_{args.embedding_type}_celltype_statistics.png')
            )
            
            # 保存细胞类型统计数据
            celltype_stats_df.to_csv(os.path.join(args.output_dir, f'{celltypes_str}_{datasets_str}_{args.embedding_type}_celltype_statistics.csv'), index=False)
        
        # 各种可视化...
        plot_data_overview(bulk_data, celltypes_str, args.datasets, 
                          save_path=os.path.join(args.output_dir, f'{celltypes_str}_{datasets_str}_{args.embedding_type}_data_overview.png'))
        
        plot_dimensionality_reduction(bulk_data, celltypes_str,
                                     save_path=os.path.join(args.output_dir, f'{celltypes_str}_{datasets_str}_{args.embedding_type}_dimensionality_reduction.png'))
        
        corr_results = plot_feature_correlation_analysis(bulk_data, celltypes_str,
                                                        save_path=os.path.join(args.output_dir, f'{celltypes_str}_{datasets_str}_{args.embedding_type}_feature_correlations.png'))
        corr_results.to_csv(os.path.join(args.output_dir, f'{celltypes_str}_{datasets_str}_{args.embedding_type}_correlations.csv'))
        
        plot_top_correlated_genes(bulk_data, corr_results, celltypes_str, n_genes=6,
                                  save_path=os.path.join(args.output_dir, f'{celltypes_str}_{datasets_str}_{args.embedding_type}_top_genes.png'))
        
        plot_data_split_overview(bulk_data, train_ids, val_ids, test_ids, celltypes_str,
                                save_path=os.path.join(args.output_dir, f'{celltypes_str}_{datasets_str}_{args.embedding_type}_data_split.png'))
    
        print("Data analysis completed!")

        # 仅数据分析模式
        print("Creating data analysis report...")
        report = create_data_analysis_report(bulk_data, corr_results, celltypes_str, args.datasets,
                                           train_ids, val_ids, test_ids, args,
                                           save_path=os.path.join(args.output_dir, f'{celltypes_str}_{datasets_str}_{args.embedding_type}_data_analysis_report.md'))
        
        # 保存数据分析结果
        analysis_results_path = os.path.join(args.output_dir, f'{celltypes_str}_{datasets_str}_{args.embedding_type}_data_analysis.pkl')
        with open(analysis_results_path, 'wb') as f:
            pickle.dump({
                'bulk_data': bulk_data,
                'corr_results': corr_results,
                'train_ids': train_ids,
                'val_ids': val_ids,
                'test_ids': test_ids,
                'celltype_stats_df': celltype_stats_df,
                'donor_celltype_counts': donor_celltype_counts,
                'args': vars(args)
            }, f)
        
        print(f"Data analysis completed! Results saved to: {args.output_dir}")
        print("To run training later, remove the --skip_training flag")
        return

    # 完整训练模式
    print("Preparing data loaders for training...")
    train_loader, val_loader, test_loader, scaler, max_cells = prepare_data_loaders(
        cell_data, ages, train_ids, val_ids, test_ids, batch_size=args.batch_size
    )

    # 创建模型
    print(f"Creating {args.model_type} model with input_dim={n_genes}...")

    if args.model_type == 'deepsets':
        model = DeepSetsAgePredictor(
            input_dim=n_genes, 
            hidden_dim=args.hidden_dim,
            max_cells=max_cells
        )
    elif args.model_type == 'transformer':
        model = SimpleSetTransformerAgePredictor(
            input_dim=n_genes,
            d_model=args.hidden_dim
        )
    elif args.model_type == 'sparse_transformer':
        sparse_config = {
            'sparse_type': 'random',
            'block_size': 32,  # 确保是合适的块大小
            'sparsity': 0.1,
            'model_type': 'single_cell'
        }
        
        model = UnifiedSparseTransformerAgePredictor(
            input_dim=n_genes,
            d_model=args.hidden_dim,
            n_heads=4,
            n_layers=4,
            max_cells=max_cells,
            sparse_config=sparse_config
        )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    print("Starting training...")
    save_path = os.path.join(args.output_dir, f'{celltypes_str}_{datasets_str}_{args.model_type}_{args.embedding_type}_best.pth')
    
    history = train_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs, lr=args.lr, device=device,
        save_path=save_path
    )
    
    # 加载最佳模型
    model.load_state_dict(torch.load(save_path))
    
    # 评估模型
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device)

    # 跨数据集评估
    if args.cross_dataset_eval:
        all_datasets = ['AIDA', 'eQTL', 'HCA', 'siAge']
        available_datasets = list(set(donor_dataset_mapping.values()))
        train_datasets = args.datasets
        test_datasets = [d for d in all_datasets if d in available_datasets]
        
        cross_results = evaluate_cross_dataset(
            model, cell_data, ages, donor_dataset_mapping,
            train_datasets, test_datasets, scaler, device
        )
        
        # 可视化跨数据集结果
        plot_cross_dataset_results(
            cross_results, celltypes_str, args.model_type, args.embedding_type,
            save_path=os.path.join(args.output_dir, f'{celltypes_str}_{datasets_str}_{args.model_type}_{args.embedding_type}_cross_dataset.png')
        )
    else:
        cross_results = {}
    
    # 训练相关的可视化
    print("Creating training visualizations...")

    # 创建bulk数据用于可视化
    bulk_data = create_bulk_data_for_visualization(cell_data, ages, feature_names, donor_dataset_mapping)
    
    # 训练历史
    plot_training_history(history, 
                            save_path=os.path.join(args.output_dir, f'{celltypes_str}_{datasets_str}_{args.model_type}_{args.embedding_type}_training_history.png'))
    
    # 全面的预测结果分析
    plot_prediction_results_comprehensive(results, bulk_data, celltypes_str, args.model_type,
                                        train_ids, val_ids, test_ids,
                                        save_path=os.path.join(args.output_dir, f'{celltypes_str}_{datasets_str}_{args.model_type}_{args.embedding_type}_results.png'))
    
    # 创建综合报告
    print("Creating comprehensive report...")
    report = create_comprehensive_report(results, history, bulk_data,
                                        celltypes_str, args.model_type, args.datasets, 
                                        train_ids, val_ids, test_ids, args,
                                        save_path=os.path.join(args.output_dir, f'{celltypes_str}_{datasets_str}_{args.model_type}_{args.embedding_type}_report.md'))
    
    # 保存完整结果
    results_path = os.path.join(args.output_dir, f'{celltypes_str}_{datasets_str}_{args.model_type}_{args.embedding_type}_complete_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            'results': results, 
            'cross_dataset_results': cross_results,
            'history': history, 
            'args': vars(args),
        }, f)
    
    print(f"Training completed! Results saved to: {args.output_dir}")
    
    return results, history, cross_results

if __name__ == "__main__":
    main()