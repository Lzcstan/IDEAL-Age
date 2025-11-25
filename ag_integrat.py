#!/usr/bin/env python3
"""
AutoGluon + DeepSets 缓存优化版本 - 修复版
参考AutoGluon官方教程优化集成方式
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import warnings
import gc
import pickle
import hashlib
import json
from pathlib import Path
import time
warnings.filterwarnings('ignore')

# AutoGluon imports
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.common import space

from ag_extra_models import DeepSetsTabularModel, DeepSetsTabularModelAttn

# 导入现有脚本的功能
try:
    from dist_train_cache import (
        set_seed
    )
    print("✅ Successfully imported from dist_train_cache.py", flush=True)
except ImportError as e:
    print(f"❌ Failed to import from dist_train_cache.py: {e}", flush=True)
    sys.exit(1)

try:
    from pseudo_redo import (
        BenchmarkItem
    )
    print("✅ Successfully imported from pseudo_redo.py", flush=True)
except ImportError as e:
    print(f"❌ Failed to import from pseudo_redo.py: {e}", flush=True)
    sys.exit(1)

try:
    import scanpy as sc
    print("✅ Scanpy imported successfully", flush=True)
except ImportError as e:
    print(f"❌ Failed to import scanpy: {e}", flush=True)
    sys.exit(1)

# 设置种子和全局参数
set_seed(42)

# 全局设置
# MAX_CELLS_PER_DONOR = 10000
MAX_CELLS_PER_DONOR = 1000 # Middle version
# MAX_CELLS_PER_DONOR = 500 # Fast version
# MAX_CELLS_PER_DONOR = 100 # Fastest version
CACHE_DIR = "./donor_cache_ag_marker"
CACHE_VERSION = "v1.1"
# CACHE_DIR = "./donor_cache_ag"
# CACHE_VERSION = "v1.0"

EPOCH = 60

USE_RAW = True

# ===============================
# 缓存管理系统 (保持原有逻辑)
# ===============================

class DataCache:
    """数据缓存管理器"""
    
    def __init__(self, cache_dir: str = CACHE_DIR, version: str = CACHE_VERSION):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.version = version
        print(f"📦 Cache manager initialized: {self.cache_dir}", flush=True)
    
    def _get_cache_key(self, **kwargs) -> str:
        cache_data = {
            'version': self.version,
            'max_cells_per_donor': MAX_CELLS_PER_DONOR,
            **kwargs
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str, data_type: str) -> Path:
        return self.cache_dir / f"{data_type}_{cache_key}.pkl"
    
    def _is_cache_valid(self, cache_path: Path, source_dir: str) -> bool:
        if not cache_path.exists():
            return False
        
        cache_time = cache_path.stat().st_mtime
        
        if os.path.exists(source_dir):
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.getmtime(file_path) > cache_time:
                        return False
        
        return True
    
    def save_data(self, data: Any, cache_key: str, data_type: str):
        cache_path = self._get_cache_path(cache_key, data_type)
        print(f"💾 Saving {data_type} cache: {cache_path}", flush=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'data': data,
                'timestamp': time.time(),
                'version': self.version
            }, f)
    
    def load_data(self, cache_key: str, data_type: str, source_dir: str = None) -> Optional[Any]:
        cache_path = self._get_cache_path(cache_key, data_type)
        
        if not self._is_cache_valid(cache_path, source_dir):
            return None
        
        try:
            print(f"📂 Loading {data_type} cache: {cache_path}", flush=True)
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            
            if cached.get('version') != self.version:
                print(f"⚠️ Cache version mismatch, ignoring cache", flush=True)
                return None
            
            return cached['data']
        
        except Exception as e:
            print(f"⚠️ Failed to load cache: {e}", flush=True)
            return None
    
    def clear_cache(self):
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print("🗑️ Cache cleared", flush=True)

# 全局缓存管理器
cache_manager = DataCache()

# ===============================
# 缓存优化的数据加载函数 (保持原有逻辑)
# ===============================

def get_pseudobulk_from_donor_files_cached(donor_files_dir, donor_list, dataset_name, use_raw=True, gene_names=None):
    """带缓存的pseudobulk数据获取"""
    
    cache_key = cache_manager._get_cache_key(
        dataset=dataset_name,
        donors=sorted(donor_list),
        use_raw=use_raw,
        function='pseudobulk'
    )
    
    cached_data = cache_manager.load_data(cache_key, f'pseudobulk_{dataset_name}', donor_files_dir)
    if cached_data is not None:
        print(f"  ✅ Loaded {dataset_name} pseudobulk from cache ({cached_data.shape})", flush=True)
        return cached_data
    
    print(f"  📊 Computing {dataset_name} pseudobulk ({len(donor_list)} donors)...", flush=True)
    
    pseudobulk_data = []
    
    for i, donor in enumerate(donor_list):
        donor_file = os.path.join(donor_files_dir, f'donor_{donor}.h5ad')
        
        if not os.path.exists(donor_file):
            continue
        
        try:
            donor_adata = sc.read_h5ad(donor_file)
            
            if gene_names is None:
                gene_names = donor_adata.var_names.tolist()
            
            if use_raw and 'counts' in donor_adata.layers:
                donor_expr = donor_adata[:, gene_names].layers['counts'].mean(axis=0).A1
            else:
                donor_expr = donor_adata[:, gene_names].X.mean(axis=0).A1
            
            pseudobulk_data.append(donor_expr)
            
            del donor_adata
            gc.collect()
            
            if (i + 1) % 100 == 0:
                print(f"    📈 Processed {i + 1}/{len(donor_list)} donors", flush=True)
        
        except Exception as e:
            print(f"    ⚠️ Error processing donor {donor}: {e}", flush=True)
            continue
    
    if not pseudobulk_data:
        raise ValueError(f"No valid donor data found for {dataset_name}")
    
    pseudobulk_df = pd.DataFrame(
        data=np.array(pseudobulk_data),
        index=[donor_list[i] for i in range(len(pseudobulk_data))],
        columns=gene_names
    )
    
    cache_manager.save_data(pseudobulk_df, cache_key, f'pseudobulk_{dataset_name}')
    
    print(f"  ✅ {dataset_name} pseudobulk computed and cached: {pseudobulk_df.shape}", flush=True)
    return pseudobulk_df

def load_single_cell_data_cached(donor_files_dir, donor_list, subset_name="training", use_raw=True, gene_names=None):
    """带缓存的单细胞数据加载"""
    
    cache_key = cache_manager._get_cache_key(
        donors=sorted(donor_list),
        subset=subset_name,
        use_raw=use_raw,
        function='single_cell'
    )
    
    cached_data = cache_manager.load_data(cache_key, f'single_cell_{subset_name}', donor_files_dir)
    if cached_data is not None:
        cell_data, actual_max_cells = cached_data
        print(f"  ✅ Loaded {subset_name} single cell data from cache:", flush=True)
        print(f"      {len(cell_data)} donors, max_cells={actual_max_cells}", flush=True)
        return cell_data, actual_max_cells
    
    print(f"  🔬 Computing {subset_name} single cell data ({len(donor_list)} donors)...", flush=True)
    print(f"  📏 Max cells per donor: {MAX_CELLS_PER_DONOR}", flush=True)
    
    cell_data = {}
    processed_count = 0
    actual_max_cells = 0
    
    for donor in donor_list:
        donor_file = os.path.join(donor_files_dir, f'donor_{donor}.h5ad')
        
        if not os.path.exists(donor_file):
            continue
        
        try:
            donor_adata = sc.read_h5ad(donor_file)

            if gene_names is None:
                gene_names = donor_adata.var_names.tolist()
            
            if use_raw and 'counts' in donor_adata.layers:
                cells = donor_adata[:, gene_names].layers['counts'].toarray().astype(np.float32)
            else:
                cells = donor_adata[:, gene_names].X.toarray().astype(np.float32)
            
            if cells.shape[0] > MAX_CELLS_PER_DONOR:
                print(f'    >> donor {donor} has a lot of cells.')
                idx = np.random.choice(cells.shape[0], MAX_CELLS_PER_DONOR, replace=False)
                cells = cells[idx]
            
            cell_data[donor] = cells
            actual_max_cells = max(actual_max_cells, cells.shape[0])
            processed_count += 1
            
            del donor_adata, cells
            gc.collect()
            
            if processed_count % 50 == 0:
                print(f"    📈 Loaded {processed_count} donors", flush=True)
        
        except Exception as e:
            print(f"    ⚠️ Error loading donor {donor}: {e}", flush=True)
            continue
    
    result_data = (cell_data, actual_max_cells)
    
    cache_manager.save_data(result_data, cache_key, f'single_cell_{subset_name}')
    
    print(f"  ✅ {subset_name} single cell data computed and cached:", flush=True)
    print(f"      {len(cell_data)} donors, max_cells={actual_max_cells}", flush=True)
    
    return cell_data, actual_max_cells

def get_available_donors_cached(donor_files_dir, donor_info):
    """带缓存的可用donors列表获取"""
    
    cache_key = cache_manager._get_cache_key(
        function='available_donors',
        donor_info_shape=donor_info.shape
    )
    
    cached_data = cache_manager.load_data(cache_key, 'available_donors', donor_files_dir)
    if cached_data is not None:
        print(f"  ✅ Loaded available donors from cache: {len(cached_data)} donors", flush=True)
        return cached_data
    
    print("  📋 Computing available donors...", flush=True)
    available_files = [f for f in os.listdir(donor_files_dir) 
                      if f.startswith('donor_') and f.endswith('.h5ad')]
    available_donors = [f.replace('donor_', '').replace('.h5ad', '') for f in available_files]
    
    cache_manager.save_data(available_donors, cache_key, 'available_donors')
    
    print(f"  ✅ Available donors computed and cached: {len(available_donors)} donors", flush=True)
    return available_donors

# ===============================
# 缓存优化的数据准备函数 (保持原有逻辑)
# ===============================

def prepare_data_cached(use_marker=True):
    """缓存优化的数据准备函数"""
    print("📥 Loading data with intelligent caching...", flush=True)
    print(f"🔧 MAX_CELLS_PER_DONOR = {MAX_CELLS_PER_DONOR}", flush=True)
    print(f"📦 Cache directory: {cache_manager.cache_dir}", flush=True)
    
    original_dir = os.getcwd()
    os.chdir("/personal/ImmAge")
    
    try:
        # 读取donor metadata
        donor_info = pd.read_excel("donor_metadata.xlsx", index_col=0)
        if use_marker:
            df_marker = pd.read_csv("marker_genes.csv", index_col=0)
        donor_files_dir = "donor_files"
        
        print(f"Donor info: {donor_info.shape}", flush=True)
        
        if not os.path.exists(donor_files_dir):
            raise FileNotFoundError(f"Donor files directory not found: {donor_files_dir}")
        
        # 获取可用donors（带缓存）
        available_donors = get_available_donors_cached(donor_files_dir, donor_info)
        
        # 准备各数据集的donor列表
        datasets_donors = {
            'HCA': [d for d in donor_info[donor_info["dataset"] == "HCA"].index.tolist() 
                   if d in available_donors],
            'siAge': [d for d in donor_info[donor_info["dataset"] == "siAge"].index.tolist() 
                     if d in available_donors],
            'AIDA_train': [d for d in donor_info.loc[
                (donor_info["dataset"] == "AIDA") & donor_info["is_train"]
            ].index.tolist() if d in available_donors],
            'AIDA_test': [d for d in donor_info.loc[
                (donor_info["dataset"] == "AIDA") & ~donor_info["is_train"]
            ].index.tolist() if d in available_donors],
            'eQTL_train': [d for d in donor_info.loc[
                (donor_info["dataset"] == "eQTL") & donor_info["is_train"]
            ].index.tolist() if d in available_donors],
            'eQTL_test': [d for d in donor_info.loc[
                (donor_info["dataset"] == "eQTL") & ~donor_info["is_train"]
            ].index.tolist() if d in available_donors]
        }
        
        # 处理所有pseudobulk数据（带缓存）
        print("\n📊 Processing all pseudobulk data with caching...", flush=True)
        
        pseudobulk_data = {}
        for dataset_name, donors in datasets_donors.items():
            if donors:
                print(f"Processing {dataset_name} ({len(donors)} donors)...", flush=True)
                if use_marker:
                    pseudobulk_data[dataset_name] = get_pseudobulk_from_donor_files_cached(
                        donor_files_dir, donors, dataset_name, gene_names=df_marker.index, use_raw=USE_RAW
                    )
                else:
                    pseudobulk_data[dataset_name] = get_pseudobulk_from_donor_files_cached(
                        donor_files_dir, donors, dataset_name, use_raw=USE_RAW
                    )
        
        # 合并训练数据
        train_data = pd.concat([
            pseudobulk_data['AIDA_train'], 
            pseudobulk_data['eQTL_train']
        ], axis=0)
        train_age = donor_info.loc[train_data.index, "age"]

        test_data = pd.concat([
            pseudobulk_data['AIDA_test'],
            pseudobulk_data['eQTL_test'],
            pseudobulk_data['HCA'],
            pseudobulk_data['siAge']
        ], axis=0)
        test_age = donor_info.loc[test_data.index, "age"]
        
        # 获取各数据集的年龄信息
        age_data = {}
        for dataset_name, pseudo_df in pseudobulk_data.items():
            age_data[dataset_name] = donor_info.loc[pseudo_df.index, "age"]
        
        gc.collect()
        
        # 准备单细胞数据（带缓存）
        print("\n🔬 Loading single cell data with caching...", flush=True)
        train_donors_subset = train_data.index.tolist()
        # train_donors_subset = train_data.index.tolist()[:200] # Fast
        test_donors_subset = test_data.index.tolist()
        
        if use_marker:
            train_cell_data, train_max_cells = load_single_cell_data_cached(
                donor_files_dir, train_donors_subset, "training_subset", gene_names=df_marker.index, use_raw=USE_RAW
            )
            test_cell_data, test_max_cells = load_single_cell_data_cached(
                donor_files_dir, test_donors_subset, "testing_subset", gene_names=df_marker.index, use_raw=USE_RAW
            )
        else:
            train_cell_data, train_max_cells = load_single_cell_data_cached(
                donor_files_dir, train_donors_subset, "training_subset", use_raw=USE_RAW
            )
            test_cell_data, test_max_cells = load_single_cell_data_cached(
                donor_files_dir, test_donors_subset, "testing_subset", use_raw=USE_RAW
            )
        
        print(f"\n✅ All data prepared with caching!", flush=True)
        print(f"   Train data: {train_data.shape}", flush=True)
        print(f"   Train cell data: {len(train_cell_data)} donors", flush=True)
        print(f"   Test data: {test_data.shape}", flush=True)
        print(f"   Test cell data: {len(test_cell_data)} donors", flush=True)
        print(f"   📏 Train max cells: {train_max_cells}", flush=True)
        print(f"   📏 Test max cells: {test_max_cells}", flush=True)
        
        return {
            'train_data': train_data,
            'train_age': train_age,
            # 'test_data': test_data,
            # 'test_age': test_age,
            'test_AIDA_pseudo': pseudobulk_data['AIDA_test'],
            'test_AIDA_age': age_data['AIDA_test'],
            'test_eQTL_pseudo': pseudobulk_data['eQTL_test'],
            'test_eQTL_age': age_data['eQTL_test'],
            'HCA_pseudo': pseudobulk_data['HCA'],
            'HCA_age': age_data['HCA'],
            'siAge_pseudo': pseudobulk_data['siAge'],
            'siAge_age': age_data['siAge'],
            'train_cell_data': train_cell_data,
            'train_max_cells': train_max_cells,
            'test_cell_data': test_cell_data,
            'test_max_cells': test_max_cells
        }
    
    finally:
        os.chdir(original_dir)

# ===============================
# 修复的AutoGluon训练函数
# ===============================

def train_autogluon_with_deepsets(data):
    """修复的AutoGluon训练函数 - 分步评估"""
    print("\n🤖 Training AutoGluon with DeepSets (Fixed)...", flush=True)
    
    # 设置共享的cell data
    DeepSetsTabularModelAttn._shared_train_cell_data = data['train_cell_data']
    DeepSetsTabularModelAttn._shared_train_max_cells = data['train_max_cells']
    DeepSetsTabularModelAttn._shared_test_cell_data = data['test_cell_data']
    DeepSetsTabularModelAttn._shared_test_max_cells = data['test_max_cells']
    DeepSetsTabularModelAttn._shared_cell_data = data['train_cell_data'] | data['test_cell_data']
    DeepSetsTabularModelAttn._shared_max_cells = max(data['train_max_cells'], data['test_max_cells'])
    
    print(f"📏 Shared train_max_cells: {data['train_max_cells']}", flush=True)
    print(f"📏 Shared test_max_cells: {data['test_max_cells']}", flush=True)

    if not USE_RAW:
        DeepSetsTabularModelAttn.is_scaled = True
    
    # 准备训练数据
    train_data_with_age = pd.concat([data['train_data'], data['train_age']], axis=1)
    print(f"Training data: {train_data_with_age.shape}", flush=True)
    
    # 创建TabularDataset
    train_dataset = TabularDataset(train_data_with_age)
    
    # # 创建预测器
    # predictor = TabularPredictor(
    #     label="age",
    #     problem_type="regression", 
    #     path="AutogluonModels/deepsets_hpo",
    #     verbosity=2
    # )

    # custom_hyperparameters_hpo = {DeepSetsTabularModelAttn: {
    #     'hidden_dim': space.Int(lower=4, upper=10, default=10),
    #     # 'dropout': space.Real(lower=0.0, upper=0.9),
    #     'lr': space.Real(log=True, lower=1e-3, upper=1e-1),
    # }}

    # predictor = TabularPredictor(label="age").fit(
    #     train_dataset,
    #     hyperparameters=custom_hyperparameters_hpo,
    #     hyperparameter_tune_kwargs='auto',  # enables HPO
    # )

    # leaderboard_hpo = predictor.leaderboard()
    # print(leaderboard_hpo)

    # best_model_name = leaderboard_hpo[leaderboard_hpo['stack_level'] == 1]['model'].iloc[0]

    # predictor_info = predictor.info()
    # best_model_info = predictor_info['model_info'][best_model_name]

    # print(best_model_info)

    # print(f'Best Model Hyperparameters ({best_model_name}):')
    # print(best_model_info['hyperparameters'])
    
    # 🔥 第一步：先训练标准模型
    print("🚀 Step 1: Training standard AutoGluon models...", flush=True)

    # 创建预测器
    predictor = TabularPredictor(
        label="age",
        problem_type="regression", 
        path="AutogluonModels/deepsetsattn_integration",
        verbosity=2,
        # eval_metric='mean_absolute_error'
    )
    
    predictor.fit(
        train_dataset,
        time_limit=600,  # 10分钟
        included_model_types=['FASTAI', 'NN_TORCH']
    )
    
    print("✅ Standard models training completed!", flush=True)
    
    # 🔍 评估标准模型
    print("\n📊 Evaluating Standard Models (Before DeepSets)...", flush=True)
    standard_results, standard_preds = evaluate_all_datasets(predictor, data, model_suffix="standard")
    
    # 显示标准模型leaderboard
    try:
        leaderboard_standard = predictor.leaderboard(silent=True)
        print("📊 Standard Models Leaderboard:", flush=True)
        print(leaderboard_standard.head(10), flush=True)
    except Exception as e:
        print(f"⚠️ Could not display standard leaderboard: {e}", flush=True)
    
    # 🔥 第二步：使用fit_extra添加自定义DeepSets模型
    print("\n🚀 Step 2: Adding custom DeepSets model...", flush=True)


    hyperparameters = {
        'NN_TORCH': {},
        'FASTAI': {},
        # DeepSetsTabularModelAttn: best_model_info['hyperparameters']
        DeepSetsTabularModelAttn: {'hidden_dim': 10, 'dropout': 0.2, 'lr': 0.001, 'batch_size': 8, 'epochs': EPOCH} # Tid
    }
    
    try:
        # 使用fit_extra添加自定义模型
        predictor.fit_extra(
            hyperparameters=hyperparameters,
            # time_limit=1800,   # 30分钟训练自定义模型
        )
        
        print("✅ Custom DeepSets model added successfully!", flush=True)
        
        # 🔍 评估包含DeepSets的模型
        print("\n📊 Evaluating Models with DeepSets (After DeepSets)...", flush=True)
        deepsets_results, deepsets_preds = evaluate_all_datasets(predictor, data, model_suffix="with_deepsets")
        
        # 显示最终leaderboard
        try:
            leaderboard_final = predictor.leaderboard(silent=True)
            print("📊 Final Model Leaderboard (With DeepSets):", flush=True)
            print(leaderboard_final.head(10), flush=True)
        except Exception as e:
            print(f"⚠️ Could not display final leaderboard: {e}", flush=True)
        
        # 📈 性能对比分析
        print("\n🔬 Performance Comparison Analysis:", flush=True)
        print("="*70, flush=True)
        print(f"{'Dataset':<10} {'Standard r':<12} {'DeepSets r':<12} {'Improvement':<12} {'Standard MAE':<12} {'DeepSets MAE':<12} {'MAE Change':<12}")
        print("-"*70, flush=True)
        
        for dataset in ['AIDA', 'eQTL', 'HCA', 'siAge']:
            if dataset in standard_results and dataset in deepsets_results:
                std_r = standard_results[dataset].pcc
                ds_r = deepsets_results[dataset].pcc
                std_mae = standard_results[dataset].mae
                ds_mae = deepsets_results[dataset].mae
                
                r_improvement = ds_r - std_r
                mae_change = ds_mae - std_mae
                
                print(f"{dataset:<10} {std_r:<12.3f} {ds_r:<12.3f} {r_improvement:<+12.3f} {std_mae:<12.3f} {ds_mae:<12.3f} {mae_change:<+12.3f}")

        df_all = save_all_predictions(standard_preds, deepsets_preds, save_dir="./results_debug")
        
        # 🔥 新增：创建可视化对比
        visualize_prediction_comparison(df_all, save_dir="./results_debug")
        
        return predictor, standard_results, deepsets_results
        
    except Exception as e:
        print(f"❌ Custom model integration failed: {e}", flush=True)
        print("🔄 Returning standard models only...", flush=True)
        return predictor, standard_results, None

# def evaluate_all_datasets(predictor, data, model_suffix=""):
#     """评估所有数据集"""
#     print(f"\n📈 Evaluating all datasets {model_suffix}...", flush=True)
    
#     test_sets = {
#         'AIDA': (data['test_AIDA_pseudo'], data['test_AIDA_age']),
#         'eQTL': (data['test_eQTL_pseudo'], data['test_eQTL_age']),
#         'HCA': (data['HCA_pseudo'], data['HCA_age']),
#         'siAge': (data['siAge_pseudo'], data['siAge_age'])
#     }
    
#     colors = {
#         'AIDA': '#c57541',
#         'eQTL': '#777acc', 
#         'HCA': '#73a85d',
#         'siAge': '#c45a95'
#     }
    
#     results = {}
    
#     for dataset_name, (test_pseudo, test_age) in test_sets.items():
#         print(f"\n📊 {dataset_name} Dataset {model_suffix}:", flush=True)
        
#         try:
#             # 进行预测
#             ag_pred = predictor.predict(test_pseudo)
#             ag_item = BenchmarkItem(ag_pred, test_age)
            
#             print(f"  Pearson r: {ag_item.pcc:.3f}", flush=True)
#             print(f"  Spearman ρ: {ag_item.rho:.3f}", flush=True)
#             print(f"  MAE: {ag_item.mae:.3f}", flush=True)
#             print(f"  RAE: {ag_item.rae:.3f}", flush=True)
            
#             # 保存散点图 - 添加suffix到文件名
#             file_suffix = f"_{model_suffix}" if model_suffix else ""
#             ag_item.scatter_plot(
#                 color=colors[dataset_name], 
#                 title=f"AutoGluon + DeepSets {model_suffix} - {dataset_name}",
#                 save_dir="./results_debug",
#                 filename=f"fixed_autogluon_deepsets_{dataset_name.lower()}{file_suffix}"
#             )
            
#             results[dataset_name] = ag_item
            
#         except Exception as e:
#             print(f"  ❌ Error evaluating {dataset_name}: {e}", flush=True)
#             continue
    
#     return results

def evaluate_all_datasets(predictor, data, model_suffix=""):
    """评估所有数据集并收集预测结果"""
    print(f"\n📈 Evaluating all datasets {model_suffix}...", flush=True)
    
    test_sets = {
        'AIDA': (data['test_AIDA_pseudo'], data['test_AIDA_age']),
        'eQTL': (data['test_eQTL_pseudo'], data['test_eQTL_age']),
        'HCA': (data['HCA_pseudo'], data['HCA_age']),
        'siAge': (data['siAge_pseudo'], data['siAge_age'])
    }
    
    colors = {
        'AIDA': '#c57541',
        'eQTL': '#777acc', 
        'HCA': '#73a85d',
        'siAge': '#c45a95'
    }
    
    results = {}
    all_predictions = {}  # 新增：存储所有预测结果
    
    for dataset_name, (test_pseudo, test_age) in test_sets.items():
        print(f"\n📊 {dataset_name} Dataset {model_suffix}:", flush=True)
        
        try:
            # 进行预测
            ag_pred = predictor.predict(test_pseudo)
            ag_item = BenchmarkItem(ag_pred, test_age)
            
            print(f"  Pearson r: {ag_item.pcc:.3f}", flush=True)
            print(f"  Spearman ρ: {ag_item.rho:.3f}", flush=True)
            print(f"  MAE: {ag_item.mae:.3f}", flush=True)
            print(f"  RAE: {ag_item.rae:.3f}", flush=True)
            
            # 保存散点图
            file_suffix = f"_{model_suffix}" if model_suffix else ""
            ag_item.scatter_plot(
                color=colors[dataset_name], 
                title=f"AutoGluon + DeepSets {model_suffix} - {dataset_name}",
                save_dir="./results_debug",
                filename=f"fixed_autogluon_deepsets_{dataset_name.lower()}{file_suffix}"
            )
            
            results[dataset_name] = ag_item
            
            # 🔥 新增：保存预测结果
            all_predictions[dataset_name] = {
                'donor_ids': test_pseudo.index.tolist(),
                'true_age': test_age.values,
                'predicted_age': ag_pred.values
            }
            
        except Exception as e:
            print(f"  ❌ Error evaluating {dataset_name}: {e}", flush=True)
            continue
    
    return results, all_predictions  # 返回预测结果

def save_all_predictions(standard_preds, deepsets_preds, save_dir="./results_debug"):
    """
    保存所有donor的真实年龄和各模型预测结果
    
    Args:
        standard_preds: dict, 标准模型的预测结果
        deepsets_preds: dict, 包含DeepSets的模型预测结果
        save_dir: str, 保存目录
    """
    print("\n💾 Saving all predictions to CSV...", flush=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # 为每个数据集创建一个DataFrame
    for dataset_name in standard_preds.keys():
        if dataset_name not in deepsets_preds:
            continue
            
        std_data = standard_preds[dataset_name]
        ds_data = deepsets_preds[dataset_name]
        
        # 创建DataFrame
        df = pd.DataFrame({
            'donor_id': std_data['donor_ids'],
            'true_age': std_data['true_age'],
            'predicted_age_standard': std_data['predicted_age'],
            'predicted_age_with_deepsets': ds_data['predicted_age'],
        })
        
        # 计算误差
        df['error_standard'] = df['predicted_age_standard'] - df['true_age']
        df['error_with_deepsets'] = df['predicted_age_with_deepsets'] - df['true_age']
        df['abs_error_standard'] = df['error_standard'].abs()
        df['abs_error_with_deepsets'] = df['error_with_deepsets'].abs()
        
        # 计算改进
        df['improvement'] = df['abs_error_standard'] - df['abs_error_with_deepsets']
        
        # 保存CSV
        csv_path = os.path.join(save_dir, f"predictions_{dataset_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  ✅ Saved {dataset_name}: {csv_path}", flush=True)
        
        # 打印统计信息
        print(f"\n  📊 {dataset_name} Statistics:", flush=True)
        print(f"     Total donors: {len(df)}", flush=True)
        print(f"     MAE (standard): {df['abs_error_standard'].mean():.3f}", flush=True)
        print(f"     MAE (with DeepSets): {df['abs_error_with_deepsets'].mean():.3f}", flush=True)
        print(f"     Donors improved: {(df['improvement'] > 0).sum()} ({(df['improvement'] > 0).sum()/len(df)*100:.1f}%)", flush=True)
        print(f"     Donors worsened: {(df['improvement'] < 0).sum()} ({(df['improvement'] < 0).sum()/len(df)*100:.1f}%)", flush=True)
    
    # 创建汇总文件
    all_datasets = []
    for dataset_name in standard_preds.keys():
        if dataset_name not in deepsets_preds:
            continue
        std_data = standard_preds[dataset_name]
        ds_data = deepsets_preds[dataset_name]
        
        df_temp = pd.DataFrame({
            'dataset': dataset_name,
            'donor_id': std_data['donor_ids'],
            'true_age': std_data['true_age'],
            'predicted_age_standard': std_data['predicted_age'],
            'predicted_age_with_deepsets': ds_data['predicted_age'],
        })
        all_datasets.append(df_temp)
    
    df_all = pd.concat(all_datasets, ignore_index=True)
    df_all['error_standard'] = df_all['predicted_age_standard'] - df_all['true_age']
    df_all['error_with_deepsets'] = df_all['predicted_age_with_deepsets'] - df_all['true_age']
    df_all['abs_error_standard'] = df_all['error_standard'].abs()
    df_all['abs_error_with_deepsets'] = df_all['error_with_deepsets'].abs()
    df_all['improvement'] = df_all['abs_error_standard'] - df_all['abs_error_with_deepsets']
    
    summary_path = os.path.join(save_dir, "predictions_all_datasets.csv")
    df_all.to_csv(summary_path, index=False)
    print(f"\n  ✅ Saved combined results: {summary_path}", flush=True)
    
    return df_all

def visualize_prediction_comparison(df_all, save_dir="./results_debug"):
    """可视化预测结果对比"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\n📊 Creating comparison visualizations...", flush=True)
    
    # 1. 误差分布对比（所有数据集）
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].hist(df_all['abs_error_standard'], bins=50, alpha=0.6, label='Standard', color='blue')
    axes[0].hist(df_all['abs_error_with_deepsets'], bins=50, alpha=0.6, label='With DeepSets', color='red')
    axes[0].set_xlabel('Absolute Error (years)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Error Distribution Comparison')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 2. 改进分布
    axes[1].hist(df_all['improvement'], bins=50, alpha=0.7, color='green')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Improvement (years)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Prediction Improvement Distribution\n(Positive = Better with DeepSets)')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_comparison_all.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 按数据集分组的对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    datasets = df_all['dataset'].unique()
    for idx, dataset in enumerate(datasets):
        if idx >= 4:
            break
        df_subset = df_all[df_all['dataset'] == dataset]
        
        axes[idx].scatter(df_subset['true_age'], df_subset['predicted_age_standard'], 
                         alpha=0.5, s=30, label='Standard', color='blue')
        axes[idx].scatter(df_subset['true_age'], df_subset['predicted_age_with_deepsets'], 
                         alpha=0.5, s=30, label='With DeepSets', color='red')
        
        # 添加对角线
        min_age = min(df_subset['true_age'].min(), 
                     df_subset['predicted_age_standard'].min(),
                     df_subset['predicted_age_with_deepsets'].min())
        max_age = max(df_subset['true_age'].max(),
                     df_subset['predicted_age_standard'].max(),
                     df_subset['predicted_age_with_deepsets'].max())
        axes[idx].plot([min_age, max_age], [min_age, max_age], 'k--', alpha=0.3)
        
        axes[idx].set_xlabel('True Age')
        axes[idx].set_ylabel('Predicted Age')
        axes[idx].set_title(f'{dataset} Dataset')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scatter_comparison_by_dataset.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Visualizations saved to {save_dir}", flush=True)


# ===============================
# 缓存管理工具 (保持原有逻辑)  
# ===============================

def cache_management():
    """缓存管理功能"""
    print("\n🗂️ Cache Management Options:", flush=True)
    print("1. Show cache info", flush=True)
    print("2. Clear data cache", flush=True)
    print("3. Clear DeepSets weights cache", flush=True)
    print("4. Clear all cache", flush=True)
    print("5. Continue with training", flush=True)
    
    choice = input("Enter your choice (1-5): ").strip()
    
    if choice == '1':
        # 显示数据缓存信息
        cache_files = list(cache_manager.cache_dir.glob("*.pkl"))
        print(f"\n📊 Data Cache Status:", flush=True)
        print(f"   Cache directory: {cache_manager.cache_dir}", flush=True)
        print(f"   Number of cache files: {len(cache_files)}", flush=True)
        
        total_size = sum(f.stat().st_size for f in cache_files)
        print(f"   Total cache size: {total_size / (1024*1024):.1f} MB", flush=True)
        
        if cache_files:
            print("\n   Data cache files:", flush=True)
            for f in cache_files:
                size_mb = f.stat().st_size / (1024*1024)
                mod_time = time.ctime(f.stat().st_mtime)
                print(f"     {f.name}: {size_mb:.1f}MB (modified: {mod_time})", flush=True)
        
        # 显示DeepSets权重缓存信息
        from pathlib import Path
        weights_dir = Path("./results_debug/deepsets_weights")
        if weights_dir.exists():
            weight_files = list(weights_dir.glob("*.pth"))
            print(f"\n📊 DeepSets Weights Cache Status:", flush=True)
            print(f"   Cache directory: {weights_dir}", flush=True)
            print(f"   Number of weight files: {len(weight_files)}", flush=True)
            
            weights_size = sum(f.stat().st_size for f in weight_files)
            print(f"   Total weights size: {weights_size / (1024*1024):.1f} MB", flush=True)
            
            if weight_files:
                print("\n   Weight cache files:", flush=True)
                for f in weight_files:
                    size_mb = f.stat().st_size / (1024*1024)
                    mod_time = time.ctime(f.stat().st_mtime)
                    print(f"     {f.name}: {size_mb:.1f}MB (modified: {mod_time})", flush=True)
        
        return cache_management()
    
    elif choice == '2':
        confirm = input("Are you sure you want to clear data cache? (y/N): ").strip().lower()
        if confirm == 'y':
            cache_manager.clear_cache()
            print("✅ Data cache cleared!", flush=True)
        return True
    
    elif choice == '3':
        confirm = input("Are you sure you want to clear DeepSets weights cache? (y/N): ").strip().lower()
        if confirm == 'y':
            from pathlib import Path
            import shutil
            weights_dir = Path("./results_debug/deepsets_weights")
            if weights_dir.exists():
                shutil.rmtree(weights_dir)
                print("✅ DeepSets weights cache cleared!", flush=True)
            else:
                print("⚠️ No weights cache found", flush=True)
        return True
    
    elif choice == '4':
        confirm = input("Are you sure you want to clear ALL cache? (y/N): ").strip().lower()
        if confirm == 'y':
            # 清理数据缓存
            cache_manager.clear_cache()
            # 清理权重缓存
            from pathlib import Path
            import shutil
            weights_dir = Path("./results_debug/deepsets_weights")
            if weights_dir.exists():
                shutil.rmtree(weights_dir)
            print("✅ All cache cleared!", flush=True)
        return True
    
    else:
        return True

def analyze_deepsets_contributions(predictor, data, donor_ids_to_analyze=None):
    """分析DeepSets模型的贡献"""
    
    print("\n🔬 Starting DeepSets Contribution Analysis...", flush=True)
    
    # 获取DeepSets模型实例
    try:
        # 方法1：通过trainer获取
        deepsets_model = None
        model_names = predictor.model_names()
        print(f"📋 Available models: {model_names}", flush=True)
        
        # 查找DeepSets模型
        deepsets_name = None
        for name in model_names:
            if 'DeepSets' in name:
                deepsets_name = name
                break
        
        if deepsets_name is None:
            print("⚠️ No DeepSets model found, skipping contribution analysis", flush=True)
            return None
            
        print(f"🎯 Found DeepSets model: {deepsets_name}", flush=True)
        
        # 通过trainer获取模型实例
        deepsets_model = predictor._trainer.load_model(deepsets_name)
        
        if deepsets_model is None:
            print("❌ Failed to load DeepSets model", flush=True)
            return None
            
    except Exception as e:
        print(f"❌ Error loading DeepSets model: {e}", flush=True)
        return None
    
    # 准备分析的供体列表
    if donor_ids_to_analyze is None:
        available_donors = list(data['test_cell_data'].keys())
        donor_ids_to_analyze = available_donors[:5]  # 分析前5个供体
    
    print(f"📊 Analyzing {len(donor_ids_to_analyze)} donors: {donor_ids_to_analyze[:3]}{'...' if len(donor_ids_to_analyze) > 3 else ''}", flush=True)
    
    try:
        # 获取细胞贡献
        print("\n🔬 Computing cell contributions...", flush=True)
        cell_contrib = deepsets_model.get_cell_contributions(
            donor_ids_to_analyze, method='activation' # Tid: gradient, grad_input, attention, activation, integrated_gradient
        )
        
        # 获取基因贡献
        print("\n🧬 Computing gene contributions...", flush=True)
        gene_names = data['train_data'].columns.tolist()
        gene_contrib = deepsets_model.get_gene_contributions(
            donor_ids_to_analyze, gene_names=gene_names, method='grad_input' # Tid: gradient, grad_input, integrated_gradient
        )
        
        # 可视化前2个供体的结果
        print("\n📊 Creating visualizations...", flush=True)
        for i, donor_id in enumerate(donor_ids_to_analyze[:2]):
            print(f"📈 Visualizing donor {donor_id}...", flush=True)
            
            deepsets_model.visualize_contributions(
                cell_contrib, donor_id, plot_type='cell', save_dir="./results_debug"
            )
            deepsets_model.visualize_contributions(
                gene_contrib, donor_id, plot_type='gene', gene_names=gene_names, save_dir="./results_debug"
            )
        
        # 打印总结
        print("\n📋 Contribution Analysis Summary:", flush=True)
        print("-" * 50, flush=True)
        
        for donor_id in donor_ids_to_analyze:
            if donor_id in cell_contrib and donor_id in gene_contrib:
                cell_result = cell_contrib[donor_id]
                gene_result = gene_contrib[donor_id]
                
                print(f"\n🔬 Donor {donor_id}:", flush=True)
                print(f"  Predicted Age: {cell_result['age_pred']:.1f} years", flush=True)
                print(f"  Cells analyzed: {cell_result['n_cells']}", flush=True)
                print(f"  Genes analyzed: {gene_result['n_genes']}", flush=True)
                print(f"  Top contributing cell: #{np.argmax(cell_result['cell_contributions'])} "
                      f"(score: {np.max(cell_result['cell_contributions']):.3f})", flush=True)
                print(f"  Top 3 genes: {[f'{name}({score:.3f})' for name, score in gene_result['top_genes'][:3]]}", flush=True)
        
        print(f"\n✅ Contribution analysis completed!", flush=True)
        print(f"📁 Visualizations saved in ./results_debug/", flush=True)
        
        return {
            'cell_contributions': cell_contrib,
            'gene_contributions': gene_contrib,
            'analyzed_donors': donor_ids_to_analyze
        }
        
    except Exception as e:
        print(f"❌ Error during contribution analysis: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None
    
def main():
    """主函数"""
    print("🎯 AutoGluon + DeepSets with Intelligent Caching (FIXED)", flush=True)
    print("=" * 60, flush=True)
    print(f"🔧 MAX_CELLS_PER_DONOR = {MAX_CELLS_PER_DONOR}", flush=True)
    print(f"📦 Cache version = {CACHE_VERSION}", flush=True)
    
    os.makedirs("./results_debug", exist_ok=True)
    
    # 缓存管理选项
    if not cache_management():
        return None, None, None
    
    try:
        start_time = time.time()
        
        # 数据准备（带缓存）
        data = prepare_data_cached() if 'marker' in CACHE_DIR else prepare_data_cached(use_marker=False)
        
        data_time = time.time()
        print(f"\n⏱️ Data preparation took: {data_time - start_time:.1f}s", flush=True)
        
        # 一致性检查
        print(f"\n✅ Consistency check:", flush=True)
        print(f"   MAX_CELLS_PER_DONOR: {MAX_CELLS_PER_DONOR}", flush=True)
        print(f"   actual train_max_cells: {data['train_max_cells']}", flush=True)
        print(f"   Consistent: {data['train_max_cells'] <= MAX_CELLS_PER_DONOR}", flush=True)
        print(f"   actual test_max_cells: {data['test_max_cells']}", flush=True)
        print(f"   Consistent: {data['test_max_cells'] <= MAX_CELLS_PER_DONOR}", flush=True)
        
        # 训练和评估
        result = train_autogluon_with_deepsets(data)
        if len(result) == 3:
            predictor, standard_results, deepsets_results = result
        else:
            predictor, standard_results = result
            deepsets_results = None
        
        train_time = time.time()
        print(f"⏱️ Training took: {train_time - data_time:.1f}s", flush=True)
        print(f"⏱️ Total runtime: {train_time - start_time:.1f}s", flush=True)
        
        print("\n🎉 Integration completed successfully!", flush=True)
        print("📁 Results saved to ./results_debug/", flush=True)
        print("📦 Data cached for faster future runs!", flush=True)
        
        print("\n📋 Standard Models Summary:", flush=True)
        for dataset, benchmark in standard_results.items():
            print(f"  {dataset}: r={benchmark.pcc:.3f}, MAE={benchmark.mae:.3f}", flush=True)
        
        if deepsets_results:
            print("\n📋 Models with DeepSets Summary:", flush=True)
            for dataset, benchmark in deepsets_results.items():
                print(f"  {dataset}: r={benchmark.pcc:.3f}, MAE={benchmark.mae:.3f}", flush=True)

        # 🔬 添加贡献分析
        if deepsets_results is not None:  # 只有当DeepSets训练成功时才分析
            print("\n" + "="*60, flush=True)
            print("🔬 DEEPSETS CONTRIBUTION ANALYSIS", flush=True)
            print("="*60, flush=True)
            
            contribution_results = analyze_deepsets_contributions(predictor, data)
            
            if contribution_results:
                print("\n🎊 All analyses completed successfully!", flush=True)
                return predictor, standard_results, deepsets_results, contribution_results
        
        return predictor, standard_results, deepsets_results, None
        
    except Exception as e:
        print(f"❌ Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    results = main()
    
    if results[0]:  # 如果有predictor
        predictor, standard_results, deepsets_results, contrib_results = results
        print("\n✅ Done! Next run will be much faster thanks to caching!", flush=True)
        print("  predictor.predict(new_data)  # Make predictions", flush=True)
        print("  Check ./results_debug/ for plots and contribution analysis", flush=True)
        print(f"  Cache stored in: {CACHE_DIR}", flush=True)
        
        if contrib_results:
            print("\n🔬 Contribution analysis results available:", flush=True)
            print("  - Cell contribution plots", flush=True)
            print("  - Gene contribution analysis", flush=True)
            print("  - Visualization saved in ./results_debug/", flush=True)