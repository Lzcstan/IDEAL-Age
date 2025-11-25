#!/usr/bin/env python3
"""
AutoGluon + DeepSets 集成预测脚本（最终版）
- 训练数据使用缓存
- 测试数据不使用缓存
- 保存DeepSets贡献度
- 保存各子模型预测
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import warnings
import gc
import pickle
import hashlib
import json
from pathlib import Path
import time
from tqdm import tqdm
warnings.filterwarnings('ignore')

import torch
from autogluon.tabular import TabularPredictor, TabularDataset

try:
    from dist_train_cache import set_seed
    print("✅ Successfully imported from dist_train_cache.py", flush=True)
except ImportError as e:
    print(f"❌ Failed to import from dist_train_cache.py: {e}", flush=True)
    sys.exit(1)

try:
    from pseudo_redo import BenchmarkItem
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

from ag_extra_models import DeepSetsTabularModelAttn

# ===============================
# 配置参数
# ===============================

class Config:
    """配置"""
    DATA_DIR = "/personal/ImmAge"
    DONOR_FILES_DIR = "donor_files"
    OUTPUT_DIR = "./ensemble_predictions"
    
    TRAIN_METADATA_FILE = "donor_metadata.xlsx"
    TEST_METADATA_FILE = "test_metadata.csv"
    MARKER_FILE = "marker_genes.csv"
    
    AUTOGLUON_MODEL_PATH = "./AutogluonModels/deepsets_integration"
    
    CACHE_DIR = "./donor_cache_ag_marker"
    CACHE_VERSION = "v1.1"
    
    MAX_CELLS_PER_DONOR_TRAIN = 1000
    MAX_CELLS_PER_DONOR_TEST = 1000
    USE_RAW = True
    USE_MARKER = True
    EPOCH = 60
    
    TRAIN_HYPERPARAMETERS = {
        'hidden_dim': 10,
        'dropout': 0.2,
        'lr': 0.001,
        'batch_size': 8,
        'epochs': EPOCH
    }
    
    # 🔥 贡献度分析配置
    COMPUTE_CONTRIBUTIONS = False
    CELL_CONTRIB_METHOD = 'integrated_gradient'
    GENE_CONTRIB_METHOD = 'integrated_gradient'
    
    RANDOM_SEEDS = [42]
    MAX_DONORS_PER_DATASET = None

config = Config()
set_seed(42)

MAX_CELLS_PER_DONOR_TRAIN = config.MAX_CELLS_PER_DONOR_TRAIN
MAX_CELLS_PER_DONOR_TEST = config.MAX_CELLS_PER_DONOR_TEST
CACHE_DIR = config.CACHE_DIR
CACHE_VERSION = config.CACHE_VERSION
USE_RAW = config.USE_RAW

# ===============================
# 缓存系统（训练数据用）
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
            'max_cells_per_donor': MAX_CELLS_PER_DONOR_TRAIN,
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
        print(f"💾 Saving {data_type} cache: {cache_path.name}", flush=True)
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
            print(f"📂 Loading {data_type} cache: {cache_path.name}", flush=True)
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

cache_manager = DataCache()

# ===============================
# 数据加载函数（带缓存，训练用）
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
    print(f"  📏 Max cells per donor: {MAX_CELLS_PER_DONOR_TRAIN}", flush=True)
    
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
            
            if cells.shape[0] > MAX_CELLS_PER_DONOR_TRAIN:
                idx = np.random.choice(cells.shape[0], MAX_CELLS_PER_DONOR_TRAIN, replace=False)
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

def prepare_data_cached(use_marker=True):
    """缓存优化的数据准备函数"""
    print("📥 Loading data with intelligent caching...", flush=True)
    print(f"🔧 MAX_CELLS_PER_DONOR = {MAX_CELLS_PER_DONOR_TRAIN}", flush=True)
    print(f"📦 Cache directory: {cache_manager.cache_dir}", flush=True)
    
    original_dir = os.getcwd()
    os.chdir(config.DATA_DIR)
    
    try:
        donor_info = pd.read_excel(config.TRAIN_METADATA_FILE, index_col=0)
        if use_marker:
            df_marker = pd.read_csv(config.MARKER_FILE, index_col=0)
        donor_files_dir = config.DONOR_FILES_DIR
        
        print(f"Donor info: {donor_info.shape}", flush=True)
        
        if not os.path.exists(donor_files_dir):
            raise FileNotFoundError(f"Donor files directory not found: {donor_files_dir}")
        
        available_donors = get_available_donors_cached(donor_files_dir, donor_info)
        
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
        
        age_data = {}
        for dataset_name, pseudo_df in pseudobulk_data.items():
            age_data[dataset_name] = donor_info.loc[pseudo_df.index, "age"]
        
        gc.collect()
        
        print("\n🔬 Loading single cell data with caching...", flush=True)
        train_donors_subset = train_data.index.tolist()
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
            'test_max_cells': test_max_cells,
            'all_pseudobulk': pseudobulk_data,
            'donor_info': donor_info
        }
    
    finally:
        os.chdir(original_dir)

# ===============================
# 训练AutoGluon
# ===============================

def train_autogluon_with_deepsets(data):
    """训练AutoGluon with DeepSets"""
    print("\n🤖 Training AutoGluon with DeepSets...", flush=True)
    
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
    
    train_data_with_age = pd.concat([data['train_data'], data['train_age']], axis=1)
    print(f"Training data: {train_data_with_age.shape}", flush=True)
    
    train_dataset = TabularDataset(train_data_with_age)
    
    print("🚀 Step 1: Training standard AutoGluon models...", flush=True)

    predictor = TabularPredictor(
        label="age",
        problem_type="regression", 
        path=config.AUTOGLUON_MODEL_PATH,
        verbosity=2,
    )
    
    predictor.fit(
        train_dataset,
        time_limit=600,
        included_model_types=['FASTAI', 'NN_TORCH']
    )
    
    print("✅ Standard models training completed!", flush=True)
    
    print("\n🚀 Step 2: Adding custom DeepSets model...", flush=True)

    hyperparameters = {
        'NN_TORCH': {},
        'FASTAI': {},
        DeepSetsTabularModelAttn: config.TRAIN_HYPERPARAMETERS
    }
    
    try:
        predictor.fit_extra(hyperparameters=hyperparameters)
        
        print("✅ Custom DeepSets model added successfully!", flush=True)
        
        try:
            leaderboard_final = predictor.leaderboard(silent=True)
            print("📊 Final Model Leaderboard (With DeepSets):", flush=True)
            print(leaderboard_final.head(10), flush=True)
        except Exception as e:
            print(f"⚠️ Could not display final leaderboard: {e}", flush=True)
        
        return predictor
        
    except Exception as e:
        print(f"❌ Custom model integration failed: {e}", flush=True)
        print("🔄 Returning standard models only...", flush=True)
        return predictor

# ===============================
# 🔥 测试数据准备（不使用缓存）
# ===============================

def load_test_metadata(metadata_file: str) -> Tuple[pd.DataFrame, bool]:
    """加载测试metadata"""
    print(f"\n📋 Loading test metadata: {metadata_file}", flush=True)
    
    metadata_path = os.path.join(config.DATA_DIR, metadata_file)
    
    if metadata_file.endswith('.csv'):
        metadata = pd.read_csv(metadata_path, index_col=0)
    elif metadata_file.endswith('.xlsx'):
        metadata = pd.read_excel(metadata_path, index_col=0)
    else:
        raise ValueError(f"Unsupported format: {metadata_file}")
    
    if 'dataset' not in metadata.columns:
        raise ValueError("Metadata missing required column: 'dataset'")
    
    has_age = 'age' in metadata.columns
    
    print(f"   ✅ Loaded {len(metadata)} test donors", flush=True)
    print(f"   Datasets: {metadata['dataset'].unique().tolist()}", flush=True)
    print(f"   Has age labels: {'Yes' if has_age else 'No'}", flush=True)
    
    return metadata, has_age

def sample_cells_from_donor_with_seed(
    donor_file: str,
    donor_id: str,
    seed: int,
    max_cells: int,
    gene_names: Optional[List[str]] = None,
    use_raw: bool = True
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """从donor文件采样细胞（指定seed）"""
    try:
        adata = sc.read_h5ad(donor_file)
        
        if gene_names is not None:
            common_genes = [g for g in gene_names if g in adata.var_names]
            adata = adata[:, common_genes]
        
        if use_raw and 'counts' in adata.layers:
            expr_matrix = adata.layers['counts'].toarray().astype(np.float32)
        else:
            expr_matrix = adata.X.toarray().astype(np.float32)
        
        n_cells = expr_matrix.shape[0]
        
        # 🔥 关键：使用seed进行采样
        np.random.seed(seed)
        if n_cells > max_cells:
            selected_idx = np.random.choice(n_cells, max_cells, replace=False)
            selected_idx = np.sort(selected_idx)
        else:
            selected_idx = np.arange(n_cells)
        
        cell_data = expr_matrix[selected_idx]
        
        if hasattr(adata.obs.index, 'tolist'):
            original_indices = adata.obs.index[selected_idx].tolist()
        else:
            original_indices = [str(i) for i in selected_idx]
        
        cell_metadata = adata.obs.iloc[selected_idx].copy()
        cell_metadata['sampled_position'] = range(len(selected_idx))
        cell_metadata['original_index'] = original_indices
        
        del adata, expr_matrix
        gc.collect()
        
        return cell_data, original_indices, cell_metadata
        
    except Exception as e:
        print(f"   ⚠️ Error sampling donor {donor_id}: {e}", flush=True)
        raise

def prepare_test_data_no_cache_with_seeds(
    test_metadata: pd.DataFrame,
    seeds: List[int],
    gene_names: Optional[List[str]] = None
) -> Dict:
    """
    🔥 准备测试数据（不使用缓存，支持多个seeds）
    
    为每个seed分别采样，确保结果不同
    """
    print(f"\n📊 Preparing test data for {len(seeds)} seeds (NO CACHE)...", flush=True)
    
    original_dir = os.getcwd()
    os.chdir(config.DATA_DIR)
    
    try:
        donor_files_dir = config.DONOR_FILES_DIR
        
        # 按dataset分组
        test_datasets_donors = {}
        for dataset in test_metadata['dataset'].unique():
            donors = test_metadata[test_metadata['dataset'] == dataset].index.tolist()
            if config.MAX_DONORS_PER_DATASET:
                donors = donors[:config.MAX_DONORS_PER_DATASET]
            test_datasets_donors[dataset] = donors
        
        # 🔥 为每个seed准备数据
        results_by_seed = {}
        
        for seed in seeds:
            print(f"\n🎲 Processing seed {seed}...", flush=True)
            
            pseudobulk_data = {}
            cell_data_dict = {}
            
            for dataset_name, donors in test_datasets_donors.items():
                if not donors:
                    continue
                
                print(f"  📊 {dataset_name} ({len(donors)} donors) - seed {seed}...", flush=True)
                
                # Pseudobulk: 不依赖seed，只计算一次即可
                # 但为了代码统一，每个seed都计算
                pseudo_list = []
                for donor in donors:
                    donor_file = os.path.join(donor_files_dir, f'donor_{donor}.h5ad')
                    if not os.path.exists(donor_file):
                        continue
                    
                    try:
                        adata = sc.read_h5ad(donor_file)
                        if gene_names is not None:
                            adata = adata[:, gene_names]
                        
                        if USE_RAW and 'counts' in adata.layers:
                            expr = adata.layers['counts'].mean(axis=0).A1
                        else:
                            expr = adata.X.mean(axis=0).A1
                        
                        pseudo_list.append(expr)
                        del adata
                        gc.collect()
                    except Exception as e:
                        print(f"    ⚠️ Error {donor}: {e}", flush=True)
                        continue
                
                if pseudo_list:
                    pseudobulk_data[dataset_name] = pd.DataFrame(
                        data=np.array(pseudo_list),
                        index=donors[:len(pseudo_list)],
                        columns=gene_names if gene_names else []
                    )
                
                # 🔥 单细胞: 使用不同的seed进行采样
                cell_data = {}
                cell_indices = {}
                cell_metadata = {}
                max_cells = 0
                
                for donor in donors:
                    donor_file = os.path.join(donor_files_dir, f'donor_{donor}.h5ad')
                    if not os.path.exists(donor_file):
                        continue
                    
                    try:
                        cells, indices, metadata = sample_cells_from_donor_with_seed(
                            donor_file, donor, seed, MAX_CELLS_PER_DONOR_TEST, # Tid
                            gene_names, USE_RAW
                        )
                        
                        cell_data[donor] = cells
                        cell_indices[donor] = indices
                        cell_metadata[donor] = metadata
                        max_cells = max(max_cells, cells.shape[0])
                        
                    except Exception as e:
                        print(f"    ⚠️ Error {donor}: {e}", flush=True)
                        continue
                
                cell_data_dict[dataset_name] = {
                    'cell_data': cell_data,
                    'cell_indices': cell_indices,
                    'cell_metadata': cell_metadata,
                    'max_cells': max_cells
                }
                
                print(f"    ✅ Loaded {len(cell_data)} donors, max_cells={max_cells}", flush=True)
            
            results_by_seed[seed] = {
                'pseudobulk_data': pseudobulk_data,
                'cell_data_dict': cell_data_dict,
                'test_datasets_donors': test_datasets_donors
            }
            
            gc.collect()
        
        return results_by_seed
    
    finally:
        os.chdir(original_dir)

# ===============================
# 🔥 DeepSets贡献度计算
# ===============================

def compute_deepsets_contributions(
    predictor: TabularPredictor,
    test_cell_data: Dict,
    gene_names: List[str],
    config: Config
) -> Optional[Dict]:
    """计算DeepSets模型的贡献度"""
    
    if not config.COMPUTE_CONTRIBUTIONS:
        print("\n   ℹ️ Contribution analysis disabled", flush=True)
        return None
    
    print("\n🔬 Computing DeepSets contributions...", flush=True)
    
    try:
        # 查找DeepSets模型
        model_names = predictor.model_names()
        deepsets_name = None
        for name in model_names:
            if 'DeepSets' in name:
                deepsets_name = name
                break
        
        if deepsets_name is None:
            print("   ⚠️ No DeepSets model found", flush=True)
            return None
        
        print(f"   🎯 Found DeepSets model: {deepsets_name}", flush=True)
        
        # 加载模型
        deepsets_model_wrapper = predictor._trainer.load_model(deepsets_name)
        deepsets_model = deepsets_model_wrapper.model
        deepsets_scaler = deepsets_model_wrapper.scaler
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 计算贡献度
        all_cell_contrib = []
        all_gene_contrib = []
        
        donor_ids = list(test_cell_data.keys())
        
        for donor_id in tqdm(donor_ids, desc="Computing contributions"):
            if donor_id not in test_cell_data:
                continue
            
            cells = test_cell_data[donor_id]
            n_cells = cells.shape[0]
            
            # 细胞贡献度
            cells_scaled = deepsets_scaler.transform(cells).astype(np.float32)
            cells_tensor = torch.from_numpy(cells_scaled).unsqueeze(0).to(device)
            mask = torch.ones(1, n_cells, device=device, dtype=torch.bool)
            
            if config.CELL_CONTRIB_METHOD in ['attention', 'activation']:
                cell_result = deepsets_model.get_cell_contributions_attn(
                    cells_tensor, mask=mask, method=config.CELL_CONTRIB_METHOD
                )
            else:
                cell_result = deepsets_model.get_cell_contributions(
                    cells_tensor, mask=mask, method=config.CELL_CONTRIB_METHOD, 
                    target='H', normalize=True
                )
            
            # 基因贡献度
            gene_result = deepsets_model.get_gene_contributions(
                cells_tensor, mask=mask, method=config.GENE_CONTRIB_METHOD, per_cell=True
            )
            
            # 保存细胞贡献度
            for cell_idx, contrib in enumerate(cell_result['cell_contributions'][0]):
                all_cell_contrib.append({
                    'donor_id': donor_id,
                    'cell_position': cell_idx,
                    'contribution_score': float(contrib),
                    'predicted_age': float(cell_result['age_pred'][0])
                })
            
            # 保存基因贡献度
            gene_contribs = np.asarray(gene_result['gene_contributions'][0]).reshape(-1)
            for gene_idx, (gene_name, contrib) in enumerate(zip(gene_names, gene_contribs)):
                all_gene_contrib.append({
                    'donor_id': donor_id,
                    'gene_name': gene_name,
                    'contribution_score': float(contrib),
                    'predicted_age': float(gene_result['age_pred'][0])
                })
        
        df_cell_contrib = pd.DataFrame(all_cell_contrib)
        df_gene_contrib = pd.DataFrame(all_gene_contrib)
        
        print(f"   ✅ Contributions computed: {len(df_cell_contrib)} cells, {len(df_gene_contrib)} genes", flush=True)
        
        return {
            'cell_contributions': df_cell_contrib,
            'gene_contributions': df_gene_contrib
        }
        
    except Exception as e:
        print(f"   ❌ Contribution computation failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

# ===============================
# 🔥 预测函数（获取所有模型的预测）
# ===============================

def predict_with_all_models(
    predictor: TabularPredictor,
    test_pseudobulk: pd.DataFrame,
    test_cell_data: Dict,
    test_metadata: pd.DataFrame,
    dataset_name: str,
    seed: int,
    has_age: bool
) -> Dict:
    """使用所有模型进行预测"""
    print(f"  🔮 Predicting {dataset_name} (seed={seed})...", flush=True)
    
    # 更新共享单细胞数据
    DeepSetsTabularModelAttn._shared_test_cell_data = test_cell_data
    all_shared = DeepSetsTabularModelAttn._shared_train_cell_data.copy()
    all_shared.update(test_cell_data)
    DeepSetsTabularModelAttn._shared_cell_data = all_shared
    
    try:
        # 🔥 获取集成预测
        ensemble_predictions = predictor.predict(test_pseudobulk)
        
        # 🔥 获取每个模型的预测
        model_predictions = {}
        for model_name in predictor.model_names():
            try:
                model_preds = predictor.predict(test_pseudobulk, model=model_name)
                model_predictions[model_name] = model_preds.values
            except:
                pass
        
        result = {
            'donor_ids': test_pseudobulk.index.tolist(),
            'ensemble_predictions': ensemble_predictions.values,
            'model_predictions': model_predictions,
            'dataset': dataset_name,
            'seed': seed
        }
        
        if has_age:
            true_ages = test_metadata.loc[test_pseudobulk.index, 'age']
            result['true_ages'] = true_ages.values
            
            benchmark = BenchmarkItem(ensemble_predictions, true_ages)
            result['pearson_r'] = benchmark.pcc
            result['mae'] = benchmark.mae
            
            print(f"     ✅ Pearson r: {benchmark.pcc:.4f}, MAE: {benchmark.mae:.3f}", flush=True)
        else:
            result['true_ages'] = None
            print(f"     ✅ Predicted {len(ensemble_predictions)} samples", flush=True)
        
        return result
        
    except Exception as e:
        print(f"     ❌ Prediction failed: {e}", flush=True)
        raise

# ===============================
# 🔥 结果保存（包含详细信息）
# ===============================

def save_detailed_predictions(
    all_predictions: Dict,
    all_contributions: Optional[Dict],
    output_dir: str,
    has_age: bool
):
    """保存详细预测结果"""
    print(f"\n💾 Saving detailed results to {output_dir}...", flush=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存集成预测 + 各模型预测
    all_records = []
    for dataset, seeds_results in all_predictions.items():
        for seed, result in seeds_results.items():
            for i, donor_id in enumerate(result['donor_ids']):
                record = {
                    'seed': seed,
                    'dataset': dataset,
                    'donor_id': donor_id,
                    'ensemble_prediction': result['ensemble_predictions'][i],
                }
                
                # 添加各模型预测
                for model_name, preds in result['model_predictions'].items():
                    safe_model_name = model_name.replace('/', '_').replace(' ', '_')
                    record[f'pred_{safe_model_name}'] = preds[i]
                
                if has_age and result['true_ages'] is not None:
                    record['true_age'] = result['true_ages'][i]
                    record['error'] = result['ensemble_predictions'][i] - result['true_ages'][i]
                    record['abs_error'] = abs(record['error'])
                
                all_records.append(record)
    
    df_predictions = pd.DataFrame(all_records)
    pred_path = os.path.join(output_dir, 'predictions_detailed.csv')
    df_predictions.to_csv(pred_path, index=False)
    print(f"   ✅ Detailed predictions: {pred_path} ({len(df_predictions)} records)", flush=True)
    
    # 2. 保存贡献度
    if all_contributions:
        contrib_dir = os.path.join(output_dir, 'contributions')
        os.makedirs(contrib_dir, exist_ok=True)
        
        for dataset, contrib in all_contributions.items():
            if contrib:
                cell_path = os.path.join(contrib_dir, f'cell_contributions_{dataset}.csv')
                contrib['cell_contributions'].to_csv(cell_path, index=False)
                
                gene_path = os.path.join(contrib_dir, f'gene_contributions_{dataset}.csv')
                contrib['gene_contributions'].to_csv(gene_path, index=False)
        
        print(f"   ✅ Contributions: {contrib_dir}", flush=True)
    
    # 3. 统计摘要
    if has_age:
        summary_stats = []
        for dataset in df_predictions['dataset'].unique():
            for seed in df_predictions['seed'].unique():
                subset = df_predictions[
                    (df_predictions['dataset'] == dataset) & 
                    (df_predictions['seed'] == seed)
                ]
                
                if len(subset) > 0 and 'true_age' in subset.columns:
                    from scipy.stats import pearsonr
                    r, _ = pearsonr(subset['true_age'], subset['ensemble_prediction'])
                    
                    summary_stats.append({
                        'dataset': dataset,
                        'seed': seed,
                        'n_donors': len(subset),
                        'mae': subset['abs_error'].mean(),
                        'rmse': np.sqrt((subset['error'] ** 2).mean()),
                        'pearson_r': r
                    })
        
        if summary_stats:
            df_summary = pd.DataFrame(summary_stats)
            summary_path = os.path.join(output_dir, 'summary_statistics.csv')
            df_summary.to_csv(summary_path, index=False)
            print(f"   ✅ Summary: {summary_path}", flush=True)
            
            print("\n📊 Performance Summary:", flush=True)
            print("="*70)
            print(df_summary.groupby('dataset')[['n_donors', 'mae', 'pearson_r']].mean().to_string())
            print("="*70)
    
    config_info = {
        'test_metadata': config.TEST_METADATA_FILE,
        'model_path': config.AUTOGLUON_MODEL_PATH,
        'has_age_labels': has_age,
        'compute_contributions': config.COMPUTE_CONTRIBUTIONS,
    }
    
    config_path = os.path.join(output_dir, 'prediction_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_info, f, indent=2)
    print(f"   ✅ Config: {config_path}", flush=True)

# ===============================
# 主流程
# ===============================

def main():
    """主函数"""
    print("="*80)
    print("🎯 AutoGluon + DeepSets Prediction (Final Version)")
    print("="*80)
    
    print(f"\n⚙️ Configuration:")
    print(f"   Model Path: {config.AUTOGLUON_MODEL_PATH}")
    print(f"   Test Metadata: {config.TEST_METADATA_FILE}")
    print(f"   Cache for Training: {config.CACHE_DIR}")
    print(f"   Test Data: NO CACHE")
    print(f"   Random Seeds: {config.RANDOM_SEEDS}")
    print(f"   Contributions: {config.COMPUTE_CONTRIBUTIONS}")
    
    original_dir = os.getcwd()
    
    try:
        start_time = time.time()
        
        # Step 1: 准备训练数据（带缓存）
        print("\n" + "="*80)
        print("Step 1: Preparing Training Data (With Cache)")
        print("="*80)
        
        data = prepare_data_cached(use_marker=config.USE_MARKER)
        
        # Step 2: 训练模型
        print("\n" + "="*80)
        print("Step 2: Training AutoGluon Model")
        print("="*80)
        
        predictor = train_autogluon_with_deepsets(data)
        
        model_time = time.time()
        print(f"\n⏱️ Model ready: {model_time - start_time:.1f}s", flush=True)
        
        # Step 3: 加载测试metadata
        print("\n" + "="*80)
        print("Step 3: Loading Test Metadata")
        print("="*80)
        
        test_metadata, has_age = load_test_metadata(config.TEST_METADATA_FILE)
        
        # Step 4: 准备测试数据（不使用缓存，支持多个seeds）
        print("\n" + "="*80)
        print("Step 4: Preparing Test Data for All Seeds (NO CACHE)")
        print("="*80)
        
        gene_names = None
        if config.USE_MARKER:
            os.chdir(config.DATA_DIR)
            df_marker = pd.read_csv(config.MARKER_FILE, index_col=0)
            gene_names = df_marker.index.tolist()
            os.chdir(original_dir)
        
        # 🔥 修改：为所有seeds准备数据
        test_data_by_seed = prepare_test_data_no_cache_with_seeds(
            test_metadata, config.RANDOM_SEEDS, gene_names
        )
        
        # Step 5: 进行预测
        print("\n" + "="*80)
        print("Step 5: Running Predictions (All Models, All Seeds)")
        print("="*80)
        
        all_predictions = {}
        all_contributions = {}
        
        # 🔥 修改：按dataset组织，每个seed使用不同的数据
        # 首先获取所有datasets
        first_seed = config.RANDOM_SEEDS[0]
        all_datasets = list(test_data_by_seed[first_seed]['test_datasets_donors'].keys())

        # for model_name in predictor.model_names():
        #     if 'DeepSets' in model_name:
        #         deepsets_name = model_name
        #         break
        # breakpoint()
        # predictor._trainer.load_model(deepsets_name).max_cells = MAX_CELLS_PER_DONOR_TEST
        
        for dataset_name in all_datasets:
            print(f"\n📊 Dataset: {dataset_name}")
            print("-"*70)
            
            dataset_predictions = {}
            
            # 🔥 关键修改：每个seed使用自己的数据
            for seed in config.RANDOM_SEEDS:
                print(f"\n🎲 Seed {seed}:")
                
                test_data = test_data_by_seed[seed]
                
                if dataset_name not in test_data['pseudobulk_data']:
                    print(f"  ⚠️ No data for {dataset_name} in seed {seed}")
                    continue
                
                pred_result = predict_with_all_models(
                    predictor,
                    test_data['pseudobulk_data'][dataset_name],
                    test_data['cell_data_dict'][dataset_name]['cell_data'],
                    test_metadata,
                    dataset_name,
                    seed,
                    has_age
                )
                
                dataset_predictions[seed] = pred_result
            
            all_predictions[dataset_name] = dataset_predictions
            
            # 🔥 计算贡献度（使用第一个seed的数据）
            if config.COMPUTE_CONTRIBUTIONS:
                print(f"\n🔬 Computing contributions for {dataset_name}...")
                first_seed = config.RANDOM_SEEDS[0]
                contrib_result = compute_deepsets_contributions(
                    predictor,
                    test_data_by_seed[first_seed]['cell_data_dict'][dataset_name]['cell_data'],
                    gene_names if gene_names else [],
                    config
                )
                if contrib_result:
                    all_contributions[dataset_name] = contrib_result
        
        # Step 6: 保存结果
        print("\n" + "="*80)
        print("Step 6: Saving Detailed Results")
        print("="*80)
        
        save_detailed_predictions(all_predictions, all_contributions, config.OUTPUT_DIR, has_age)
        
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("🎉 Prediction Completed!")
        print("="*80)
        print(f"⏱️ Total: {total_time:.1f}s")
        print(f"📁 Results: {config.OUTPUT_DIR}")
        print(f"🎲 Seeds: {config.RANDOM_SEEDS}")
        print(f"   - predictions_detailed.csv (ensemble + all models)")
        if config.COMPUTE_CONTRIBUTIONS:
            print(f"   - contributions/ (DeepSets contributions)")
        
        # 🔥 验证seeds是否生效
        if len(config.RANDOM_SEEDS) > 1:
            print("\n🔍 Verifying seed effect...")
            for dataset in all_predictions.keys():
                if len(all_predictions[dataset]) >= 2:
                    seeds_list = list(all_predictions[dataset].keys())
                    pred1 = all_predictions[dataset][seeds_list[0]]['ensemble_predictions']
                    pred2 = all_predictions[dataset][seeds_list[1]]['ensemble_predictions']
                    
                    diff = np.abs(pred1 - pred2).mean()
                    print(f"   {dataset}: Mean prediction difference between seeds = {diff:.6f}")
                    if diff > 0:
                        print(f"      ✅ Seeds are working (predictions differ)")
                    else:
                        print(f"      ⚠️ Seeds might not be working (predictions identical)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    main()
