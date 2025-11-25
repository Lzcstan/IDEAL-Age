#!/usr/bin/env python3
"""
DeepSets模型权重测试与贡献度分析脚本（优化版 - 修复gene contributions重复问题）
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
import gc
import json
import hashlib
from pathlib import Path
import time
from tqdm import tqdm
warnings.filterwarnings('ignore')

import torch
from autogluon.tabular import TabularPredictor

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

set_seed(42)

# ===============================
# 配置参数
# ===============================

class TestConfig:
    """测试配置"""
    # 路径配置
    DATA_DIR = "/personal/ImmAge"
    DONOR_FILES_DIR = "donor_files"
    OUTPUT_DIR = "./deepsetsattn_test_results"
    
    # 权重缓存配置
    WEIGHTS_CACHE_DIR = "./results_debug/deepsets_weights"
    
    # 🔥 模型超参数配置（用于定位权重文件）
    MODEL_HYPERPARAMETERS = {
        'hidden_dim': 10,
        'dropout': 0.2,
        'lr': 0.001,
        'batch_size': 8,
        'epochs': 60,
    }
    
    # 数据配置（需要与训练时一致）
    MAX_CELLS_PER_DONOR_TEST = 10000
    MAX_CELLS_PER_DONOR_TRAIN = 1000
    USE_RAW = True
    USE_MARKER = True
    IS_SCALED = not USE_RAW
    
    # 采样配置
    RANDOM_SEEDS = [42]
    
    # 贡献度计算配置
    CELL_CONTRIB_METHOD = 'integrated_gradient'
    GENE_CONTRIB_METHOD = 'integrated_gradient'
    
    # 测试的数据集
    TEST_DATASETS = ['AIDA_test', 'eQTL_test', 'HCA', 'siAge']
    
    MAX_DONORS_PER_DATASET = None

# ===============================
# 权重缓存管理
# ===============================

class WeightsCacheManager:
    """DeepSets权重缓存管理器"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"📦 Weights cache manager initialized: {self.cache_dir}", flush=True)
    
    def get_cache_key(self, hyperparameters: Dict, n_donors: int, n_genes: int, is_scaled: bool = True) -> str:
        """根据超参数和数据特征生成缓存key"""
        cache_params = {
            'hidden_dim': hyperparameters.get('hidden_dim', 8),
            'dropout': hyperparameters.get('dropout', 0.2),
            'lr': hyperparameters.get('lr', 1e-3),
            'batch_size': hyperparameters.get('batch_size', 8),
            'epochs': hyperparameters.get('epochs', 30),
            'max_cells': hyperparameters.get('max_cells', 1000),
            'model_type': 'DeepSetsTabularModelAttn',
            'n_donors': n_donors,
            'n_genes': n_genes,
            # 'is_scaled': is_scaled,
        }

        if is_scaled:
            cache_params['is_scaled'] = is_scaled
        
        cache_str = json.dumps(cache_params, sort_keys=True)
        cache_key = hashlib.md5(cache_str.encode()).hexdigest()
        
        return cache_key, cache_params
    
    def get_cache_path(self, cache_key: str) -> Path:
        """获取权重缓存文件路径"""
        return self.cache_dir / f"deepsets_weights_{cache_key}.pth"
    
    def load_weights(self, cache_key: str, device: str = 'cuda') -> Optional[Dict]:
        """加载缓存的权重"""
        cache_path = self.get_cache_path(cache_key)
        
        if not cache_path.exists():
            print(f"   ⚠️ Weights cache not found: {cache_path}", flush=True)
            return None
        
        try:
            print(f"   📂 Loading weights from cache: {cache_path}", flush=True)
            checkpoint = torch.load(cache_path, map_location=device, weights_only=False)
            
            cached_params = checkpoint.get('cache_params', {})
            timestamp = checkpoint.get('timestamp', 0)
            
            print(f"   ✅ Loaded cached weights (saved at {time.ctime(timestamp)})", flush=True)
            print(f"   📋 Cached params:", flush=True)
            for key, value in cached_params.items():
                print(f"      {key}: {value}", flush=True)
            
            return checkpoint
            
        except Exception as e:
            print(f"   ❌ Failed to load weights cache: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None
    
    def list_available_weights(self) -> List[Dict]:
        """列出所有可用的权重文件"""
        weight_files = list(self.cache_dir.glob("deepsets_weights_*.pth"))
        
        available_weights = []
        for weight_file in weight_files:
            try:
                checkpoint = torch.load(weight_file, map_location='cpu', weights_only=False)
                cache_params = checkpoint.get('cache_params', {})
                timestamp = checkpoint.get('timestamp', 0)
                
                available_weights.append({
                    'file': weight_file.name,
                    'cache_key': weight_file.stem.replace('deepsets_weights_', ''),
                    'params': cache_params,
                    'timestamp': timestamp,
                    'size_mb': weight_file.stat().st_size / 1024 / 1024
                })
            except Exception as e:
                print(f"   ⚠️ Failed to read {weight_file}: {e}", flush=True)
                continue
        
        return available_weights

# ===============================
# DeepSets模型加载器
# ===============================

def load_deepsets_model_from_cache(
    checkpoint: Dict,
    n_genes: int,
    device: str = 'cuda'
):
    """从checkpoint加载DeepSets模型"""
    from ag_extra_models import DeepSetsAgePredictorAttn
    from sklearn.preprocessing import StandardScaler
    
    cache_params = checkpoint['cache_params']
    hidden_dim = 2 ** cache_params.get('hidden_dim', 8)
    dropout = cache_params.get('dropout', 0.2)
    num_classes = 10
    
    print(f"   🧠 Creating DeepSets model:", flush=True)
    print(f"      input_dim={n_genes}, hidden_dim={hidden_dim}, dropout={dropout}", flush=True)
    
    model = DeepSetsAgePredictorAttn(
        input_dim=n_genes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_classes=num_classes,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model loaded: {total_params:,} parameters", flush=True)
    
    scaler = StandardScaler()
    scaler.mean_ = checkpoint['scaler_mean']
    scaler.scale_ = checkpoint['scaler_scale']
    scaler.n_features_in_ = len(checkpoint['scaler_mean'])
    
    max_cells = checkpoint['max_cells']
    
    return model, scaler, max_cells

# ===============================
# 数据加载函数
# ===============================

def load_donor_metadata(data_dir: str) -> pd.DataFrame:
    """加载donor metadata"""
    print("📋 Loading donor metadata...", flush=True)
    donor_info = pd.read_excel(os.path.join(data_dir, "donor_metadata.xlsx"), index_col=0)
    print(f"   Loaded {len(donor_info)} donors", flush=True)
    return donor_info

def load_marker_genes(data_dir: str) -> Optional[pd.DataFrame]:
    """加载marker基因"""
    marker_path = os.path.join(data_dir, "marker_genes.csv")
    if os.path.exists(marker_path):
        print("🧬 Loading marker genes...", flush=True)
        df_marker = pd.read_csv(marker_path, index_col=0)
        print(f"   Loaded {len(df_marker)} marker genes", flush=True)
        return df_marker
    return None

def get_test_donors(donor_info: pd.DataFrame, donor_files_dir: str, config: TestConfig) -> Dict[str, List[str]]:
    """获取测试集的donor列表"""
    print("📊 Preparing test donor lists...", flush=True)
    
    available_files = [f for f in os.listdir(donor_files_dir) 
                      if f.startswith('donor_') and f.endswith('.h5ad')]
    available_donors = [f.replace('donor_', '').replace('.h5ad', '') for f in available_files]
    
    test_donors = {
        'AIDA_test': [d for d in donor_info.loc[
            (donor_info["dataset"] == "AIDA") & ~donor_info["is_train"]
        ].index.tolist() if d in available_donors],
        
        'eQTL_test': [d for d in donor_info.loc[
            (donor_info["dataset"] == "eQTL") & ~donor_info["is_train"]
        ].index.tolist() if d in available_donors],
        
        'HCA': [d for d in donor_info[donor_info["dataset"] == "HCA"].index.tolist() 
               if d in available_donors],
        
        'siAge': [d for d in donor_info[donor_info["dataset"] == "siAge"].index.tolist() 
                 if d in available_donors]
    }
    
    if config.MAX_DONORS_PER_DATASET is not None:
        for dataset in test_donors:
            if len(test_donors[dataset]) > config.MAX_DONORS_PER_DATASET:
                test_donors[dataset] = test_donors[dataset][:config.MAX_DONORS_PER_DATASET]
    
    for dataset, donors in test_donors.items():
        print(f"   {dataset}: {len(donors)} donors", flush=True)
    
    return test_donors

# ===============================
# 单细胞数据采样函数
# ===============================

def sample_cells_from_donor(
    donor_file: str,
    donor_id: str,
    seed: int,
    max_cells: int,
    gene_names: Optional[List[str]] = None,
    use_raw: bool = True
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """从单个donor的h5ad文件中采样细胞"""
    try:
        adata = sc.read_h5ad(donor_file)
        
        if gene_names is not None:
            adata = adata[:, gene_names]
        
        if use_raw and 'counts' in adata.layers:
            expr_matrix = adata.layers['counts'].toarray().astype(np.float32)
        else:
            expr_matrix = adata.X.toarray().astype(np.float32)
        
        n_cells = expr_matrix.shape[0]
        
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

def load_test_data_with_seeds(
    donor_files_dir: str,
    test_donors: Dict[str, List[str]],
    seeds: List[int],
    max_cells: int,
    gene_names: Optional[List[str]] = None,
    use_raw: bool = True
) -> Dict:
    """为每个随机种子加载测试数据"""
    print(f"\n🔬 Loading test data for {len(seeds)} random seeds...", flush=True)
    
    all_donors = set()
    for donors in test_donors.values():
        all_donors.update(donors)
    all_donors = sorted(list(all_donors))
    
    print(f"   Total unique donors to process: {len(all_donors)}", flush=True)
    
    results = {}
    
    for seed in seeds:
        print(f"\n📌 Processing seed {seed}...", flush=True)
        
        seed_data = {
            'cell_data': {},
            'cell_indices': {},
            'cell_metadata': {},
            'max_cells': 0
        }
        
        for donor_id in tqdm(all_donors, desc=f"Seed {seed}"):
            donor_file = os.path.join(donor_files_dir, f'donor_{donor_id}.h5ad')
            
            if not os.path.exists(donor_file):
                continue
            
            try:
                cell_data, cell_indices, cell_metadata = sample_cells_from_donor(
                    donor_file, donor_id, seed, max_cells, gene_names, use_raw
                )
                
                seed_data['cell_data'][donor_id] = cell_data
                seed_data['cell_indices'][donor_id] = cell_indices
                seed_data['cell_metadata'][donor_id] = cell_metadata
                seed_data['max_cells'] = max(seed_data['max_cells'], cell_data.shape[0])
                
            except Exception as e:
                print(f"   ❌ Failed to process donor {donor_id}: {e}", flush=True)
                continue
        
        results[seed] = seed_data
        print(f"   ✅ Seed {seed}: {len(seed_data['cell_data'])} donors, max_cells={seed_data['max_cells']}", flush=True)
        
        gc.collect()
    
    return results

# ===============================
# 贡献度计算函数（优化版）
# ===============================

def compute_contributions_with_model(
    model,
    scaler,
    seed: int,
    seed_data: Dict,
    test_donors: Dict[str, List[str]],
    gene_names: List[str],
    config: TestConfig,
    device: str = 'cuda'
) -> Dict:
    """使用加载的模型计算贡献度"""
    print(f"\n🔬 Computing contributions for seed {seed}...", flush=True)
    
    all_test_donors = []
    donor_to_dataset = {}
    for dataset, donors in test_donors.items():
        for donor in donors:
            if donor in seed_data['cell_data']:
                all_test_donors.append(donor)
                donor_to_dataset[donor] = dataset
    
    print(f"   Analyzing {len(all_test_donors)} donors...", flush=True)
    
    # 1. 计算细胞贡献度
    print("   📊 Computing cell contributions...", flush=True)
    cell_contrib_results = compute_cell_contributions(
        model, scaler, all_test_donors, seed_data['cell_data'], 
        config.CELL_CONTRIB_METHOD, device
    )
    
    # 2. 计算基因和细胞-基因贡献度
    print("   🧬 Computing gene and cell-gene contributions...", flush=True)
    gene_contrib_results = compute_gene_contributions(
        model, scaler, all_test_donors, seed_data['cell_data'],
        gene_names, config.GENE_CONTRIB_METHOD, device
    )
    
    # 转换为DataFrame格式
    print("   📝 Formatting results...", flush=True)
    
    cell_contrib_df = format_cell_contributions(
        cell_contrib_results, seed_data, seed, donor_to_dataset
    )
    
    # 🔥 修复：gene_contributions应该是每个donor的每个基因（不是每个细胞）
    gene_contrib_df = format_gene_contributions(
        gene_contrib_results, seed, donor_to_dataset, gene_names
    )
    
    # cell_gene_contrib_df = format_cell_gene_contributions(
    #     gene_contrib_results, seed_data, seed, donor_to_dataset, gene_names
    # )
    
    return {
        'cell_contributions': cell_contrib_df,
        'gene_contributions': gene_contrib_df,
        # 'cell_gene_contributions': cell_gene_contrib_df
    }

def compute_cell_contributions(
    model, scaler, donor_ids: List[str], cell_data: Dict,
    method: str, device: str
) -> Dict:
    """计算细胞贡献度"""
    results = {}
    
    for donor_id in donor_ids:
        cells = cell_data[donor_id]
        n_cells = cells.shape[0]
        
        cells_scaled = scaler.transform(cells).astype(np.float32)
        cells_tensor = torch.from_numpy(cells_scaled).unsqueeze(0).to(device)
        mask = torch.ones(1, n_cells, device=device, dtype=torch.bool)
        
        if method in ['attention', 'activation']:
            contrib_result = model.get_cell_contributions_attn(
                cells_tensor, mask=mask, method=method
            )
        elif method in ['gradient', 'grad_input', 'integrated_gradient']:
            contrib_result = model.get_cell_contributions(
                cells_tensor, mask=mask, method=method, target='H', normalize=True
            )
        else:
            raise ValueError(f"Unsupported cell contribution method: {method}")
        
        results[donor_id] = {
            'cell_contributions': contrib_result['cell_contributions'][0],
            'age_pred': contrib_result['age_pred'][0],
            'n_cells': n_cells,
        }
    
    return results

def compute_gene_contributions(
    model, scaler, donor_ids: List[str], cell_data: Dict,
    gene_names: List[str], method: str, device: str
) -> Dict:
    """计算基因和细胞-基因贡献度"""
    results = {}
    
    for donor_id in donor_ids:
        cells = cell_data[donor_id]
        n_cells, n_genes = cells.shape
        
        cells_scaled = scaler.transform(cells).astype(np.float32)
        cells_tensor = torch.FloatTensor(cells_scaled).unsqueeze(0).to(device)
        mask = torch.ones(1, n_cells, device=device, dtype=torch.bool)
        
        contrib_result = model.get_gene_contributions(
            cells_tensor, mask=mask, method=method, per_cell=True
        )
        
        # cell_gene_contributions = contrib_result.get('cell_gene_contributions', None)
        # if cell_gene_contributions is not None:
        #     cell_gene_contributions = cell_gene_contributions[0]  # [N, G]
        
        gene_contributions = np.asarray(contrib_result['gene_contributions'][0]).reshape(-1)
        
        results[donor_id] = {
            'gene_contributions': gene_contributions,
            # 'cell_gene_contributions': cell_gene_contributions if cell_gene_contributions is not None else np.empty((0, 0)),
            'age_pred': contrib_result['age_pred'][0],
            'n_cells': n_cells,
            'n_genes': n_genes,
        }
    
    return results

# ===============================
# 结果格式化函数（修复版）
# ===============================

def format_cell_contributions(
    contrib_results: Dict,
    seed_data: Dict,
    seed: int,
    donor_to_dataset: Dict
) -> pd.DataFrame:
    """
    格式化细胞贡献度结果
    
    输出格式：每个donor的每个细胞一行
    Columns: seed, dataset, donor_id, cell_index, cell_position, contribution_score, predicted_age
    """
    records = []
    
    for donor_id, result in contrib_results.items():
        cell_indices = seed_data['cell_indices'][donor_id]
        cell_contribs = result['cell_contributions']
        predicted_age = result['age_pred']
        
        for pos, (cell_idx, contrib) in enumerate(zip(cell_indices, cell_contribs)):
            records.append({
                'seed': seed,
                'dataset': donor_to_dataset[donor_id],
                'donor_id': donor_id,
                'cell_index': cell_idx,
                'cell_position': pos,
                'contribution_score': float(contrib),
                'predicted_age': float(predicted_age)
            })
    
    return pd.DataFrame(records)

def format_gene_contributions(
    contrib_results: Dict,
    seed: int,
    donor_to_dataset: Dict,
    gene_names: List[str]
) -> pd.DataFrame:
    """
    🔥 修复：格式化基因贡献度结果
    
    输出格式：每个donor的每个基因一行（不是每个细胞）
    Columns: seed, dataset, donor_id, gene_name, contribution_score, predicted_age
    
    gene_contributions 应该是对所有细胞聚合后的基因贡献
    """
    records = []
    
    for donor_id, result in contrib_results.items():
        # 🔥 关键修复：gene_contributions已经是聚合后的结果
        # 不需要按细胞展开，直接使用
        gene_contribs = result['gene_contributions']  # shape: (n_genes,)
        predicted_age = result['age_pred']
        n_genes = len(gene_contribs)
        
        if gene_names is not None and len(gene_names) >= n_genes:
            gene_list = gene_names[:n_genes]
        else:
            gene_list = [f"gene_{i}" for i in range(n_genes)]
        
        # 每个donor的每个基因一条记录
        for gene_name, contrib in zip(gene_list, gene_contribs):
            records.append({
                'seed': seed,
                'dataset': donor_to_dataset[donor_id],
                'donor_id': donor_id,
                'gene_name': gene_name,
                'contribution_score': float(contrib),
                'predicted_age': float(predicted_age)
            })
    
    return pd.DataFrame(records)

def format_cell_gene_contributions(
    contrib_results: Dict,
    seed_data: Dict,
    seed: int,
    donor_to_dataset: Dict,
    gene_names: List[str]
) -> pd.DataFrame:
    """
    格式化细胞-基因贡献度结果
    
    输出格式：每个donor的每个细胞的每个基因一行
    Columns: seed, dataset, donor_id, cell_index, cell_position, gene_name, contribution_score, predicted_age
    """
    records = []
    
    for donor_id, result in contrib_results.items():
        if 'cell_gene_contributions' not in result:
            continue
        
        cell_gene_contribs = result['cell_gene_contributions']  # [n_cells, n_genes]
        
        if cell_gene_contribs.size == 0:
            continue
        
        cell_indices = seed_data['cell_indices'][donor_id]
        predicted_age = result['age_pred']
        n_cells, n_genes = cell_gene_contribs.shape
        
        if gene_names is not None and len(gene_names) >= n_genes:
            gene_list = gene_names[:n_genes]
        else:
            gene_list = [f"gene_{i}" for i in range(n_genes)]
        
        # 为每个细胞的每个基因创建记录
        for cell_pos in range(n_cells):
            cell_idx = cell_indices[cell_pos]
            for gene_idx, gene_name in enumerate(gene_list):
                contrib = cell_gene_contribs[cell_pos, gene_idx]
                records.append({
                    'seed': seed,
                    'dataset': donor_to_dataset[donor_id],
                    'donor_id': donor_id,
                    'cell_index': cell_idx,
                    'cell_position': cell_pos,
                    'gene_name': gene_name,
                    'contribution_score': float(contrib),
                    'predicted_age': float(predicted_age)
                })
    
    return pd.DataFrame(records)

# ===============================
# 结果保存函数
# ===============================

def save_contributions(
    all_results: Dict,
    output_dir: str,
    config: TestConfig
):
    """保存所有贡献度结果"""
    print(f"\n💾 Saving contribution results to {output_dir}...", flush=True)
    os.makedirs(output_dir, exist_ok=True)
    
    all_cell_contrib = []
    all_gene_contrib = []
    # all_cell_gene_contrib = []
    
    for seed, results in all_results.items():
        all_cell_contrib.append(results['cell_contributions'])
        all_gene_contrib.append(results['gene_contributions'])
        # all_cell_gene_contrib.append(results['cell_gene_contributions'])
    
    # 1. 保存细胞贡献度
    df_cell = pd.concat(all_cell_contrib, ignore_index=True)
    cell_path = os.path.join(output_dir, 'cell_contributions.csv')
    df_cell.to_csv(cell_path, index=False)
    print(f"   ✅ Cell contributions saved: {cell_path}", flush=True)
    print(f"      Shape: {df_cell.shape}, Size: {os.path.getsize(cell_path) / 1024 / 1024:.2f} MB", flush=True)
    
    # 2. 保存基因贡献度
    df_gene = pd.concat(all_gene_contrib, ignore_index=True)
    gene_path = os.path.join(output_dir, 'gene_contributions.csv')
    df_gene.to_csv(gene_path, index=False)
    print(f"   ✅ Gene contributions saved: {gene_path}", flush=True)
    print(f"      Shape: {df_gene.shape}, Size: {os.path.getsize(gene_path) / 1024 / 1024:.2f} MB", flush=True)
    
    # 3. 保存细胞-基因贡献度（压缩）
    # df_cell_gene = pd.concat(all_cell_gene_contrib, ignore_index=True)
    # cell_gene_path = os.path.join(output_dir, 'cell_gene_contributions.csv.gz')
    # df_cell_gene.to_csv(cell_gene_path, index=False, compression='gzip')
    # print(f"   ✅ Cell-gene contributions saved: {cell_gene_path}", flush=True)
    # print(f"      Shape: {df_cell_gene.shape}, Size: {os.path.getsize(cell_gene_path) / 1024 / 1024:.2f} MB", flush=True)
    
    # 4. 保存配置信息
    config_info = {
        'random_seeds': config.RANDOM_SEEDS,
        'max_cells_per_donor': config.MAX_CELLS_PER_DONOR_TEST,
        'use_raw': config.USE_RAW,
        'use_marker': config.USE_MARKER,
        'is_scaled': config.IS_SCALED,
        'cell_contrib_method': config.CELL_CONTRIB_METHOD,
        'gene_contrib_method': config.GENE_CONTRIB_METHOD,
        'test_datasets': config.TEST_DATASETS,
        'model_hyperparameters': config.MODEL_HYPERPARAMETERS,
        'weights_cache_dir': config.WEIGHTS_CACHE_DIR,
        'max_donors_per_dataset': config.MAX_DONORS_PER_DATASET
    }
    
    config_path = os.path.join(output_dir, 'test_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_info, f, indent=2)
    print(f"   ✅ Config saved: {config_path}", flush=True)
    
    # 5. 生成统计摘要
    generate_summary_statistics(
        df_cell, 
        df_gene, 
        # df_cell_gene, 
        output_dir
    )
    
    # 6. 保存每个seed的单独文件
    save_per_seed_results(all_results, output_dir)

def generate_summary_statistics(
    df_cell: pd.DataFrame,
    df_gene: pd.DataFrame,
    # df_cell_gene: pd.DataFrame,
    output_dir: str
):
    """生成统计摘要"""
    print("\n📊 Generating summary statistics...", flush=True)
    
    summary_records = []
    
    for dataset in df_cell['dataset'].unique():
        for seed in df_cell['seed'].unique():
            cell_mask = (df_cell['dataset'] == dataset) & (df_cell['seed'] == seed)
            cell_subset = df_cell[cell_mask]
            
            gene_mask = (df_gene['dataset'] == dataset) & (df_gene['seed'] == seed)
            gene_subset = df_gene[gene_mask]
            
            # cg_mask = (df_cell_gene['dataset'] == dataset) & (df_cell_gene['seed'] == seed)
            # cg_subset = df_cell_gene[cg_mask]
            
            if len(cell_subset) > 0:
                n_donors = cell_subset['donor_id'].nunique()
                n_genes_per_donor = len(gene_subset) // n_donors if n_donors > 0 else 0
                
                summary_records.append({
                    'dataset': dataset,
                    'seed': seed,
                    'n_donors': n_donors,
                    'n_cells_total': len(cell_subset),
                    'n_cells_per_donor_avg': len(cell_subset) / n_donors if n_donors > 0 else 0,
                    'n_genes': n_genes_per_donor,
                    'mean_cell_contrib': cell_subset['contribution_score'].mean(),
                    'std_cell_contrib': cell_subset['contribution_score'].std(),
                    'mean_gene_contrib': gene_subset['contribution_score'].mean() if len(gene_subset) > 0 else 0,
                    'std_gene_contrib': gene_subset['contribution_score'].std() if len(gene_subset) > 0 else 0,
                    'mean_predicted_age': cell_subset['predicted_age'].mean(),
                    # 'n_cell_gene_records': len(cg_subset)
                })
    
    df_summary = pd.DataFrame(summary_records)
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"   ✅ Summary statistics saved: {summary_path}", flush=True)
    
    print("\n📈 Summary Overview:", flush=True)
    print("="*100)
    print(df_summary.groupby('dataset')[['n_donors', 'n_cells_total', 'n_genes', 'mean_cell_contrib', 'mean_gene_contrib']].mean().to_string())
    print("="*100)

def save_per_seed_results(all_results: Dict, output_dir: str):
    """保存每个seed的单独结果"""
    seed_dir = os.path.join(output_dir, 'per_seed')
    os.makedirs(seed_dir, exist_ok=True)
    
    print(f"\n📁 Saving per-seed results to {seed_dir}...", flush=True)
    
    for seed, results in all_results.items():
        seed_subdir = os.path.join(seed_dir, f'seed_{seed}')
        os.makedirs(seed_subdir, exist_ok=True)
        
        results['cell_contributions'].to_csv(
            os.path.join(seed_subdir, 'cell_contributions.csv'), index=False
        )
        results['gene_contributions'].to_csv(
            os.path.join(seed_subdir, 'gene_contributions.csv'), index=False
        )
        # results['cell_gene_contributions'].to_csv(
        #     os.path.join(seed_subdir, 'cell_gene_contributions.csv.gz'), 
        #     index=False, compression='gzip'
        # )
    
    print(f"   ✅ Per-seed results saved for {len(all_results)} seeds", flush=True)

# ===============================
# 主测试流程
# ===============================

def main():
    """主测试流程"""
    print("="*80)
    print("🧪 DeepSets Model Weight Testing & Contribution Analysis")
    print("   (With Automatic Weight Cache Loading - Fixed Gene Contributions)")
    print("="*80)
    
    config = TestConfig()
    
    print(f"\n⚙️ Configuration:")
    print(f"   Random Seeds: {config.RANDOM_SEEDS}")
    print(f"   Max Cells per Donor: {config.MAX_CELLS_PER_DONOR_TEST}")
    print(f"   Cell Contribution Method: {config.CELL_CONTRIB_METHOD}")
    print(f"   Gene Contribution Method: {config.GENE_CONTRIB_METHOD}")
    print(f"   Use Raw Counts: {config.USE_RAW}")
    print(f"   Use Marker Genes: {config.USE_MARKER}")
    print(f"   Is Scaled: {config.IS_SCALED}")
    print(f"   Weights Cache Dir: {config.WEIGHTS_CACHE_DIR}")
    print(f"   Output Directory: {config.OUTPUT_DIR}")
    print(f"\n   Model Hyperparameters:")
    for key, value in config.MODEL_HYPERPARAMETERS.items():
        print(f"      {key}: {value}")
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    weights_manager = WeightsCacheManager(config.WEIGHTS_CACHE_DIR)
    
    print(f"\n📦 Available weight caches:")
    available_weights = weights_manager.list_available_weights()
    if available_weights:
        for idx, weight_info in enumerate(available_weights):
            print(f"\n   [{idx}] {weight_info['file']}")
            print(f"       Size: {weight_info['size_mb']:.2f} MB")
            print(f"       Saved: {time.ctime(weight_info['timestamp'])}")
            print(f"       Params: {weight_info['params']}")
    else:
        print("   ⚠️ No weight caches found!")
    
    original_dir = os.getcwd()
    os.chdir(config.DATA_DIR)
    
    try:
        start_time = time.time()
        
        donor_info = load_donor_metadata(config.DATA_DIR)
        
        gene_names = None
        if config.USE_MARKER:
            df_marker = load_marker_genes(config.DATA_DIR)
            if df_marker is not None:
                gene_names = df_marker.index.tolist()
        
        donor_files_dir = os.path.join(config.DATA_DIR, config.DONOR_FILES_DIR)
        test_donors = get_test_donors(donor_info, donor_files_dir, config)
        
        # all_test_donors_list = []
        # for donors in test_donors.values():
        #     all_test_donors_list.extend(donors)
        # n_donors = len(set(all_test_donors_list))
        n_donors = 1284
        n_genes = len(gene_names) if gene_names else 0
        
        print(f"\n🔑 Generating cache key for model weights...")
        print(f"   n_donors: {n_donors}, n_genes: {n_genes}")
        
        hyperparams_with_max_cells = config.MODEL_HYPERPARAMETERS.copy()
        # hyperparams_with_max_cells['max_cells'] = config.MAX_CELLS_PER_DONOR
        hyperparams_with_max_cells['max_cells'] = config.MAX_CELLS_PER_DONOR_TRAIN
        
        cache_key, cache_params = weights_manager.get_cache_key(
            hyperparams_with_max_cells, n_donors, n_genes, config.IS_SCALED
        )
        
        print(f"   Cache key: {cache_key}")
        print(f"   Cache params: {cache_params}")
        
        print(f"\n🔥 Loading model weights from cache...")
        checkpoint = weights_manager.load_weights(cache_key, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        if checkpoint is None:
            print("\n❌ No matching weights found! Please train the model first.")
            print(f"   Expected cache key: {cache_key}")
            print(f"   Expected params: {cache_params}")
            return
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, scaler, max_cells = load_deepsets_model_from_cache(checkpoint, n_genes, device)
        
        print(f"\n✅ Model loaded successfully!")
        print(f"   Device: {device}")
        print(f"   Max cells: {max_cells}")
        
        test_data_by_seed = load_test_data_with_seeds(
            donor_files_dir,
            test_donors,
            config.RANDOM_SEEDS,
            config.MAX_CELLS_PER_DONOR_TEST,
            gene_names,
            config.USE_RAW
        )
        
        load_time = time.time()
        print(f"\n⏱️ Data loading took: {load_time - start_time:.1f}s", flush=True)
        
        all_results = {}
        
        for seed_idx, seed in enumerate(config.RANDOM_SEEDS):
            print(f"\n{'='*80}")
            print(f"🔬 Processing Seed {seed} ({seed_idx + 1}/{len(config.RANDOM_SEEDS)})")
            print(f"{'='*80}")
            
            seed_start = time.time()
            
            try:
                seed_results = compute_contributions_with_model(
                    model,
                    scaler,
                    seed,
                    test_data_by_seed[seed],
                    test_donors,
                    gene_names if gene_names else [],
                    config,
                    device
                )
                
                all_results[seed] = seed_results
                
                seed_time = time.time() - seed_start
                print(f"   ⏱️ Seed {seed} completed in {seed_time:.1f}s", flush=True)
                
            except Exception as e:
                print(f"   ❌ Error processing seed {seed}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                continue
            
            gc.collect()
        
        compute_time = time.time()
        print(f"\n⏱️ Total contribution computation took: {compute_time - load_time:.1f}s", flush=True)
        
        if len(all_results) > 0:
            os.chdir(original_dir)
            save_contributions(all_results, config.OUTPUT_DIR, config)
        else:
            print("\n❌ No results to save!", flush=True)
            return
        
        total_time = time.time()
        print(f"\n⏱️ Total runtime: {total_time - start_time:.1f}s", flush=True)
        
        print("\n" + "="*80)
        print("🎉 Testing completed successfully!")
        print("="*80)
        print(f"📁 Results saved to: {config.OUTPUT_DIR}")
        print(f"📊 Seeds tested: {list(all_results.keys())}")
        print(f"🔬 Contribution methods:")
        print(f"   - Cell: {config.CELL_CONTRIB_METHOD}")
        print(f"   - Gene: {config.GENE_CONTRIB_METHOD}")
        print(f"\n📋 Output files:")
        print(f"   - cell_contributions.csv (每个donor的每个细胞)")
        print(f"   - gene_contributions.csv (每个donor的每个基因)")
        # print(f"   - cell_gene_contributions.csv.gz (每个donor的每个细胞的每个基因)")
        print(f"   - summary_statistics.csv")
        print(f"   - test_config.json")
        print(f"   - per_seed/ (individual seed results)")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}", flush=True)
        import traceback
        traceback.print_exc()
        
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    main()
