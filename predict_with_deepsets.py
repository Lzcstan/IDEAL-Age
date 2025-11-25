#!/usr/bin/env python3
"""
使用已训练的AutoGluon + DeepSets模型进行预测
- 不加载训练数据
- 直接加载模型
- 记录采样的细胞ID
- 贡献度文件名包含方法名
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
import gc
import json
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

from ag_extra_models import DeepSetsTabularModelAttn

# ===============================
# 配置参数
# ===============================

class PredictConfig:
    """预测配置"""
    DATA_DIR = "/personal/ImmAge"
    DONOR_FILES_DIR = "donor_files"
    OUTPUT_DIR = "./ensemble_predictions"
    
    TEST_METADATA_FILE = "test_metadata_sub.csv"
    MARKER_FILE = "marker_genes.csv"
    
    # 🔥 新增：细胞选择文件（可选）
    # CELL_SELECTION_FILE = None  # 如果为None，则随机采样；否则按文件读取
    CELL_SELECTION_FILE = "selected_cells.csv"  # 取消注释以使用
    
    AUTOGLUON_MODEL_PATH = "./AutogluonModels/deepsets_integration"
    
    MAX_CELLS_PER_DONOR = 10000
    USE_RAW = True
    USE_MARKER = True
    
    COMPUTE_CONTRIBUTIONS = True
    CELL_CONTRIB_METHOD = 'integrated_gradient'
    GENE_CONTRIB_METHOD = 'integrated_gradient'
    
    RANDOM_SEEDS = [42] # No use if provide CELL_SELECTION_FILE, should be set list with only one seed
    # MAX_DONORS_PER_DATASET = None
    MAX_DONORS_PER_DATASET = 2

config = PredictConfig()
set_seed(42)

# ===============================
# 加载测试metadata
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

# ===============================
# 🔥 采样函数（记录细胞ID）
# ===============================
# ===============================
# 🔥 新增：加载细胞选择文件
# ===============================

def load_cell_selection_file(cell_selection_file: str) -> Dict[str, List[str]]:
    """
    加载细胞选择文件
    
    Args:
        cell_selection_file: CSV文件路径
        
    Returns:
        Dict[donor_id, List[cell_ids]]
    """
    print(f"\n📋 Loading cell selection file: {cell_selection_file}", flush=True)
    
    file_path = os.path.join(config.DATA_DIR, cell_selection_file)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cell selection file not found: {file_path}")
    
    df_cells = pd.read_csv(file_path)
    
    # 验证列名
    required_cols = ['donor_id', 'cell_id']
    if not all(col in df_cells.columns for col in required_cols):
        raise ValueError(f"Cell selection file must have columns: {required_cols}")
    
    # 按donor分组
    cell_selection = {}
    for donor_id, group in df_cells.groupby('donor_id'):
        cell_selection[donor_id] = group['cell_id'].tolist()
    
    total_cells = sum(len(cells) for cells in cell_selection.values())
    print(f"   ✅ Loaded cell selection for {len(cell_selection)} donors", flush=True)
    print(f"   Total cells: {total_cells}", flush=True)
    
    for donor_id in list(cell_selection.keys())[:5]:
        print(f"      {donor_id}: {len(cell_selection[donor_id])} cells", flush=True)
    if len(cell_selection) > 5:
        print(f"      ... and {len(cell_selection) - 5} more donors", flush=True)
    
    return cell_selection

def sample_cells_from_donor_with_seed(
    donor_file: str,
    donor_id: str,
    seed: int,
    max_cells: int,
    gene_names: Optional[List[str]] = None,
    use_raw: bool = True,
    cell_selection: Optional[Dict[str, List[str]]] = None  # 🔥 新增参数
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    从donor文件采样细胞
    
    Args:
        cell_selection: 如果提供，则按指定的cell_ids读取；否则随机采样
    """
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
        
        # 🔥 关键修改：根据是否提供cell_selection决定采样方式
        if cell_selection is not None and donor_id in cell_selection:
            # 模式1：按指定的cell_ids读取
            specified_cell_ids = cell_selection[donor_id]
            
            # 找到这些cell_ids在adata中的位置
            selected_idx = []
            selected_cell_ids = []
            
            for cell_id in specified_cell_ids:
                if cell_id in adata.obs.index:
                    idx = adata.obs.index.get_loc(cell_id)
                    selected_idx.append(idx)
                    selected_cell_ids.append(cell_id)
            
            if not selected_idx:
                raise ValueError(f"None of the specified cells found in {donor_id}")
            
            selected_idx = np.array(selected_idx)
            original_indices = selected_cell_ids
            
            print(f"    📌 Using {len(selected_idx)} specified cells for donor {donor_id}", flush=True)
        
        else:
            # 模式2：随机采样（原有逻辑）
            np.random.seed(seed)
            if n_cells > max_cells:
                selected_idx = np.random.choice(n_cells, max_cells, replace=False)
                selected_idx = np.sort(selected_idx)
            else:
                selected_idx = np.arange(n_cells)
            
            # 记录原始细胞ID
            if hasattr(adata.obs.index, 'tolist'):
                original_indices = adata.obs.index[selected_idx].tolist()
            else:
                original_indices = [str(i) for i in selected_idx]
        
        cell_data = expr_matrix[selected_idx]
        
        # 保存细胞元数据
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
    gene_names: Optional[List[str]] = None,
    cell_selection: Optional[Dict[str, List[str]]] = None  # 🔥 新增参数
) -> Dict:
    """准备测试数据（支持指定细胞）"""
    
    if cell_selection:
        print(f"\n📊 Preparing test data with SPECIFIED CELLS (NO CACHE)...", flush=True)
    else:
        print(f"\n📊 Preparing test data for {len(seeds)} seeds (NO CACHE)...", flush=True)
    
    original_dir = os.getcwd()
    os.chdir(config.DATA_DIR)
    
    try:
        donor_files_dir = config.DONOR_FILES_DIR
        
        test_datasets_donors = {}
        for dataset in test_metadata['dataset'].unique():
            donors = test_metadata[test_metadata['dataset'] == dataset].index.tolist()
            if config.MAX_DONORS_PER_DATASET:
                donors = donors[:config.MAX_DONORS_PER_DATASET]
            test_datasets_donors[dataset] = donors
        
        results_by_seed = {}
        
        for seed in seeds:
            print(f"\n🎲 Processing seed {seed}...", flush=True)
            
            pseudobulk_data = {}
            cell_data_dict = {}
            cell_indices_dict = {}
            cell_metadata_dict = {}
            
            for dataset_name, donors in test_datasets_donors.items():
                if not donors:
                    continue
                
                print(f"  📊 {dataset_name} ({len(donors)} donors) - seed {seed}...", flush=True)
                
                # Pseudobulk数据
                pseudo_list = []
                for donor in donors:
                    donor_file = os.path.join(donor_files_dir, f'donor_{donor}.h5ad')
                    if not os.path.exists(donor_file):
                        continue
                    
                    try:
                        adata = sc.read_h5ad(donor_file)
                        if gene_names is not None:
                            adata = adata[:, gene_names]
                        
                        if config.USE_RAW and 'counts' in adata.layers:
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
                
                # 🔥 单细胞数据：传入cell_selection
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
                            donor_file, donor, seed, config.MAX_CELLS_PER_DONOR,
                            gene_names, config.USE_RAW,
                            cell_selection=cell_selection  # 🔥 传入
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
                    'max_cells': max_cells
                }
                cell_indices_dict[dataset_name] = cell_indices
                cell_metadata_dict[dataset_name] = cell_metadata
                
                print(f"    ✅ Loaded {len(cell_data)} donors, max_cells={max_cells}", flush=True)
            
            results_by_seed[seed] = {
                'pseudobulk_data': pseudobulk_data,
                'cell_data_dict': cell_data_dict,
                'cell_indices_dict': cell_indices_dict,
                'cell_metadata_dict': cell_metadata_dict,
                'test_datasets_donors': test_datasets_donors
            }
            
            gc.collect()
        
        return results_by_seed
    
    finally:
        os.chdir(original_dir)

# ===============================
# 🔥 加载已训练的模型（不加载训练数据）
# ===============================

def load_trained_model(model_path: str, test_cell_data: Dict) -> TabularPredictor:
    """
    加载已训练的AutoGluon模型
    
    Args:
        model_path: 模型路径
        test_cell_data: 测试集的单细胞数据（DeepSets需要）
    """
    print(f"\n🤖 Loading trained model from: {model_path}", flush=True)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    try:
        predictor = TabularPredictor.load(model_path)
        
        model_names = predictor.model_names()
        print(f"   ✅ Loaded {len(model_names)} models", flush=True)
        for name in model_names:
            print(f"      - {name}", flush=True)
        
        # 🔥 设置DeepSets的共享数据（只需要测试数据）
        DeepSetsTabularModelAttn._shared_cell_data = test_cell_data
        DeepSetsTabularModelAttn._shared_test_cell_data = test_cell_data
        DeepSetsTabularModelAttn._shared_max_cells = max(
            [cells.shape[0] for cells in test_cell_data.values()]
        ) if test_cell_data else 1000
        
        print(f"   📏 Set shared test cell data: {len(test_cell_data)} donors", flush=True)
        
        return predictor
        
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}", flush=True)
        raise

# ===============================
# DeepSets贡献度计算
# ===============================

# ===============================
# 🔥 修改：DeepSets贡献度计算（保存cell_gene_contributions）
# ===============================

def compute_deepsets_contributions(
    predictor: TabularPredictor,
    test_cell_data: Dict,
    cell_indices: Dict,  # 🔥 新增参数：细胞ID
    gene_names: List[str],
    method_cell: str,
    method_gene: str
) -> Optional[Dict]:
    """计算DeepSets贡献度"""
    
    print(f"\n🔬 Computing DeepSets contributions...", flush=True)
    print(f"   Cell method: {method_cell}", flush=True)
    print(f"   Gene method: {method_gene}", flush=True)
    
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
        
        deepsets_model_wrapper = predictor._trainer.load_model(deepsets_name)
        deepsets_model = deepsets_model_wrapper.model
        deepsets_scaler = deepsets_model_wrapper.scaler
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        all_cell_contrib = []
        all_gene_contrib = []
        all_cell_gene_contrib = []  # 🔥 新增
        
        donor_ids = list(test_cell_data.keys())
        
        for donor_id in tqdm(donor_ids, desc="Computing contributions"):
            if donor_id not in test_cell_data:
                continue
            
            cells = test_cell_data[donor_id]
            n_cells = cells.shape[0]
            
            cells_scaled = deepsets_scaler.transform(cells).astype(np.float32)
            cells_tensor = torch.from_numpy(cells_scaled).unsqueeze(0).to(device)
            mask = torch.ones(1, n_cells, device=device, dtype=torch.bool)
            
            # 细胞贡献度
            if method_cell in ['attention', 'activation']:
                cell_result = deepsets_model.get_cell_contributions_attn(
                    cells_tensor, mask=mask, method=method_cell
                )
            else:
                cell_result = deepsets_model.get_cell_contributions(
                    cells_tensor, mask=mask, method=method_cell, 
                    target='H', normalize=True
                )
            
            # 🔥 基因贡献度（per_cell=True以获取cell_gene_contributions）
            gene_result = deepsets_model.get_gene_contributions(
                cells_tensor, mask=mask, method=method_gene, per_cell=True
            )
            
            # 获取细胞ID
            donor_cell_ids = cell_indices.get(donor_id, [f"cell_{i}" for i in range(n_cells)])
            
            # 保存细胞贡献度
            for cell_idx, contrib in enumerate(cell_result['cell_contributions'][0]):
                cell_id = donor_cell_ids[cell_idx] if cell_idx < len(donor_cell_ids) else f"cell_{cell_idx}"
                all_cell_contrib.append({
                    'donor_id': donor_id,
                    'cell_id': cell_id,  # 🔥 使用真实cell_id
                    'cell_position': cell_idx,
                    'contribution_score': float(contrib),
                    'predicted_age': float(cell_result['age_pred'][0])
                })
            
            # 保存基因贡献度（聚合）
            gene_contribs = np.asarray(gene_result['gene_contributions'][0]).reshape(-1)
            for gene_idx, (gene_name, contrib) in enumerate(zip(gene_names, gene_contribs)):
                all_gene_contrib.append({
                    'donor_id': donor_id,
                    'gene_name': gene_name,
                    'contribution_score': float(contrib),
                    'predicted_age': float(gene_result['age_pred'][0])
                })
            
            # 🔥 保存细胞-基因贡献度（如果有）
            if gene_result['cell_gene_contributions'] is not None:
                cell_gene_contribs = gene_result['cell_gene_contributions'][0]  # [N, G]
                
                for cell_idx in range(n_cells):
                    cell_id = donor_cell_ids[cell_idx] if cell_idx < len(donor_cell_ids) else f"cell_{cell_idx}"
                    
                    for gene_idx, gene_name in enumerate(gene_names):
                        all_cell_gene_contrib.append({
                            'donor_id': donor_id,
                            'cell_id': cell_id,  # 🔥 使用真实cell_id
                            'cell_position': cell_idx,
                            'gene_name': gene_name,
                            'contribution_score': float(cell_gene_contribs[cell_idx, gene_idx]),
                            'predicted_age': float(gene_result['age_pred'][0])
                        })
        
        df_cell_contrib = pd.DataFrame(all_cell_contrib)
        df_gene_contrib = pd.DataFrame(all_gene_contrib)
        df_cell_gene_contrib = pd.DataFrame(all_cell_gene_contrib) if all_cell_gene_contrib else None
        
        print(f"   ✅ Computed:", flush=True)
        print(f"      Cell contributions: {len(df_cell_contrib)} records", flush=True)
        print(f"      Gene contributions: {len(df_gene_contrib)} records", flush=True)
        if df_cell_gene_contrib is not None:
            print(f"      Cell-gene contributions: {len(df_cell_gene_contrib)} records", flush=True)
        
        return {
            'cell_contributions': df_cell_contrib,
            'gene_contributions': df_gene_contrib,
            'cell_gene_contributions': df_cell_gene_contrib,  # 🔥 新增
            'cell_method': method_cell,
            'gene_method': method_gene
        }
        
    except Exception as e:
        print(f"   ❌ Failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

# ===============================
# 预测函数
# ===============================

def predict_with_all_models(
    predictor: TabularPredictor,
    test_pseudobulk: pd.DataFrame,
    test_metadata: pd.DataFrame,
    dataset_name: str,
    seed: int,
    has_age: bool
) -> Dict:
    """使用所有模型进行预测"""
    print(f"  🔮 Predicting {dataset_name} (seed={seed})...", flush=True)
    
    try:
        ensemble_predictions = predictor.predict(test_pseudobulk)
        
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
        print(f"     ❌ Failed: {e}", flush=True)
        raise

# ===============================
# 🔥 结果保存（包含细胞ID和方法名）
# ===============================

def save_detailed_predictions(
    all_predictions: Dict,
    all_contributions: Optional[Dict],
    all_cell_indices: Optional[Dict],
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
    print(f"   ✅ Predictions: {pred_path}", flush=True)
    
    # 🔥 2. 保存采样的细胞ID
    if all_cell_indices:
        cell_indices_dir = os.path.join(output_dir, 'sampled_cell_indices')
        os.makedirs(cell_indices_dir, exist_ok=True)
        
        for dataset, seeds_data in all_cell_indices.items():
            for seed, cell_indices in seeds_data.items():
                records = []
                for donor_id, indices in cell_indices.items():
                    for pos, cell_id in enumerate(indices):
                        records.append({
                            'donor_id': donor_id,
                            'cell_position': pos,
                            'cell_id': cell_id
                        })
                
                if records:
                    df_indices = pd.DataFrame(records)
                    indices_path = os.path.join(
                        cell_indices_dir, 
                        f'sampled_cells_{dataset}_seed{seed}.csv'
                    )
                    df_indices.to_csv(indices_path, index=False)
        
        print(f"   ✅ Cell indices: {cell_indices_dir}", flush=True)

    # 🔥 3. 保存贡献度（包含cell_gene_contributions）
    if all_contributions:
        contrib_dir = os.path.join(output_dir, 'contributions')
        os.makedirs(contrib_dir, exist_ok=True)
        
        for dataset, contrib in all_contributions.items():
            if contrib:
                cell_method = contrib['cell_method']
                gene_method = contrib['gene_method']
                
                # 细胞贡献度
                cell_path = os.path.join(
                    contrib_dir, 
                    f'cell_contributions_{dataset}_{cell_method}.csv'
                )
                contrib['cell_contributions'].to_csv(cell_path, index=False)
                
                # 基因贡献度
                gene_path = os.path.join(
                    contrib_dir, 
                    f'gene_contributions_{dataset}_{gene_method}.csv'
                )
                contrib['gene_contributions'].to_csv(gene_path, index=False)
                
                # 🔥 细胞-基因贡献度（新增）
                if contrib['cell_gene_contributions'] is not None:
                    cell_gene_path = os.path.join(
                        contrib_dir,
                        f'cell_gene_contributions_{dataset}_{gene_method}.csv.gz'
                    )
                    contrib['cell_gene_contributions'].to_csv(
                        cell_gene_path, index=False, compression='gzip'
                    )
                    
                    print(f"      Cell-gene: {cell_gene_path}", flush=True)
        
        print(f"   ✅ Contributions: {contrib_dir}", flush=True)
    
    # 4. 统计摘要
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
    
    # 5. 保存配置
    config_info = {
        'test_metadata': config.TEST_METADATA_FILE,
        'model_path': config.AUTOGLUON_MODEL_PATH,
        'has_age_labels': has_age,
        'compute_contributions': config.COMPUTE_CONTRIBUTIONS,
        'cell_contrib_method': config.CELL_CONTRIB_METHOD,
        'gene_contrib_method': config.GENE_CONTRIB_METHOD,
        'random_seeds': config.RANDOM_SEEDS,
        'max_cells_per_donor': config.MAX_CELLS_PER_DONOR,
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
    print("🎯 Prediction with Trained AutoGluon + DeepSets Model")
    print("="*80)
    
    print(f"\n⚙️ Configuration:")
    print(f"   Model Path: {config.AUTOGLUON_MODEL_PATH}")
    print(f"   Test Metadata: {config.TEST_METADATA_FILE}")
    print(f"   Cell Selection: {config.CELL_SELECTION_FILE if config.CELL_SELECTION_FILE else 'Random sampling'}")
    print(f"   Seeds: {config.RANDOM_SEEDS}")
    print(f"   Contributions: {config.COMPUTE_CONTRIBUTIONS}")
    if config.COMPUTE_CONTRIBUTIONS:
        print(f"   Cell Method: {config.CELL_CONTRIB_METHOD}")
        print(f"   Gene Method: {config.GENE_CONTRIB_METHOD}")
    
    original_dir = os.getcwd()
    
    try:
        start_time = time.time()
        
        # Step 1: 加载测试metadata
        print("\n" + "="*80)
        print("Step 1: Loading Test Metadata")
        print("="*80)
        
        test_metadata, has_age = load_test_metadata(config.TEST_METADATA_FILE)
        
        # Step 2: 加载gene names
        print("\n" + "="*80)
        print("Step 2: Loading Gene Names")
        print("="*80)
        
        gene_names = None
        if config.USE_MARKER:
            os.chdir(config.DATA_DIR)
            df_marker = pd.read_csv(config.MARKER_FILE, index_col=0)
            gene_names = df_marker.index.tolist()
            print(f"   ✅ Loaded {len(gene_names)} marker genes", flush=True)
            os.chdir(original_dir)
        
        # 🔥 Step 2.5: 加载细胞选择文件（如果提供）
        cell_selection = None
        if config.CELL_SELECTION_FILE:
            cell_selection = load_cell_selection_file(config.CELL_SELECTION_FILE)
        
        # Step 3: 准备测试数据
        print("\n" + "="*80)
        if cell_selection:
            print("Step 3: Preparing Test Data (Using Specified Cells)")
        else:
            print("Step 3: Preparing Test Data (Random Sampling)")
        print("="*80)
        
        test_data_by_seed = prepare_test_data_no_cache_with_seeds(
            test_metadata, config.RANDOM_SEEDS, gene_names,
            cell_selection=cell_selection  # 🔥 传入
        )
        
        # Step 4: 加载已训练的模型（不加载训练数据）
        print("\n" + "="*80)
        print("Step 4: Loading Trained Model (NO Training Data)")
        print("="*80)
        
        # 使用第一个seed的数据来初始化DeepSets
        first_seed = config.RANDOM_SEEDS[0]
        all_test_cell_data = {}
        for dataset in test_data_by_seed[first_seed]['cell_data_dict'].keys():
            all_test_cell_data.update(
                test_data_by_seed[first_seed]['cell_data_dict'][dataset]['cell_data']
            )
        
        predictor = load_trained_model(config.AUTOGLUON_MODEL_PATH, all_test_cell_data)

        # Step 5: 进行预测
        print("\n" + "="*80)
        print("Step 5: Running Predictions")
        print("="*80)
        
        first_seed = config.RANDOM_SEEDS[0]
        all_datasets = list(test_data_by_seed[first_seed]['test_datasets_donors'].keys())
        
        all_predictions = {}
        all_contributions = {}
        all_cell_indices = {}
        
        for dataset_name in all_datasets:
            print(f"\n📊 Dataset: {dataset_name}")
            print("-"*70)
            
            dataset_predictions = {}
            dataset_cell_indices = {}
            
            for seed in config.RANDOM_SEEDS:
                print(f"\n🎲 Seed {seed}:")
                
                test_data = test_data_by_seed[seed]
                
                if dataset_name not in test_data['pseudobulk_data']:
                    continue
                
                # 更新共享数据
                DeepSetsTabularModelAttn._shared_test_cell_data = test_data['cell_data_dict'][dataset_name]['cell_data']
                
                pred_result = predict_with_all_models(
                    predictor,
                    test_data['pseudobulk_data'][dataset_name],
                    test_metadata,
                    dataset_name,
                    seed,
                    has_age
                )
                
                dataset_predictions[seed] = pred_result
                dataset_cell_indices[seed] = test_data['cell_indices_dict'][dataset_name]
            
            all_predictions[dataset_name] = dataset_predictions
            all_cell_indices[dataset_name] = dataset_cell_indices
            
            # 🔥 计算贡献度（传入cell_indices）
            if config.COMPUTE_CONTRIBUTIONS:
                print(f"\n🔬 Computing contributions for {dataset_name}...")
                first_seed = config.RANDOM_SEEDS[0]
                contrib_result = compute_deepsets_contributions(
                    predictor,
                    test_data_by_seed[first_seed]['cell_data_dict'][dataset_name]['cell_data'],
                    test_data_by_seed[first_seed]['cell_indices_dict'][dataset_name],  # 🔥 传入cell_indices
                    gene_names if gene_names else [],
                    config.CELL_CONTRIB_METHOD,
                    config.GENE_CONTRIB_METHOD
                )
                if contrib_result:
                    all_contributions[dataset_name] = contrib_result
        
        # Step 6: 保存结果
        print("\n" + "="*80)
        print("Step 6: Saving Results")
        print("="*80)
        
        save_detailed_predictions(
            all_predictions, all_contributions, all_cell_indices,
            config.OUTPUT_DIR, has_age
        )
        
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("🎉 Prediction Completed!")
        print("="*80)
        print(f"⏱️ Total: {total_time:.1f}s")
        print(f"📁 Results: {config.OUTPUT_DIR}")
        print(f"\n📋 Output files:")
        print(f"   - predictions_detailed.csv (ensemble + all models)")
        print(f"   - sampled_cell_indices/ (cell IDs by seed)")
        if config.COMPUTE_CONTRIBUTIONS:
            print(f"   - contributions/ (with method names)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        
    finally:
        os.chdir(original_dir)
    
if __name__ == "__main__":
    main()
