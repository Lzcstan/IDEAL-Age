#!/usr/bin/env python3
"""
AutoGluon + DeepSets 缓存优化版本 - 修复版
参考AutoGluon官方教程优化集成方式
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# AutoGluon imports
from autogluon.core.models import AbstractModel
# from autogluon.core.utils import get_gpu_count

# 导入现有脚本的功能
try:
    from dist_train_cache import (
        DeepSetsAgePredictor,
        SingleCellDataset, 
        collate_fn,
        set_seed
    )
    # 从现有代码导入
    from iage_lr_train_by_dataset_debug import (
        scNETAgePredictor,
        scNETAgePredictorFast,
        # PINNACLEAgePredictor,
        # apply_zscore_normalization
    )
    print("✅ Successfully imported from dist_train_cache.py", flush=True)
except ImportError as e:
    print(f"❌ Failed to import from dist_train_cache.py: {e}", flush=True)
    sys.exit(1)


# ===============================
# 缓存管理系统 (保持原有逻辑)
# ===============================

class DeepSetsTabularModel(AbstractModel):
    """
    修复的DeepSets AutoGluon模型 
    参考: https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-model-advanced.html
    """
    
    def __init__(self, **kwargs):
        # 🔥 确保正确初始化父类
        super().__init__(**kwargs)
        self.model = None
        self.scaler = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_cells = None
        self._features_in = None
        
        # 设置模型类型标识
        self._model_type = 'DeepSetsTabularModel'
    
    def _get_default_auxiliary_params(self) -> dict:
        """确保返回字典格式"""
        default_auxiliary_params = super()._get_default_auxiliary_params()
        # 修复：确保ignored_type_group参数是正确的格式
        extra_auxiliary_params = dict(
            ignored_type_group_raw=[],  # 这些应该保持为list
            ignored_type_group_special=[],  # 这些应该保持为list
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
    
    def _set_default_params(self):
        """设置默认参数"""
        default_params = {
            'hidden_dim': 8,
            'dropout': 0.2,
            'lr': 1e-3,
            'batch_size': 8,
            'epochs': 30,
        }
        for key, value in default_params.items():
            self._set_default_param_value(key, value)
    
    def _get_default_resources(self):
        """设置资源需求"""
        # 使用torch API替代AutoGluon的get_gpu_count
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return [2, 0.5]  # 使用部分GPU
        else:
            return [4, 0]

    def _get_default_stopping_metric(self):
        """设置停止指标"""
        return 'root_mean_squared_error'
    
    def _more_tags(self):
        # 指示这是一个自定义模型
        return {
            'can_use_gpu': torch.cuda.is_available(),
            'requires_fit': True,
            'requires_positive_X': False,
        }

    def _get_default_ag_args_fit(self, **kwargs) -> dict:
        """确保返回字典"""
        try:
            default_ag_args_fit = super()._get_default_ag_args_fit(**kwargs)
            if not isinstance(default_ag_args_fit, dict):
                default_ag_args_fit = {}
        except:
            default_ag_args_fit = {}
        return default_ag_args_fit
    
    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, **kwargs):
        """训练DeepSets模型"""
        
        print(f"    🤖 DeepSets training started...", flush=True)
        
        # 检查cell data是否可用
        if not hasattr(self.__class__, '_shared_cell_data'):
            print("    ⚠️ No single cell data available for DeepSets, using dummy model", flush=True)
            # 创建一个简单的线性模型作为fallback
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
            
            # 只使用前100个特征避免过拟合
            n_features = min(100, X.shape[1])
            X_subset = X.iloc[:, :n_features]
            self.model.fit(X_subset, y)
            self._features_in = X_subset.columns.tolist()
            return self
        
        cell_data = self.__class__._shared_cell_data
        
        # 设置max_cells
        if hasattr(self.__class__, '_shared_max_cells'):
            self.max_cells = self.__class__._shared_max_cells
        else:
            all_donor_ids = list(cell_data.keys())
            self.max_cells = max([cell_data[donor_id].shape[0] for donor_id in all_donor_ids])
        
        print(f"    📏 DeepSets using max_cells: {self.max_cells}", flush=True)
        
        # 准备训练数据
        train_donor_ids = X.index.tolist()
        train_ages = y if isinstance(y, pd.Series) else pd.Series(y, index=train_donor_ids)
        
        valid_donors = [d for d in train_donor_ids if d in cell_data]
        if len(valid_donors) < 5:
            print(f"    ⚠️ Too few valid donors with cell data: {len(valid_donors)}, using linear fallback", flush=True)
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
            
            n_features = min(100, X.shape[1])
            X_subset = X.iloc[:, :n_features]
            self.model.fit(X_subset, train_ages)
            self._features_in = X_subset.columns.tolist()
            return self
        
        train_ages = train_ages.loc[valid_donors]
        
        print(f"    📐 Preparing scaler for {len(valid_donors)} donors...", flush=True)
        
        # 准备StandardScaler
        train_cells_list = []
        for donor_id in valid_donors[:min(len(valid_donors), 50)]:  # 只用50个donors来fit scaler
            if donor_id in cell_data:
                train_cells_list.append(cell_data[donor_id])
        
        if train_cells_list:
            train_cells = np.vstack(train_cells_list)
            self.scaler = StandardScaler()
            self.scaler.fit(train_cells)
            print("    ✅ StandardScaler fitted!", flush=True)
        else:
            # Fallback到identity scaler
            class IdentityScaler:
                def fit(self, X): 
                    return self
                def transform(self, X): 
                    return X.astype(np.float32)
            self.scaler = IdentityScaler()
        
        # 创建数据集和数据加载器
        dataset = SingleCellDataset(cell_data, train_ages, valid_donors, self.scaler, self.max_cells)
        if dataset.n_genes == 0:
            raise ValueError("No valid training data for DeepSets")
        
        print(f"    📊 Dataset: {len(valid_donors)} donors, {dataset.n_genes} genes", flush=True)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.params['batch_size'], 
            shuffle=True, 
            collate_fn=collate_fn, 
            num_workers=0  # 避免multiprocessing问题
        )
        
        # 创建模型
        self.model = DeepSetsAgePredictor(
            input_dim=dataset.n_genes,
            hidden_dim=2**self.params['hidden_dim'],
            dropout=self.params['dropout'],
            max_cells=self.max_cells
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"    🧠 DeepSets model: {total_params:,} parameters", flush=True)
        
        # 训练设置
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])
        
        print(f"    🏃 Training for {self.params['epochs']} epochs...", flush=True)
        
        for epoch in range(self.params['epochs']):
            epoch_losses = []
            self.model.train()
            
            for batch_idx, batch in enumerate(dataloader):
                try:
                    cells = batch['cells'].to(self.device)
                    ages = batch['ages'].to(self.device)
                    masks = batch['masks'].to(self.device)
                    
                    optimizer.zero_grad()
                    pred = self.model(cells, masks)
                    loss = F.mse_loss(pred, ages)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                    
                except Exception as e:
                    print(f"    ⚠️ Batch {batch_idx} error: {e}", flush=True)
                    continue
            
            # 每10个epoch打印一次
            if (epoch + 1) % 10 == 0:
                avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
                print(f"    📊 Epoch {epoch+1:02d}: Loss = {avg_loss:.4f}", flush=True)
        
        print("    ✅ DeepSets training completed!", flush=True)
        return self

    def _predict_proba(self, X, **kwargs):
        """重写预测方法，避免AutoGluon直接调用PyTorch模型的predict"""
        predictions = self._predict(X, **kwargs)
        
        # 对于回归问题，返回predictions本身
        # AutoGluon期望_predict_proba返回概率，但对于回归问题就是预测值
        # return predictions.reshape(-1, 1)  # 确保返回正确的形状
        return predictions.flatten()  # 使用flatten()而不是reshape(-1, 1)

    def _predict(self, X, **kwargs):
        """预测函数"""
        
        if self.model is None:
            return np.full(len(X), 50.0)  # 返回默认年龄
        
        # 如果是sklearn模型(fallback)
        if hasattr(self.model, 'predict') and hasattr(self.model, 'coef_'):
            if hasattr(self, '_features_in'):
                X_subset = X[self._features_in]
                return self.model.predict(X_subset)
            else:
                n_features = min(100, X.shape[1])
                X_subset = X.iloc[:, :n_features]
                return self.model.predict(X_subset)
        
        # DeepSets模型
        if not hasattr(self.__class__, '_shared_cell_data'):
            return np.full(len(X), 50.0)
        
        cell_data = self.__class__._shared_cell_data
        donor_ids = X.index.tolist()
        
        valid_donors = [d for d in donor_ids if d in cell_data]
        if len(valid_donors) == 0:
            return np.full(len(donor_ids), 50.0)
        
        # 创建dummy ages for dataset
        dummy_ages = pd.Series([0.0] * len(valid_donors), index=valid_donors)
        dataset = SingleCellDataset(cell_data, dummy_ages, valid_donors, self.scaler, self.max_cells)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.params['batch_size'], 
            shuffle=False, 
            collate_fn=collate_fn, 
            num_workers=0
        )
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                try:
                    cells = batch['cells'].to(self.device)
                    masks = batch['masks'].to(self.device)
                    pred = self.model(cells, masks)
                    predictions.extend(pred.cpu().numpy())
                except Exception as e:
                    print(f"    ⚠️ Prediction batch error: {e}", flush=True)
                    continue
        
        # 构建预测结果字典
        pred_dict = dict(zip(valid_donors, predictions))
        final_preds = [pred_dict.get(d, 50.0) for d in donor_ids]
        
        return np.array(final_preds)

    def get_cell_contributions(self, donor_ids, method='gradient'):
        """
        获取指定供体的细胞对年龄预测的贡献分析
        
        Args:
            donor_ids: list of str, 要分析的供体ID列表
            method: str, 'gradient' | 'activation'
            
        Returns:
            dict: {
                donor_id: {
                    'cell_contributions': np.array,  # 每个细胞的贡献分数
                    'cell_features': np.array,       # 细胞编码特征
                    'age_pred': float,               # 预测年龄
                    'n_cells': int,                  # 细胞数量
                }
            }
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if not hasattr(self.__class__, '_shared_cell_data'):
            raise ValueError("No single cell data available")
        
        cell_data = self.__class__._shared_cell_data
        valid_donors = [d for d in donor_ids if d in cell_data]
        
        if len(valid_donors) == 0:
            raise ValueError("No valid donors found in cell data")
        
        results = {}
        
        for donor_id in valid_donors:
            print(f"📊 Analyzing cell contributions for donor {donor_id}...", flush=True)
            
            # 准备单个供体的数据
            cells = cell_data[donor_id]  # [n_cells, n_genes]
            n_cells = cells.shape[0]
            
            # 标准化
            cells_scaled = self.scaler.transform(cells).astype(np.float32)
            
            # 转换为tensor并添加batch维度
            cells_tensor = torch.FloatTensor(cells_scaled).unsqueeze(0).to(self.device)  # [1, n_cells, n_genes]
            
            # 创建mask
            mask = torch.ones(1, n_cells, device=self.device, dtype=torch.bool)
            
            # 获取贡献分析
            with torch.no_grad():
                if method in ['gradient']:
                    # 对于梯度方法，需要enable_grad
                    cells_tensor.requires_grad_(True)
            
            contrib_result = self.model.get_cell_contributions(
                cells_tensor, mask=mask, method=method
            )
            
            results[donor_id] = {
                'cell_contributions': contrib_result['cell_contributions'][0],  # 移除batch维度
                'cell_features': contrib_result['cell_features'][0],
                'age_pred': contrib_result['age_pred'][0],
                'n_cells': n_cells,
            }
            
            print(f"  ✅ {n_cells} cells analyzed, predicted age: {contrib_result['age_pred'][0]:.1f}", flush=True)
        
        return results

    def get_gene_contributions(self, donor_ids, gene_names=None, method='gradient'):
        """
        获取指定供体的基因对年龄预测的贡献分析
        
        Args:
            donor_ids: list of str, 要分析的供体ID列表  
            gene_names: list of str, 基因名称列表（用于结果标注）
            method: str, 'gradient' | 'integrated_gradient'
            
        Returns:
            dict: {
                donor_id: {
                    'gene_contributions': np.array,      # 每个基因的总贡献
                    'cell_gene_contributions': np.array, # 每个细胞中每个基因的贡献 
                    'age_pred': float,
                    'n_cells': int,
                    'n_genes': int,
                    'top_genes': list,  # 贡献最大的前10个基因
                }
            }
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if not hasattr(self.__class__, '_shared_cell_data'):
            raise ValueError("No single cell data available")
        
        cell_data = self.__class__._shared_cell_data
        valid_donors = [d for d in donor_ids if d in cell_data]
        
        if len(valid_donors) == 0:
            raise ValueError("No valid donors found in cell data")
        
        results = {}
        
        for donor_id in valid_donors:
            print(f"🧬 Analyzing gene contributions for donor {donor_id}...", flush=True)
            
            # 准备单个供体的数据
            cells = cell_data[donor_id]  # [n_cells, n_genes]
            n_cells, n_genes = cells.shape
            
            # 标准化
            cells_scaled = self.scaler.transform(cells).astype(np.float32)
            
            # 转换为tensor
            cells_tensor = torch.FloatTensor(cells_scaled).unsqueeze(0).to(self.device)  # [1, n_cells, n_genes]
            mask = torch.ones(1, n_cells, device=self.device, dtype=torch.bool)
            
            # 获取基因贡献
            contrib_result = self.model.get_gene_contributions(
                cells_tensor, mask=mask, method=method
            )
            
            gene_contributions = contrib_result['gene_contributions'][0]  # [n_genes]
            
            # 找出贡献最大的基因
            top_gene_indices = np.argsort(gene_contributions)[-10:][::-1]  # 前10个
            
            if gene_names is not None:
                top_genes = [(gene_names[i], gene_contributions[i]) for i in top_gene_indices]
            else:
                top_genes = [(f"gene_{i}", gene_contributions[i]) for i in top_gene_indices]
            
            results[donor_id] = {
                'gene_contributions': gene_contributions,
                'cell_gene_contributions': contrib_result['cell_gene_contributions'][0],
                'age_pred': contrib_result['age_pred'][0],
                'n_cells': n_cells,
                'n_genes': n_genes,
                'top_genes': top_genes,
            }
            
            print(f"  ✅ {n_cells} cells, {n_genes} genes analyzed", flush=True)
            print(f"  🏆 Top 3 genes: {[f'{name}({score:.3f})' for name, score in top_genes[:3]]}", flush=True)
        
        return results

    def visualize_contributions(self, contrib_results, donor_id, save_dir="./results", 
                            plot_type='cell', gene_names=None):
        """
        可视化贡献分析结果
        
        Args:
            contrib_results: get_cell_contributions或get_gene_contributions的返回结果
            donor_id: str, 要可视化的供体ID
            save_dir: str, 保存路径
            plot_type: str, 'cell' | 'gene'
            gene_names: list, 基因名称（用于gene plot）
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if donor_id not in contrib_results:
            raise ValueError(f"Donor {donor_id} not found in results")
        
        result = contrib_results[donor_id]
        os.makedirs(save_dir, exist_ok=True)
        
        if plot_type == 'cell':
            # 细胞贡献图
            plt.figure(figsize=(12, 6))
            
            cell_contributions = result['cell_contributions']
            cell_indices = np.arange(len(cell_contributions))
            
            plt.subplot(1, 2, 1)
            plt.bar(cell_indices, cell_contributions)
            plt.xlabel('Cell Index')
            plt.ylabel('Contribution Score')
            plt.title(f'Cell Contributions - {donor_id}\nPredicted Age: {result["age_pred"]:.1f}')
            
            plt.subplot(1, 2, 2)
            plt.hist(cell_contributions, bins=30, alpha=0.7)
            plt.xlabel('Contribution Score')
            plt.ylabel('Number of Cells')
            plt.title('Contribution Distribution')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/cell_contributions_{donor_id}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        elif plot_type == 'gene':
            # 基因贡献图
            plt.figure(figsize=(15, 10))
            
            gene_contributions = result['gene_contributions']
            top_genes = result['top_genes']
            
            # Top基因条形图
            plt.subplot(2, 2, 1)
            top_names, top_scores = zip(*top_genes)
            plt.barh(range(len(top_names)), top_scores)
            plt.yticks(range(len(top_names)), top_names)
            plt.xlabel('Contribution Score')
            plt.title(f'Top 10 Gene Contributions - {donor_id}')
            
            # 基因贡献分布
            plt.subplot(2, 2, 2)
            plt.hist(gene_contributions, bins=50, alpha=0.7)
            plt.xlabel('Contribution Score')
            plt.ylabel('Number of Genes')
            plt.title('Gene Contribution Distribution')
            plt.yscale('log')
            
            # 细胞-基因贡献热图（取前20个基因和前50个细胞）
            plt.subplot(2, 1, 2)
            top_gene_indices = np.argsort(gene_contributions)[-20:]
            cell_gene_contrib = result['cell_gene_contributions'][:50, top_gene_indices]  # [cells, genes]
            
            sns.heatmap(cell_gene_contrib.T, 
                    xticklabels=False, 
                    yticklabels=[top_names[i] for i in range(min(20, len(top_names)))],
                    cmap='viridis')
            plt.xlabel('Cells (First 50)')
            plt.ylabel('Top Contributing Genes') 
            plt.title('Cell-Gene Contribution Heatmap')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/gene_contributions_{donor_id}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Visualization saved: {save_dir}/{plot_type}_contributions_{donor_id}.png", flush=True)


class DeepSetsTabularModelAttn(AbstractModel):
    """
    修复的DeepSets AutoGluon模型 
    参考: https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-custom-model-advanced.html
    """
    _weights_cache_dir = "./results_debug/deepsets_weights"
    
    def __init__(self, **kwargs):
        # 🔥 确保正确初始化父类
        super().__init__(**kwargs)
        self.model = None
        self.scaler = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_cells = None
        self._features_in = None
        
        # 设置模型类型标识
        self._model_type = 'DeepSetsTabularModelAttn'
    
    def _get_default_auxiliary_params(self) -> dict:
        """确保返回字典格式"""
        default_auxiliary_params = super()._get_default_auxiliary_params()
        # 修复：确保ignored_type_group参数是正确的格式
        extra_auxiliary_params = dict(
            ignored_type_group_raw=[],  # 这些应该保持为list
            ignored_type_group_special=[],  # 这些应该保持为list
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _set_default_params(self):
        """设置默认参数"""
        default_params = {
            'hidden_dim': 8,
            'dropout': 0.2,
            'lr': 1e-3,
            'batch_size': 8,
            'epochs': 100,
            'cache_every_epoch': None,      # 🔥 新增
            'continue_training': False,     # 🔥 新增
        }
        for key, value in default_params.items():
            self._set_default_param_value(key, value)
    
    def _get_default_resources(self):
        """设置资源需求"""
        # 使用torch API替代AutoGluon的get_gpu_count
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return [2, 0.5]  # 使用部分GPU
        else:
            return [4, 0]

    def _get_default_stopping_metric(self):
        """设置停止指标"""
        return 'root_mean_squared_error'
    
    def _more_tags(self):
        # 指示这是一个自定义模型
        return {
            'can_use_gpu': torch.cuda.is_available(),
            'requires_fit': True,
            'requires_positive_X': False,
        }

    def _get_default_ag_args_fit(self, **kwargs) -> dict:
        """确保返回字典"""
        try:
            default_ag_args_fit = super()._get_default_ag_args_fit(**kwargs)
            if not isinstance(default_ag_args_fit, dict):
                default_ag_args_fit = {}
        except:
            default_ag_args_fit = {}
        return default_ag_args_fit

    def _predict_proba(self, X, **kwargs):
        """重写预测方法，避免AutoGluon直接调用PyTorch模型的predict"""
        predictions = self._predict(X, **kwargs)
        
        # 对于回归问题，返回predictions本身
        # AutoGluon期望_predict_proba返回概率，但对于回归问题就是预测值
        # return predictions.reshape(-1, 1)  # 确保返回正确的形状
        return predictions.flatten()  # 使用flatten()而不是reshape(-1, 1)

    def _predict(self, X, **kwargs):
        """预测函数"""

        # print(f'Input X:')
        # print(X)
        
        if self.model is None:
            return np.full(len(X), 50.0)  # 返回默认年龄
        
        # 如果是sklearn模型(fallback)
        if hasattr(self.model, 'predict') and hasattr(self.model, 'coef_'):
            if hasattr(self, '_features_in'):
                X_subset = X[self._features_in]
                return self.model.predict(X_subset)
            else:
                n_features = min(100, X.shape[1])
                X_subset = X.iloc[:, :n_features]
                return self.model.predict(X_subset)
        
        # DeepSets模型
        if not hasattr(self.__class__, '_shared_cell_data'):
            return np.full(len(X), 50.0)
        
        cell_data = self.__class__._shared_cell_data
        donor_ids = X.index.tolist()
        
        valid_donors = [d for d in donor_ids if d in cell_data]
        print(f'Predicting {len(valid_donors)} donors...')
        if len(valid_donors) == 0:
            return np.full(len(donor_ids), 50.0)
        
        # 创建dummy ages for dataset
        dummy_ages = pd.Series([0.0] * len(valid_donors), index=valid_donors)
        dataset = SingleCellDataset(
            cell_data, 
            dummy_ages, 
            valid_donors, 
            self.scaler, 
            # self.max_cells,
            max([cell_data[d].shape[0] for d in valid_donors])
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=1, # Tid 
            shuffle=False, 
            collate_fn=collate_fn, 
            num_workers=0
        )
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                try:
                    cells = batch['cells'].to(self.device)
                    masks = batch['masks'].to(self.device)
                    pred, cache = self.model(cells, masks)
                    predictions.extend(pred.cpu().numpy())
                except Exception as e:
                    print(f"    ⚠️ Prediction batch error: {e}", flush=True)
                    continue
        
        # 构建预测结果字典
        pred_dict = dict(zip(valid_donors, predictions))
        final_preds = [pred_dict.get(d, 50.0) for d in donor_ids]
        
        return np.array(final_preds)

    def get_cell_contributions(self, donor_ids, method='attention', mode='test'):
        if self.model is None:
            raise ValueError("Model not trained yet")
        if not hasattr(self.__class__, '_shared_cell_data'):
            raise ValueError("No single cell data available")

        cell_data = self.__class__._shared_cell_data
        valid_donors = [d for d in donor_ids if d in cell_data]
        if len(valid_donors) == 0:
            raise ValueError("No valid donors found in cell data")

        results = {}
        for donor_id in valid_donors:
            print(f"📊 Analyzing cell contributions for donor {donor_id}...", flush=True)
            cells = cell_data[donor_id]
            n_cells = cells.shape[0]
            cells_scaled = self.scaler.transform(cells).astype(np.float32)
            cells_tensor = torch.from_numpy(cells_scaled).unsqueeze(0).to(self.device)  # [1, N, G]
            mask = torch.ones(1, n_cells, device=self.device, dtype=torch.bool)

            if method in ['attention', 'activation']:
                contrib_result = self.model.get_cell_contributions_attn(
                    cells_tensor, mask=mask, method=method
                )
            elif method in ['gradient', 'grad_input', 'integrated_gradient']:
                contrib_result = self.model.get_cell_contributions(
                    # cells_tensor, mask=mask, method=method, target='H', normalize=True
                    cells_tensor, mask=mask, method=method, target='cells', normalize=True
                )
            else:
                raise ValueError("Unsupported method")

            results[donor_id] = {
                'cell_contributions': contrib_result['cell_contributions'][0],
                'age_pred': contrib_result['age_pred'][0],
                'n_cells': n_cells,
                # 可选：如果使用 attention 方法，保存每query的注意力
                # 'attn_per_query': contrib_result.get('aux', {}).get('attn_per_query', None)
            }
            print(f"  ✅ {n_cells} cells analyzed, predicted age: {contrib_result['age_pred'][0]:.1f}", flush=True)

        return results

    def get_gene_contributions(self, donor_ids, gene_names=None, method='gradient', mode='test'):
        """
        获取指定供体的基因对年龄预测的贡献分析
        
        Args:
            donor_ids: list of str, 要分析的供体ID列表  
            gene_names: list of str, 基因名称列表（用于结果标注）
            method: str, 'gradient' | 'integrated_gradient'
            
        Returns:
            dict: {
                donor_id: {
                    'gene_contributions': np.array,      # 每个基因的总贡献
                    'cell_gene_contributions': np.array, # 每个细胞中每个基因的贡献 
                    'age_pred': float,
                    'n_cells': int,
                    'n_genes': int,
                    'top_genes': list,  # 贡献最大的前10个基因
                }
            }
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if not hasattr(self.__class__, '_shared_cell_data'):
            raise ValueError("No single cell data available")
        
        cell_data = self.__class__._shared_cell_data
        valid_donors = [d for d in donor_ids if d in cell_data]
        
        if len(valid_donors) == 0:
            raise ValueError("No valid donors found in cell data")
        
        results = {}
        
        for donor_id in valid_donors:
            print(f"🧬 Analyzing gene contributions for donor {donor_id}...", flush=True)
            
            # 准备单个供体的数据
            cells = cell_data[donor_id]  # [n_cells, n_genes]
            n_cells, n_genes = cells.shape
            
            # 标准化
            cells_scaled = self.scaler.transform(cells).astype(np.float32)
            
            # 转换为tensor
            cells_tensor = torch.FloatTensor(cells_scaled).unsqueeze(0).to(self.device)  # [1, n_cells, n_genes]
            mask = torch.ones(1, n_cells, device=self.device, dtype=torch.bool)
            
            # 在 DeepSetsTabularModelAttn.get_gene_contributions 中：
            contrib_result = self.model.get_gene_contributions(
                cells_tensor, mask=mask, method=method, per_cell=True  # 显式要求返回每细胞×每基因
            )

            cell_gene_contributions = contrib_result.get('cell_gene_contributions', None)
            if cell_gene_contributions is not None:
                cell_gene_contributions = cell_gene_contributions[0]     # [N, G]
            else:
                # 兜底：如果没返回，则构造一个占位或跳过热图
                cell_gene_contributions = None

            # 找出贡献最大的基因（确保为 1D numpy）
            gene_contributions = np.asarray(contrib_result['gene_contributions'][0]).reshape(-1)

            # 规范化 gene_names
            if gene_names is not None and not isinstance(gene_names, list):
                try:
                    gene_names = list(gene_names)
                except Exception:
                    gene_names = [str(x) for x in gene_names]

            # 计算 top 索引并转换为 Python int
            top_gene_indices = np.asarray(np.argsort(gene_contributions)[-10:][::-1]).tolist()
            top_gene_indices = [int(i) for i in top_gene_indices]

            # 生成 top_genes
            if gene_names is not None:
                n_names = len(gene_names)
                top_genes = []
                for i in top_gene_indices:
                    name = gene_names[i] if 0 <= i < n_names else f"gene_{i}"
                    top_genes.append((name, float(gene_contributions[i])))
            else:
                top_genes = [(f"gene_{i}", float(gene_contributions[i])) for i in top_gene_indices]
            
            results[donor_id] = {
                'gene_contributions': gene_contributions,
                'cell_gene_contributions': cell_gene_contributions if cell_gene_contributions is not None else np.empty((0, 0)),
                'age_pred': contrib_result['age_pred'][0],
                'n_cells': n_cells,
                'n_genes': n_genes,
                'top_genes': top_genes,
            }
            
            print(f"  ✅ {n_cells} cells, {n_genes} genes analyzed", flush=True)
            print(f"  🏆 Top 3 genes: {[f'{name}({score:.3f})' for name, score in top_genes[:3]]}", flush=True)
        
        return results

    def visualize_contributions(self, contrib_results, donor_id, save_dir="./results", 
                            plot_type='cell', gene_names=None, signed=False):
        """
        可视化贡献分析结果
        
        Args:
            contrib_results: get_cell_contributions或get_gene_contributions的返回结果
            donor_id: str, 要可视化的供体ID
            save_dir: str, 保存路径
            plot_type: str, 'cell' | 'gene'
            gene_names: list, 基因名称（用于gene plot）
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if donor_id not in contrib_results:
            raise ValueError(f"Donor {donor_id} not found in results")
        
        result = contrib_results[donor_id]
        os.makedirs(save_dir, exist_ok=True)

        if plot_type == 'cell':
            cell_contributions = result['cell_contributions']
            cell_indices = np.arange(len(cell_contributions))
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # 左图：条形图
            if signed:
                # 🔥 使用红蓝配色
                colors = ['red' if x > 0 else 'blue' for x in cell_contributions]
                axes[0].bar(cell_indices, cell_contributions, color=colors)
                axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
                axes[0].set_ylabel('Signed Contribution')
                axes[0].set_title(f'Cell Contributions (Signed)\nRed=Pro-aging, Blue=Anti-aging')
            else:
                axes[0].bar(cell_indices, cell_contributions)
                axes[0].set_ylabel('Contribution Magnitude')
                axes[0].set_title('Cell Contributions (Unsigned)')
            
            # 右图：分布
            axes[1].hist(cell_contributions, bins=30, alpha=0.7)
            if signed:
                axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/cell_contributions_{donor_id}_{'signed' if signed else 'unsigned'}.png", 
                    dpi=300, bbox_inches='tight')
        elif plot_type == 'gene':
            plt.figure(figsize=(15, 10))

            gene_contributions = np.asarray(result['gene_contributions']).reshape(-1)
            top_genes = result['top_genes']

            # 子图1: Top基因条形图
            plt.subplot(2, 2, 1)
            if len(top_genes) > 0:
                top_names, top_scores = zip(*top_genes)
                
                if signed:
                    # 🔥 带符号：红蓝配色
                    colors = ['red' if x > 0 else 'blue' for x in top_scores]
                    plt.barh(range(len(top_names)), top_scores, color=colors, alpha=0.7)
                    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
                    plt.xlabel('Signed Contribution Score')
                    plt.title(f'Top 10 Genes (Signed) - {donor_id}\nRed=Pro-aging, Blue=Anti-aging')
                else:
                    plt.barh(range(len(top_names)), top_scores, color='steelblue', alpha=0.7)
                    plt.xlabel('Contribution Score')
                    plt.title(f'Top 10 Gene Contributions - {donor_id}')
                
                plt.yticks(range(len(top_names)), top_names)
            else:
                plt.barh([], [])
                plt.yticks([], [])
            plt.grid(axis='x', alpha=0.3)

            # 子图2: 基因贡献分布
            plt.subplot(2, 2, 2)
            if gene_contributions.size > 0:
                if signed:
                    # 🔥 分别显示正负贡献
                    pos_genes = gene_contributions[gene_contributions > 0]
                    neg_genes = gene_contributions[gene_contributions < 0]
                    
                    plt.hist(pos_genes, bins=30, alpha=0.6, color='red', label='Pro-aging')
                    plt.hist(neg_genes, bins=30, alpha=0.6, color='blue', label='Anti-aging')
                    plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
                    plt.legend()
                    plt.xlabel('Signed Contribution Score')
                else:
                    plt.hist(gene_contributions, bins=min(50, max(10, gene_contributions.size // 50)), alpha=0.7)
                    plt.yscale('log')
                    plt.xlabel('Contribution Score')
                
                plt.ylabel('Number of Genes')
                plt.title('Gene Contribution Distribution')
            else:
                plt.hist([], bins=1)
            plt.grid(alpha=0.3)

            # 子图3: 细胞-基因贡献热图
            plt.subplot(2, 1, 2)
            cell_gene_contrib = result.get('cell_gene_contributions', None)

            if isinstance(cell_gene_contrib, np.ndarray) and cell_gene_contrib.ndim == 2 and cell_gene_contrib.size > 0:
                G_cg = cell_gene_contrib.shape[1]
                if gene_contributions.size != G_cg:
                    G_min = min(gene_contributions.size, G_cg)
                    if G_min == 0:
                        cg_top = None
                    else:
                        gene_contributions = gene_contributions[:G_min]
                        cell_gene_contrib = cell_gene_contrib[:, :G_min]
            else:
                G_cg = gene_contributions.size

            # 🔥 选取 top 基因：按绝对值排序（对 signed 很重要）
            if gene_contributions.size == 0:
                top_gene_indices = np.array([], dtype=int)
            else:
                safe_gc = np.nan_to_num(gene_contributions, nan=0.0, posinf=0.0, neginf=0.0)
                k = min(20, safe_gc.size)
                
                if signed:
                    # 🔥 按绝对值选择最重要的基因
                    top_k_part = np.argpartition(np.abs(safe_gc), -k)[-k:]
                    top_gene_indices = top_k_part[np.argsort(np.abs(safe_gc[top_k_part]))[::-1]]
                else:
                    # 原来的逻辑：按值大小选择
                    top_k_part = np.argpartition(safe_gc, -k)[-k:]
                    top_gene_indices = top_k_part[np.argsort(safe_gc[top_k_part])[::-1]]
                
                top_gene_indices = top_gene_indices.astype(int)

            if isinstance(cell_gene_contrib, np.ndarray) and cell_gene_contrib.ndim == 2 and cell_gene_contrib.size > 0:
                valid_mask = (top_gene_indices >= 0) & (top_gene_indices < cell_gene_contrib.shape[1])
                top_gene_indices = top_gene_indices[valid_mask]

            if not (isinstance(cell_gene_contrib, np.ndarray) and cell_gene_contrib.ndim == 2 and cell_gene_contrib.size > 0 and top_gene_indices.size > 0):
                plt.text(0.5, 0.5, 'No per-cell gene contributions available (after alignment/filtering)',
                        ha='center', va='center', fontsize=12)
                plt.axis('off')
            else:
                cells_cap = min(50, cell_gene_contrib.shape[0])
                cg = cell_gene_contrib[:cells_cap, :]
                cg_top = cg[:, top_gene_indices]

                if gene_names is not None and isinstance(gene_names, (list, tuple)):
                    n_names = len(gene_names)
                    ytick = []
                    for idx in top_gene_indices:
                        idx = int(idx)
                        if 0 <= idx < n_names:
                            ytick.append(gene_names[idx])
                        else:
                            ytick.append(f'gene_{idx}')
                else:
                    ytick = [f'gene_{int(i)}' for i in top_gene_indices]

                import seaborn as sns
                vmin = np.nanmin(cg_top)
                vmax = np.nanmax(cg_top)
                
                if np.isfinite(vmin) and np.isfinite(vmax) and abs(vmax - vmin) < 1e-12:
                    eps = 1e-6 if vmax == 0 else 1e-6 * abs(vmax)
                    vmin, vmax = vmax - eps, vmax + eps

                # 🔥 根据 signed 选择 colormap
                if signed:
                    # 使用红蓝发散色图，中心为0
                    abs_max = max(abs(vmin), abs(vmax))
                    sns.heatmap(
                        cg_top.T,
                        xticklabels=False,
                        yticklabels=ytick,
                        cmap='RdBu_r',  # 红蓝发散色图
                        center=0,
                        vmin=-abs_max,
                        vmax=abs_max,
                        cbar_kws={'label': 'Signed Contribution'}
                    )
                    plt.title('Cell-Gene Contribution Heatmap (Signed)\nRed=Pro-aging, Blue=Anti-aging')
                else:
                    sns.heatmap(
                        cg_top.T,
                        xticklabels=False,
                        yticklabels=ytick,
                        cmap='viridis',
                        vmin=vmin,
                        vmax=vmax,
                        cbar_kws={'label': 'Contribution Magnitude'}
                    )
                    plt.title('Cell-Gene Contribution Heatmap')
                
                plt.xlabel('Cells (First 50)')
                plt.ylabel('Top Contributing Genes')

            plt.tight_layout()
            suffix = 'signed' if signed else 'unsigned'
            plt.savefig(f"{save_dir}/gene_contributions_{donor_id}_{suffix}.png", dpi=300, bbox_inches='tight')
            plt.close()

        print(f"📊 Visualization saved: {save_dir}/{plot_type}_contributions_{donor_id}_{'signed' if signed else 'unsigned'}.png", flush=True)
        
    def _get_weights_cache_key(self, trained_epochs=None):  # 🔥 添加 trained_epochs 参数
        """
        生成缓存key
        
        Args:
            trained_epochs: 已训练的epoch数，None表示使用目标epochs
        """
        import hashlib
        import json
        
        cache_params = {
            'hidden_dim': self.params.get('hidden_dim', 8),
            'dropout': self.params.get('dropout', 0.2),
            'lr': self.params.get('lr', 1e-3),
            'batch_size': self.params.get('batch_size', 8),
            'epochs': trained_epochs if trained_epochs is not None else self.params.get('epochs', 100),  # 🔥 修改
            'max_cells': self.max_cells,
            'model_type': 'DeepSetsTabularModelAttn',
        }
        
        if hasattr(self.__class__, '_shared_train_cell_data'):
            cell_data = self.__class__._shared_train_cell_data
            n_donors = len(cell_data)
            n_genes = list(cell_data.values())[0].shape[1] if n_donors > 0 else 0
            cache_params['n_donors'] = n_donors
            cache_params['n_genes'] = n_genes

        if hasattr(self.__class__, 'is_scaled'):
            cache_params['is_scaled'] = self.__class__.is_scaled
        
        cache_str = json.dumps(cache_params, sort_keys=True)
        cache_key = hashlib.md5(cache_str.encode()).hexdigest()
        
        return cache_key, cache_params
    
    def _get_weights_cache_path(self, cache_key):
        """获取权重缓存文件路径"""
        from pathlib import Path
        cache_dir = Path(self._weights_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"deepsets_weights_{cache_key}.pth"
        
    def _save_weights_to_cache(self, cache_key, cache_params, current_epoch=None):  # 🔥 添加 current_epoch 参数
        """
        保存模型权重到缓存
        
        Args:
            cache_key: 缓存键
            cache_params: 缓存参数
            current_epoch: 当前训练到的epoch（用于中间检查点）
        """
        if self.model is None:
            print("    ⚠️ No model to save", flush=True)
            return False
        
        try:
            cache_path = self._get_weights_cache_path(cache_key)
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'scaler_mean': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
                'scaler_scale': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None,
                'max_cells': self.max_cells,
                'cache_params': cache_params,
                'trained_epochs': current_epoch if current_epoch is not None else cache_params['epochs'],  # 🔥 新增
                'timestamp': time.time(),
            }
            
            torch.save(checkpoint, cache_path)
            
            epoch_info = f" (epoch {current_epoch})" if current_epoch is not None else ""  # 🔥 新增
            print(f"    💾 Saved weights to cache{epoch_info}: {cache_path}", flush=True)
            return True
            
        except Exception as e:
            print(f"    ⚠️ Failed to save weights cache: {e}", flush=True)
            return False

    def _load_weights_from_cache(self, cache_key, continue_training=False, target_epochs=None):
        """
        从缓存加载模型权重
        
        Args:
            cache_key: 目标epochs对应的缓存键
            continue_training: 是否尝试加载部分训练的权重
            target_epochs: 目标训练epochs数
            
        Returns:
            tuple: (state_dict, trained_epochs) 或 (False, 0)
                - state_dict: 模型权重，如果未找到则为False
                - trained_epochs: 已训练的epoch数
        """
        cache_path = self._get_weights_cache_path(cache_key)
        
        # 🔥 尝试加载完整训练的权重
        if cache_path.exists():
            checkpoint = self._try_load_checkpoint(cache_path, cache_key, target_epochs)
            if checkpoint:
                return checkpoint['model_state_dict'], checkpoint['trained_epochs']
        
        # 🔥 如果启用continue_training，尝试查找部分训练的权重
        if continue_training and target_epochs is not None:
            print(f"    🔍 Searching for partially trained weights (target: {target_epochs} epochs)...", flush=True)
            
            best_checkpoint = None
            best_trained_epochs = 0
            
            # 从目标epochs-1开始递减查找
            for search_epochs in range(target_epochs - 1, 0, -1):
                search_key, search_params = self._get_weights_cache_key(trained_epochs=search_epochs)
                search_path = self._get_weights_cache_path(search_key)
                
                if search_path.exists():
                    checkpoint = self._try_load_checkpoint(search_path, search_key, search_epochs)
                    if checkpoint and checkpoint['trained_epochs'] > best_trained_epochs:
                        best_checkpoint = checkpoint
                        best_trained_epochs = checkpoint['trained_epochs']
                        print(f"    ✅ Found checkpoint at {best_trained_epochs} epochs", flush=True)
                        break  # 找到最近的就停止
            
            if best_checkpoint:
                print(f"    📂 Loading checkpoint from epoch {best_trained_epochs}, will continue to {target_epochs}", flush=True)
                return best_checkpoint['model_state_dict'], best_trained_epochs
        
        print(f"    ℹ️ No valid cache found, will train from scratch", flush=True)
        return False, 0
    
    def _try_load_checkpoint(self, cache_path, cache_key, expected_epochs):
        """
        尝试加载并验证检查点
        
        Returns:
            checkpoint dict 或 None
        """
        try:
            print(f"    📂 Checking cache: {cache_path}", flush=True)
            checkpoint = torch.load(cache_path, map_location=self.device, weights_only=False)
            
            # 验证缓存参数（除了epochs外都要匹配）
            cached_params = checkpoint.get('cache_params', {})
            _, current_params = self._get_weights_cache_key(trained_epochs=expected_epochs)
            
            key_params = ['hidden_dim', 'dropout', 'lr', 'batch_size', 'n_donors', 'n_genes', 'model_type']
            for param in key_params:
                if cached_params.get(param) != current_params.get(param):
                    print(f"    ⚠️ Cache parameter mismatch: {param} "
                        f"(cached: {cached_params.get(param)}, current: {current_params.get(param)})", flush=True)
                    return None
            
            # 加载scaler
            if checkpoint.get('scaler_mean') is not None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                self.scaler.mean_ = checkpoint['scaler_mean']
                self.scaler.scale_ = checkpoint['scaler_scale']
                self.scaler.n_features_in_ = len(checkpoint['scaler_mean'])
            
            self.max_cells = checkpoint['max_cells']
            
            trained_epochs = checkpoint.get('trained_epochs', cached_params.get('epochs', 0))
            print(f"    ✅ Valid checkpoint found: {trained_epochs} epochs trained "
                f"(saved at {time.ctime(checkpoint['timestamp'])})", flush=True)
            
            return checkpoint
            
        except Exception as e:
            print(f"    ⚠️ Failed to load checkpoint: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None
        
    def _fallback_to_linear(self, X, y):
        """回退到线性模型"""
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
        n_features = min(100, X.shape[1])
        X_subset = X.iloc[:, :n_features]
        self.model.fit(X_subset, y)
        self._features_in = X_subset.columns.tolist()
        return self

    def _prepare_scaler(self, cell_data, valid_donors):
        """准备数据标准化器"""
        train_cells_list = []
        for donor_id in valid_donors[:min(len(valid_donors), 50)]:
            if donor_id in cell_data:
                train_cells_list.append(cell_data[donor_id])
        
        if train_cells_list:
            train_cells = np.vstack(train_cells_list)
            self.scaler = StandardScaler()
            self.scaler.fit(train_cells)
            print("    ✅ StandardScaler fitted!", flush=True)
        else:
            class IdentityScaler:
                def fit(self, X): return self
                def transform(self, X): return X.astype(np.float32)
            self.scaler = IdentityScaler()

    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, **kwargs):
        """训练DeepSets模型（带增强缓存）"""
        
        print(f"    🤖 DeepSets training started...", flush=True)
        
        # 🔥 获取新参数
        cache_every_epoch = self.params.get('cache_every_epoch', None)
        continue_training = self.params.get('continue_training', False)
        target_epochs = self.params.get('epochs', 100)
        
        print(f"    📋 Training config:", flush=True)
        print(f"       - Target epochs: {target_epochs}", flush=True)
        print(f"       - Cache every: {cache_every_epoch if cache_every_epoch else 'only at end'}", flush=True)
        print(f"       - Continue training: {continue_training}", flush=True)
        
        # 检查cell data
        if not hasattr(self.__class__, '_shared_train_cell_data'):
            print("    ⚠️ No single cell data available, using linear fallback", flush=True)
            return self._fallback_to_linear(X, y)
        
        cell_data = self.__class__._shared_train_cell_data
        
        # 设置max_cells
        if hasattr(self.__class__, '_shared_max_cells'):
            self.max_cells = self.__class__._shared_max_cells
        else:
            all_donor_ids = list(cell_data.keys())
            self.max_cells = max([cell_data[donor_id].shape[0] for donor_id in all_donor_ids])
        
        print(f"    📏 Using max_cells: {self.max_cells}", flush=True)
        
        # 🔥 生成目标epochs的缓存key
        target_cache_key, target_cache_params = self._get_weights_cache_key(trained_epochs=target_epochs)
        print(f"    🔑 Target cache key: {target_cache_key}", flush=True)
        
        # 🔥 尝试加载权重（新的返回格式）
        cached_state_dict, start_epoch = self._load_weights_from_cache(
            target_cache_key, 
            continue_training=continue_training,
            target_epochs=target_epochs
        )
        
        # 准备训练数据
        train_donor_ids = X.index.tolist()
        train_ages = y if isinstance(y, pd.Series) else pd.Series(y, index=train_donor_ids)
        valid_donors = [d for d in train_donor_ids if d in cell_data]
        
        if len(valid_donors) < 5:
            print(f"    ⚠️ Too few valid donors: {len(valid_donors)}, using fallback", flush=True)
            return self._fallback_to_linear(X, train_ages)
        
        train_ages = train_ages.loc[valid_donors]
        
        # 准备scaler（如果未从缓存加载）
        if cached_state_dict is False or self.scaler is None:
            print(f"    📐 Preparing scaler...", flush=True)
            self._prepare_scaler(cell_data, valid_donors)
        
        # 创建数据集
        dataset = SingleCellDataset(cell_data, train_ages, valid_donors, self.scaler, self.max_cells)
        if dataset.n_genes == 0:
            raise ValueError("No valid training data")
        
        print(f"    📊 Dataset: {len(valid_donors)} donors, {dataset.n_genes} genes", flush=True)
        
        # 创建模型
        num_classes = 10
        self.model = DeepSetsAgePredictorAttn(
            input_dim=dataset.n_genes,
            hidden_dim=2**self.params['hidden_dim'],
            dropout=self.params['dropout'],
            num_classes=num_classes,
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"    🧠 Model: {total_params:,} parameters", flush=True)
        
        # 🔥 如果有缓存，加载权重
        if cached_state_dict is not False:
            try:
                self.model.load_state_dict(cached_state_dict)
                if start_epoch >= target_epochs:
                    print(f"    ✅ Model already trained to {start_epoch} epochs, skipping training!", flush=True)
                    return self
                else:
                    print(f"    🔄 Resuming training from epoch {start_epoch} to {target_epochs}", flush=True)
            except Exception as e:
                print(f"    ⚠️ Failed to load weights: {e}, training from scratch", flush=True)
                start_epoch = 0
        
        # 🔥 训练模型（从start_epoch开始）
        epochs_to_train = target_epochs - start_epoch
        print(f"    🏃 Training for {epochs_to_train} epochs (from {start_epoch} to {target_epochs})...", flush=True)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.params['batch_size'], 
            shuffle=True, 
            collate_fn=collate_fn, 
            num_workers=0
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])
        
        lambda_reg = 1.0
        lambda_cls = 0.5
        lambda_cons = 0.1
        
        for epoch in range(start_epoch, target_epochs):  # 🔥 从start_epoch开始
            epoch_losses = []
            self.model.train()
            
            for batch_idx, batch in enumerate(dataloader):
                try:
                    cells = batch['cells'].to(self.device)
                    ages = batch['ages'].to(self.device)
                    masks = batch['masks'].to(self.device)

                    bins = (ages.clamp(min=0, max=100) / num_classes).long().clamp(0, num_classes - 1)
                    
                    optimizer.zero_grad()
                    pred, cache = self.model(cells, masks, return_logits=True)
                    loss = F.smooth_l1_loss(pred, ages)
                    
                    loss_cls = torch.tensor(0.0, device=self.device)
                    loss_cons = torch.tensor(0.0, device=self.device)
                    if cache['logits'] is not None:
                        loss_cls = F.cross_entropy(cache['logits'], bins)
                        with torch.no_grad():
                            probs = cache['logits'].softmax(dim=-1)
                            ks = torch.arange(num_classes, device=self.device).float()
                            y_cls = (probs * ((ks + 0.5) * 100.0 / num_classes)).sum(dim=-1)
                        loss_cons = F.smooth_l1_loss(pred, y_cls)
                    
                    loss = loss * lambda_reg + loss_cls * lambda_cls + loss_cons * lambda_cons
                    loss.backward()
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                    
                except Exception as e:
                    print(f"    ⚠️ Batch {batch_idx} error: {e}", flush=True)
                    continue
            
            current_epoch = epoch + 1
            
            # 打印进度
            if current_epoch % 10 == 0 or current_epoch == target_epochs:
                avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
                print(f"    📊 Epoch {current_epoch:03d}/{target_epochs}: Loss = {avg_loss:.4f}", flush=True)
            
            # 🔥 定期保存检查点
            if cache_every_epoch is not None and current_epoch % cache_every_epoch == 0:
                checkpoint_key, checkpoint_params = self._get_weights_cache_key(trained_epochs=current_epoch)
                self._save_weights_to_cache(checkpoint_key, checkpoint_params, current_epoch=current_epoch)
        
        print("    ✅ Training completed!", flush=True)
        
        # 🔥 保存最终权重
        final_key, final_params = self._get_weights_cache_key(trained_epochs=target_epochs)
        self._save_weights_to_cache(final_key, final_params, current_epoch=target_epochs)
        
        return self


class MultiQueryAttnPooling(nn.Module):
    """
    多查询注意力池化：
    - learnable queries: [r, d]
    - keys/values: encoded cells H: [B, N, d]
    - multi-head scaled dot-product attention over cells
    - mask: [B, N] -> 对无效位置在logits上加 -inf
    返回:
        P: [B, r, d]
        attn: [B, r, heads, N] 注意力分布（可用于可解释性）
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        r: int = 4,
        dropout: float = 0.1,
        tau: float = 1.0,
        use_sigmoid_norm: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.r = r
        self.tau = tau
        self.use_sigmoid_norm = use_sigmoid_norm

        # learnable queries (r seeds)
        self.queries = nn.Parameter(torch.randn(r, d_model) * 0.02)

        # projection
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        # init
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, H, mask=None, return_attn=False):
        """
        H: [B, N, d_model]
        mask: [B, N] with 1 for valid, 0 for pad (optional)
        return_attn: whether to return attention weights

        Returns:
            P: [B, r, d_model]
            attn: [B, r, n_heads, N] if return_attn
        """
        B, N, D = H.shape
        r = self.r
        h = self.n_heads
        d = self.d_head

        # expand queries to batch: [B, r, d_model]
        Q = self.queries.unsqueeze(0).expand(B, r, D)
        # project
        Q = self.W_q(Q)                               # [B, r, D]
        K = self.W_k(H)                               # [B, N, D]
        V = self.W_v(H)                               # [B, N, D]

        # reshape to heads
        Q = Q.view(B, r, h, d).transpose(1, 2)        # [B, h, r, d]
        K = K.view(B, N, h, d).transpose(1, 2)        # [B, h, N, d]
        V = V.view(B, N, h, d).transpose(1, 2)        # [B, h, N, d]

        # attention logits: [B, h, r, N]
        logits = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
        if self.tau is not None and self.tau > 0:
            logits = logits / self.tau

        if mask is not None:
            # mask: [B, N] -> [B, 1, 1, N]
            mask_ = mask.unsqueeze(1).unsqueeze(1)    # broadcast
            logits = logits.masked_fill(mask_ == 0, float('-inf'))

        if self.use_sigmoid_norm:
            # σ归一化，防止过度尖锐；数值上更稳定（但非严格概率分布）
            weights = torch.sigmoid(logits)
            # 对被mask为0的位置，上面已设 -inf -> sigmoid( -inf ) ~ 0
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            attn = weights / denom
        else:
            attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)                      # [B, h, r, N]

        # 聚合
        P = torch.matmul(attn, V)                      # [B, h, r, d]
        P = P.transpose(1, 2).contiguous()             # [B, r, h, d]
        P = P.view(B, r, h * d)                        # [B, r, D]

        P = self.out(P)                                # [B, r, D]
        if return_attn:
            # reshape attn to [B, r, h, N] for easier reading
            return P, attn.transpose(1, 2).contiguous()
        return P, None


def _masked_mean(x, mask, dim, eps=1e-6):
    # x: [B, N, ...], mask: [B, N]
    if mask is None:
        return x.mean(dim=dim)
    w = mask.unsqueeze(-1).expand_as(x).float()
    num = (x * w).sum(dim=dim)
    den = w.sum(dim=dim).clamp_min(eps)
    return num / den

    
class DeepSetsAgePredictorAttn(nn.Module):
    """
    方案3增强版：cell_encoder + 多查询注意力池化 + 小MLP回归
    - 不依赖 max_cells, 天然置换不变
    - 支持导出注意力作为可解释性信号
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.2,
        # pooling config
        n_heads: int = 4,
        r: int = 4,
        tau: float = 1.0,
        use_sigmoid_norm: bool = True,
        # head config
        head_mode: str = "mean",  # "mean" | "flatten"
        num_classes=None,
    ):
        super().__init__()
        assert head_mode in ["mean", "flatten"]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_mode = head_mode
        self.r = r
        self.num_classes = num_classes

        # cell encoder φ
        self.cell_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # attentive pooling over cells
        self.pool = MultiQueryAttnPooling(
            d_model=output_dim,
            n_heads=n_heads,
            r=r,
            dropout=dropout,
            tau=tau,
            use_sigmoid_norm=use_sigmoid_norm,
        )

        # donor-level head ρ
        if head_mode == "mean":
            donor_in_dim = output_dim
        else:
            donor_in_dim = r * output_dim

        self.donor_head = nn.Sequential(
            nn.Linear(donor_in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        if num_classes is not None:
            self.cls_head = nn.Sequential(
                nn.Linear(donor_in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.cls_head = None

    def forward(self, cells, mask=None, return_attn=False, return_logits=False):
        """
        cells: [B, N, G]
        mask: [B, N], 1 for valid, 0 for pad
        return_attn: 返回注意力以便可解释
        """
        age_pred, cache = self.forward_with_intermediates(cells=cells, mask=mask, return_attn=return_attn)

        if self.cls_head is not None:
            donor_repr = cache['donor_repr']
            logits = self.cls_head(donor_repr)
            if return_logits:
                cache["logits"] = logits
        
        return age_pred, cache
    
    def forward_with_intermediates(self, cells, mask=None, return_attn=True):
        """
        与 forward 类似，但返回中间量，方便解释：
        返回:
            age_pred: [B]
            cache: dict {
                'H': [B, N, d],
                'P': [B, r, d],
                'attn': [B, r, heads, N],  # 若 return_attn
                'donor_repr': [B, d] 或 [B, r*d]
            }
        """
        B, N, G = cells.shape
        H = self.cell_encoder(cells.view(-1, G)).view(B, N, -1)  # [B, N, d]
        P, attn = self.pool(H, mask=mask, return_attn=return_attn)  # [B, r, d], [B, r, heads, N] or None
        if self.head_mode == "mean":
            donor_repr = P.mean(dim=1)
        else:
            donor_repr = P.reshape(B, -1)
        age_pred = self.donor_head(donor_repr).view(-1)
        cache = {"H": H, "P": P, "donor_repr": donor_repr}
        if return_attn:
            cache["attn"] = attn  # [B, r, heads, N]
        return age_pred, cache

    @torch.no_grad()
    def get_cell_contributions_attn(self, cells, mask=None, method="attention", reduce_heads="mean"):
        """
        每细胞贡献（基于注意力或激活幅度的显式方法，免梯度）
        Args:
            cells: [B, N, G]
            mask: [B, N] (0/1)
            method: 'attention' | 'activation'
                - 'attention': 使用注意力权重作为细胞重要性；对 r 个查询与多头聚合
                - 'activation': 使用编码特征 H 的范数衡量（与顺序无关的启发式）
            reduce_heads: 'mean' | 'max' | 'sum' 头的聚合方式

        Returns:
            dict: {
                'cell_contributions': [B, N],
                'age_pred': [B],
                'aux': {
                    'attn_per_query': [B, r, N] (method='attention'时提供),
                    'H_norm': [B, N] (method='activation'时提供)
                }
            }
        """
        self.eval()
        age_pred, cache = self.forward_with_intermediates(cells, mask=mask, return_attn=True)
        B, N, d = cache["H"].shape

        aux = {}
        if method == "attention":
            assert "attn" in cache and cache["attn"] is not None, "attn not available"
            # attn: [B, r, heads, N]
            attn = cache["attn"]
            if reduce_heads == "mean":
                attn_r = attn.mean(dim=2)                # [B, r, N]
            elif reduce_heads == "max":
                attn_r, _ = attn.max(dim=2)             # [B, r, N]
            elif reduce_heads == "sum":
                attn_r = attn.sum(dim=2)                # [B, r, N]（若使用sigmoid归一化，可能>1）
            else:
                raise ValueError("reduce_heads must be 'mean'|'max'|'sum'")

            # 跨查询聚合到每细胞得分
            # 方案A：对 r 求和或均值
            cell_scores = attn_r.mean(dim=1)            # [B, N]
            # 若希望突出“任一查询强关注”的细胞，也可用 max 聚合：
            # cell_scores, _ = attn_r.max(dim=1)

            # mask无效位置为0
            if mask is not None:
                cell_scores = cell_scores * mask

            aux["attn_per_query"] = attn_r.detach().cpu().numpy()
            cell_contributions = cell_scores

        elif method == "activation":
            # 用 H 的L2范数衡量每细胞激活强度
            H = cache["H"]                               # [B, N, d]
            H_norm = torch.norm(H, dim=-1)              # [B, N]
            if mask is not None:
                H_norm = H_norm * mask
            aux["H_norm"] = H_norm.detach().cpu().numpy()
            # 可选择归一化到[0,1]
            cell_contributions = H_norm

        else:
            raise ValueError("Unsupported method for attention-based contribution")

        return {
            "cell_contributions": cell_contributions.detach().cpu().numpy(),
            "age_pred": age_pred.detach().cpu().numpy(),
            "aux": aux,
        }

    def get_cell_contributions(self, cells, mask=None, method="gradient", target="H", 
                            normalize=True, steps_ig=32, signed=True):
        """
        每细胞贡献（梯度家族方法）
        
        Args:
            method: 'gradient' | 'grad_input' | 'integrated_gradient'
            target: 'H' | 'cells'
            normalize: 是否归一化
            steps_ig: IG步数
            signed: 🔥 NEW: 是否保留符号（True=带符号，False=只看幅度）
        
        Returns:
            dict: {
                'cell_contributions': [B, N],  # 如果signed=True，可能有负值
                'age_pred': [B],
                'grads_target': ...
            }
        """
        self.eval()
        B, N, G = cells.shape

        # ============================================
        # Gradient 和 Grad×Input 方法
        # ============================================
        if method in ["gradient", "grad_input"]:
            cells_clone = cells.clone().detach().requires_grad_(True)
            age_pred, cache = self.forward_with_intermediates(cells_clone, mask=mask, return_attn=False)
            
            if target == "H":
                H = cache["H"]  # [B, N, d]
                H.retain_grad()
                loss = age_pred.sum()
                loss.backward()
                grads = H.grad  # [B, N, d]
                
                if method == "gradient":
                    if signed:
                        # 🔥 保留符号：使用平均值而不是范数
                        contrib = grads.mean(dim=-1)  # [B, N]
                    else:
                        # 原来的方式：只看幅度
                        contrib = torch.norm(grads, dim=-1)  # [B, N]
                        
                elif method == "grad_input":
                    grad_input = grads * H.detach()  # [B, N, d]
                    if signed:
                        # 🔥 保留符号：求和而不是范数
                        contrib = grad_input.sum(dim=-1)  # [B, N]
                    else:
                        contrib = torch.norm(grad_input, dim=-1)  # [B, N]
                
            elif target == "cells":
                loss = age_pred.sum()
                loss.backward()
                grads = cells_clone.grad  # [B, N, G]
                
                if method == "gradient":
                    if signed:
                        contrib = grads.mean(dim=-1)  # [B, N]
                    else:
                        contrib = torch.norm(grads, dim=-1)
                        
                elif method == "grad_input":
                    grad_input = grads * cells_clone.detach()
                    if signed:
                        contrib = grad_input.sum(dim=-1)  # [B, N]
                    else:
                        contrib = torch.norm(grad_input, dim=-1)
            else:
                raise ValueError("target must be 'H' or 'cells'")

            # 应用mask
            if mask is not None:
                contrib = contrib * mask

            # 🔥 归一化方式也需要调整
            if normalize:
                if signed:
                    # 对于带符号的值，使用标准化而不是min-max
                    contrib_mean = contrib.mean(dim=1, keepdim=True)
                    contrib_std = contrib.std(dim=1, keepdim=True) + 1e-6
                    contrib = (contrib - contrib_mean) / contrib_std
                else:
                    # 无符号：min-max归一化
                    cmin = contrib.amin(dim=1, keepdim=True)
                    cmax = contrib.amax(dim=1, keepdim=True)
                    contrib = (contrib - cmin) / (cmax - cmin + 1e-6)

            return {
                "cell_contributions": contrib.detach().cpu().numpy(),
                "age_pred": age_pred.detach().cpu().numpy(),
                "grads_target": grads.detach().cpu().numpy(),
            }

        # ============================================
        # Integrated Gradients 方法
        # ============================================
        elif method == "integrated_gradient":
            
            if target == "H":
                with torch.no_grad():
                    baseline_cells = torch.zeros_like(cells)
                    if mask is not None:
                        baseline_cells = baseline_cells * mask.unsqueeze(-1)
                    
                    _, baseline_cache = self.forward_with_intermediates(
                        baseline_cells, mask=mask, return_attn=False
                    )
                    baseline_H = baseline_cache["H"].detach()
                    
                    _, actual_cache = self.forward_with_intermediates(
                        cells.detach(), mask=mask, return_attn=False
                    )
                    actual_H = actual_cache["H"].detach()
                
                alphas = torch.linspace(0, 1, steps_ig + 1, device=cells.device, dtype=cells.dtype)
                ig_accum_H = torch.zeros_like(actual_H)
                
                for i in range(steps_ig + 1):
                    alpha = alphas[i]
                    cells_interp = baseline_cells + alpha * (cells - baseline_cells)
                    cells_interp.requires_grad_(True)
                    
                    age_pred_interp, cache_interp = self.forward_with_intermediates(
                        cells_interp, mask=mask, return_attn=False
                    )
                    H_interp = cache_interp["H"]
                    H_interp.retain_grad()
                    
                    loss_interp = age_pred_interp.sum()
                    loss_interp.backward()
                    grads_H = H_interp.grad
                    
                    ig_accum_H += grads_H.detach()
                
                avg_grads_H = ig_accum_H / (steps_ig + 1)
                ig_H = (actual_H - baseline_H) * avg_grads_H  # [B, N, d]
                
                if signed:
                    # 🔥 保留符号：求和
                    contrib = ig_H.sum(dim=-1)  # [B, N]
                else:
                    # 只看幅度：范数
                    contrib = torch.norm(ig_H, dim=-1)
                    
            elif target == "cells":
                baseline = torch.zeros_like(cells)
                if mask is not None:
                    baseline = baseline * mask.unsqueeze(-1)
                
                alphas = torch.linspace(0, 1, steps_ig + 1, device=cells.device, dtype=cells.dtype)
                ig_accum = torch.zeros_like(cells)

                for i in range(steps_ig + 1):
                    alpha = alphas[i]
                    x = baseline + alpha * (cells - baseline)
                    x.requires_grad_(True)
                    
                    age_pred_a, _ = self.forward_with_intermediates(x, mask=mask, return_attn=False)
                    loss_a = age_pred_a.sum()
                    grads_a = torch.autograd.grad(loss_a, x, create_graph=False, retain_graph=False)[0]
                    ig_accum += grads_a.detach()

                avg_grads = ig_accum / (steps_ig + 1)
                ig = (cells - baseline) * avg_grads  # [B, N, G]
                
                if signed:
                    # 🔥 保留符号：求和
                    contrib = ig.sum(dim=-1)  # [B, N]
                else:
                    # 只看幅度：范数或绝对值
                    contrib = torch.norm(ig, dim=-1)
            
            else:
                raise ValueError("target must be 'H' or 'cells'")

            # 应用mask
            if mask is not None:
                contrib = contrib * mask

            # 归一化
            if normalize:
                if signed:
                    contrib_mean = contrib.mean(dim=1, keepdim=True)
                    contrib_std = contrib.std(dim=1, keepdim=True) + 1e-6
                    contrib = (contrib - contrib_mean) / contrib_std
                else:
                    cmin = contrib.amin(dim=1, keepdim=True)
                    cmax = contrib.amax(dim=1, keepdim=True)
                    contrib = (contrib - cmin) / (cmax - cmin + 1e-6)

            with torch.no_grad():
                age_pred, _ = self.forward_with_intermediates(cells.detach(), mask=mask, return_attn=False)

            return {
                "cell_contributions": contrib.detach().cpu().numpy(),
                "age_pred": age_pred.detach().cpu().numpy(),
            }
        
        else:
            raise ValueError(f"Unsupported method: {method}")

    def get_gene_contributions(self, cells, mask=None, method="gradient", per_cell=False, 
                            normalize=True, steps_ig=32, signed=True):
        """
        每基因贡献（对输入 cells 求导）
        
        Args:
            method: 'gradient' | 'grad_input' | 'integrated_gradient'
            per_cell: True 时额外返回 cell_gene_contributions [B, N, G]
            normalize: 归一化
            steps_ig: IG 步数
            signed: 🔥 NEW: 是否保留符号（True=带符号，False=只看幅度）

        Returns:
            dict: {
                'gene_contributions': [B, G]  # 可能有负值（如果signed=True）
                'cell_gene_contributions': [B, N, G] 或 None
                'age_pred': [B]
            }
        """
        self.eval()
        B, N, G = cells.shape

        if method != "integrated_gradient":
            cells = cells.clone().detach().requires_grad_(True)
            age_pred, _ = self.forward_with_intermediates(cells, mask=mask, return_attn=False)
            loss = age_pred.sum()
            loss.backward()
            grads = cells.grad  # [B, N, G]

            if method == "gradient":
                if signed:
                    # 🔥 保留符号：直接使用梯度
                    contrib = grads  # [B, N, G]
                else:
                    # 只看幅度：使用绝对值
                    contrib = torch.abs(grads)
                    
            elif method == "grad_input":
                grad_input = grads * cells
                if signed:
                    # 🔥 保留符号
                    contrib = grad_input  # [B, N, G]
                else:
                    # 只看幅度
                    contrib = torch.abs(grad_input)
            else:
                raise ValueError("Unsupported method")

            if mask is not None:
                contrib = contrib * mask.unsqueeze(-1)

            # 保存 per-cell 版本（归一化前）
            contrib_per_cell_raw = contrib.clone()

            # 对细胞维度求和得到基因贡献
            gene_contrib = contrib.sum(dim=1)  # [B, G]

            # 归一化
            if normalize:
                if signed:
                    # 🔥 标准化（保留符号）
                    gc_mean = gene_contrib.mean(dim=1, keepdim=True)
                    gc_std = gene_contrib.std(dim=1, keepdim=True) + 1e-6
                    gene_contrib = (gene_contrib - gc_mean) / gc_std
                    
                    # per_cell 归一化
                    if per_cell:
                        # 对每个细胞的基因向量做标准化
                        cpc_mean = contrib_per_cell_raw.mean(dim=-1, keepdim=True)
                        cpc_std = contrib_per_cell_raw.std(dim=-1, keepdim=True) + 1e-6
                        contrib_per_cell_normalized = (contrib_per_cell_raw - cpc_mean) / cpc_std
                    else:
                        contrib_per_cell_normalized = None
                else:
                    # Min-Max 归一化（无符号）
                    cmin = gene_contrib.amin(dim=1, keepdim=True)
                    cmax = gene_contrib.amax(dim=1, keepdim=True)
                    gene_contrib = (gene_contrib - cmin) / (cmax - cmin + 1e-6)

                    if per_cell:
                        cmin_pc = contrib_per_cell_raw.amin(dim=-1, keepdim=True)
                        cmax_pc = contrib_per_cell_raw.amax(dim=-1, keepdim=True)
                        contrib_per_cell_normalized = (contrib_per_cell_raw - cmin_pc) / (cmax_pc - cmin_pc + 1e-6)
                    else:
                        contrib_per_cell_normalized = None
            else:
                contrib_per_cell_normalized = contrib_per_cell_raw if per_cell else None

            return {
                "gene_contributions": gene_contrib.detach().cpu().numpy(),
                "cell_gene_contributions": contrib_per_cell_normalized.detach().cpu().numpy() if per_cell else None,
                "age_pred": age_pred.detach().cpu().numpy(),
            }

        # Integrated Gradients
        baseline = torch.zeros_like(cells)
        alphas = torch.linspace(0, 1, steps_ig + 1, device=cells.device, dtype=cells.dtype)
        ig_accum = torch.zeros_like(cells)

        for i in range(steps_ig + 1):
            a = alphas[i]
            x = baseline + a * (cells - baseline)
            x.requires_grad_(True)
            age_pred_a, _ = self.forward_with_intermediates(x, mask=mask, return_attn=False)
            loss_a = age_pred_a.sum()
            grads_a = torch.autograd.grad(loss_a, x, create_graph=False, retain_graph=False)[0]
            ig_accum += grads_a

        avg_grads = ig_accum / (steps_ig + 1)
        
        if signed:
            # 🔥 保留符号
            ig = (cells - baseline) * avg_grads  # [B, N, G]
        else:
            # 只看幅度
            ig = torch.abs((cells - baseline) * avg_grads)

        if mask is not None:
            ig = ig * mask.unsqueeze(-1)

        ig_per_cell_raw = ig.clone()
        gene_contrib = ig.sum(dim=1)  # [B, G]

        # 归一化
        if normalize:
            if signed:
                gc_mean = gene_contrib.mean(dim=1, keepdim=True)
                gc_std = gene_contrib.std(dim=1, keepdim=True) + 1e-6
                gene_contrib = (gene_contrib - gc_mean) / gc_std
                
                if per_cell:
                    cpc_mean = ig_per_cell_raw.mean(dim=-1, keepdim=True)
                    cpc_std = ig_per_cell_raw.std(dim=-1, keepdim=True) + 1e-6
                    ig_per_cell_normalized = (ig_per_cell_raw - cpc_mean) / cpc_std
                else:
                    ig_per_cell_normalized = None
            else:
                cmin = gene_contrib.amin(dim=1, keepdim=True)
                cmax = gene_contrib.amax(dim=1, keepdim=True)
                gene_contrib = (gene_contrib - cmin) / (cmax - cmin + 1e-6)

                if per_cell:
                    cmin_pc = ig_per_cell_raw.amin(dim=-1, keepdim=True)
                    cmax_pc = ig_per_cell_raw.amax(dim=-1, keepdim=True)
                    ig_per_cell_normalized = (ig_per_cell_raw - cmin_pc) / (cmax_pc - cmin_pc + 1e-6)
                else:
                    ig_per_cell_normalized = None
        else:
            ig_per_cell_normalized = ig_per_cell_raw if per_cell else None

        with torch.no_grad():
            age_pred, _ = self.forward_with_intermediates(cells.detach(), mask=mask, return_attn=False)

        return {
            "gene_contributions": gene_contrib.detach().cpu().numpy(),
            "cell_gene_contributions": ig_per_cell_normalized.detach().cpu().numpy() if per_cell else None,
            "age_pred": age_pred.detach().cpu().numpy(),
        }


class scNETTabularModel(AbstractModel):
    """
    scNET架构的AutoGluon集成模型
    支持L-R矩阵、Z-score标准化和重构损失
    """
    
    _weights_cache_dir = "./results_debug/scnet_weights"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.scaler = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_cells = None
        self._features_in = None
        self._model_type = 'scNETTabularModel'
    
    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            ignored_type_group_raw=[],
            ignored_type_group_special=[],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
    
    def _set_default_params(self):
        """设置默认超参数"""
        default_params = {
            'hidden_dim': 128,
            'n_layers': 3,
            'dropout': 0.2,
            'lr': 1e-3,
            'batch_size': 4,
            'epochs': 100,
            'use_reconstruction': True,
            'lambda_recon': 0.1,
            'n_top_genes': 2000,
        }
        for key, value in default_params.items():
            self._set_default_param_value(key, value)
    
    def _get_default_resources(self):
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return [2, 0.5]
        else:
            return [4, 0]
    
    def _get_default_stopping_metric(self):
        return 'root_mean_squared_error'
    
    def _more_tags(self):
        return {
            'can_use_gpu': torch.cuda.is_available(),
            'requires_fit': True,
            'requires_positive_X': False,
        }
    
    def _get_default_ag_args_fit(self, **kwargs) -> dict:
        try:
            default_ag_args_fit = super()._get_default_ag_args_fit(**kwargs)
            if not isinstance(default_ag_args_fit, dict):
                default_ag_args_fit = {}
        except:
            default_ag_args_fit = {}
        return default_ag_args_fit
    
    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, **kwargs):
        """训练scNET模型"""
        
        print(f"    🧬 scNET training started...", flush=True)
        
        # 🔥 关键：获取共享的单细胞数据和L-R矩阵
        if not hasattr(self.__class__, '_shared_train_cell_data'):
            print("    ⚠️ No single cell data available for scNET, using linear fallback", flush=True)
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=1.0)
            n_features = min(100, X.shape[1])
            X_subset = X.iloc[:, :n_features]
            self.model.fit(X_subset, y)
            self._features_in = X_subset.columns.tolist()
            return self
        
        cell_data = self.__class__._shared_train_cell_data
        gene_connection_matrix = getattr(self.__class__, '_shared_gene_connection_matrix', None)
        expression_gene_names = getattr(self.__class__, '_shared_expression_gene_names', None)
        ligand_names = getattr(self.__class__, '_shared_ligand_names', None)
        receptor_names = getattr(self.__class__, '_shared_receptor_names', None)
        
        # 设置max_cells
        if hasattr(self.__class__, '_shared_max_cells'):
            self.max_cells = self.__class__._shared_max_cells
        else:
            all_donor_ids = list(cell_data.keys())
            self.max_cells = max([cell_data[donor_id].shape[0] for donor_id in all_donor_ids])
        
        print(f"    📏 scNET using max_cells: {self.max_cells}", flush=True)
        
        # 🔥 生成缓存key并检查缓存
        cache_key, cache_params = self._get_weights_cache_key()
        print(f"    🔑 Cache key: {cache_key}", flush=True)
        
        cached_state_dict = self._load_weights_from_cache(cache_key)
        
        # 准备训练数据
        train_donor_ids = X.index.tolist()
        train_ages = y if isinstance(y, pd.Series) else pd.Series(y, index=train_donor_ids)
        
        valid_donors = [d for d in train_donor_ids if d in cell_data]
        if len(valid_donors) < 5:
            print(f"    ⚠️ Too few valid donors: {len(valid_donors)}, using Ridge fallback", flush=True)
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=1.0)
            n_features = min(100, X.shape[1])
            X_subset = X.iloc[:, :n_features]
            self.model.fit(X_subset, train_ages)
            self._features_in = X_subset.columns.tolist()
            return self
        
        train_ages = train_ages.loc[valid_donors]
        
        # 准备StandardScaler（如果需要）
        if cached_state_dict is False or self.scaler is None:
            print(f"    📐 Preparing scaler for {len(valid_donors)} donors...", flush=True)
            from sklearn.preprocessing import StandardScaler
            train_cells_list = []
            for donor_id in valid_donors[:min(len(valid_donors), 50)]:
                if donor_id in cell_data:
                    train_cells_list.append(cell_data[donor_id])
            
            if train_cells_list:
                train_cells = np.vstack(train_cells_list)
                self.scaler = StandardScaler()
                self.scaler.fit(train_cells)
                print("    ✅ StandardScaler fitted!", flush=True)
            else:
                class IdentityScaler:
                    def fit(self, X): return self
                    def transform(self, X): return X.astype(np.float32)
                self.scaler = IdentityScaler()
        
        # 创建数据集
        dataset = SingleCellDataset(cell_data, train_ages, valid_donors, self.scaler, self.max_cells)
        if dataset.n_genes == 0:
            raise ValueError("No valid training data for scNET")
        
        print(f"    📊 Dataset: {len(valid_donors)} donors, {dataset.n_genes} genes", flush=True)
        
        # 🔥 创建scNET模型
        num_classes = 10
        self.model = scNETAgePredictorFast( # scNETAgePredictor
            n_genes=dataset.n_genes,
            hidden_dim=self.params['hidden_dim'],
            n_layers=self.params['n_layers'],
            dropout=self.params['dropout'],
            gene_connection_matrix=gene_connection_matrix,
            expression_gene_names=expression_gene_names,
            ligand_names=ligand_names,
            receptor_names=receptor_names,
            n_top_genes=self.params['n_top_genes'],
            lambda_recon=self.params['lambda_recon'],
            use_reconstruction=self.params['use_reconstruction'],
            num_classes=num_classes,
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"    🧠 scNET model: {total_params:,} parameters", flush=True)
        
        # 🔥 如果有缓存，加载权重
        if cached_state_dict is not False:
            try:
                self.model.load_state_dict(cached_state_dict)
                print("    ✅ Loaded model weights from cache, skipping training!", flush=True)
                return self
            except Exception as e:
                print(f"    ⚠️ Failed to load cached weights: {e}", flush=True)
                print("    🔄 Will train from scratch...", flush=True)
        
        # 🔥 训练模型
        print(f"    🏃 No valid cache found, training for {self.params['epochs']} epochs...", flush=True)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.params['batch_size'], 
            shuffle=True, 
            collate_fn=collate_fn, 
            num_workers=0
        )
        
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params['lr'], weight_decay=1e-4)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])
        
        lambda_reg = 1.
        lambda_cls = .5
        lambda_cons = .1

        for epoch in range(self.params['epochs']):
            epoch_losses = []
            epoch_recon_losses = []
            self.model.train()
            
            for batch_idx, batch in enumerate(dataloader):
                try:
                    cells = batch['cells'].to(self.device)
                    ages = batch['ages'].to(self.device)
                    masks = batch['masks'].to(self.device)

                    bins = (ages.clamp(min=0, max=100) / num_classes).long().clamp(0, num_classes - 1)
                    
                    optimizer.zero_grad()
                    
                    # scNET可能返回tuple (predictions, reconstruction_loss)
                    pred, cache = self.model(cells, masks, return_logits=True)
                    loss = F.smooth_l1_loss(pred, ages)

                    loss_cls = torch.tensor(.0, device=self.device)
                    loss_cons = torch.tensor(.0, device=self.device)
                    if cache.get('logits') is not None:
                        loss_cls = F.cross_entropy(cache['logits'], bins)
                        with torch.no_grad():
                            probs = cache['logits'].softmax(dim=-1)
                            ks = torch.arange(num_classes, device=self.device).float()
                            y_cls = (probs * ((ks + .5) * 100. / num_classes)).sum(dim=-1)
                        loss_cons = F.smooth_l1_loss(pred, y_cls)
                    loss = loss * lambda_reg + loss_cls * lambda_cls + loss_cons * lambda_cons

                    if cache.get('recon_loss') is not None:
                        recon_loss = cache['recon_loss'] if not isinstance(cache['recon_loss'], list) else torch.stack(cache['recon_loss']).mean()
                        if recon_loss.item() > 0:
                            total_loss = loss + self.params['lambda_recon'] * recon_loss
                            epoch_recon_losses.append(recon_loss.item())
                    else:
                        recon_loss = torch.tensor(0.0, device=self.device)
                        total_loss = loss
                    
                    total_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                    
                except Exception as e:
                    print(f"    ⚠️ Batch {batch_idx} error: {e}", flush=True)
                    continue
            
            if (epoch + 1) % 10 == 0:
                avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
                avg_recon = np.mean(epoch_recon_losses) if epoch_recon_losses else 0.0
                print(f"    📊 Epoch {epoch+1:02d}: Loss = {avg_loss:.4f}, Recon = {avg_recon:.4f}", flush=True)
        
        print("    ✅ scNET training completed!", flush=True)
        
        # 保存权重到缓存
        self._save_weights_to_cache(cache_key, cache_params)
        
        return self
    
    def _predict_proba(self, X, max_cells=None, **kwargs):
        predictions = self._predict(X, max_cells=max_cells, **kwargs)
        return predictions.flatten()
    
    def _predict(self, X, max_cells=None, **kwargs):
        """预测函数"""
        
        if self.model is None:
            return np.full(len(X), 50.0)
        
        # sklearn模型fallback
        if hasattr(self.model, 'predict') and hasattr(self.model, 'coef_'):
            if hasattr(self, '_features_in'):
                X_subset = X[self._features_in]
                return self.model.predict(X_subset)
            else:
                n_features = min(100, X.shape[1])
                X_subset = X.iloc[:, :n_features]
                return self.model.predict(X_subset)
        
        # scNET模型
        if not hasattr(self.__class__, '_shared_cell_data'):
            return np.full(len(X), 50.0)
        
        cell_data = self.__class__._shared_cell_data
        donor_ids = X.index.tolist()
        
        valid_donors = [d for d in donor_ids if d in cell_data]
        print(f'Predicting {len(valid_donors)} donors with scNET...', flush=True)
        
        if len(valid_donors) == 0:
            return np.full(len(donor_ids), 50.0)
        
        # 创建数据集
        dummy_ages = pd.Series([0.0] * len(valid_donors), index=valid_donors)
        dataset = SingleCellDataset(
            cell_data, 
            dummy_ages, 
            valid_donors, 
            self.scaler, 
            self.max_cells if max_cells is None else max_cells
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=1,
            shuffle=False, 
            collate_fn=collate_fn, 
            num_workers=0
        )
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                try:
                    cells = batch['cells'].to(self.device)
                    masks = batch['masks'].to(self.device)
                    
                    output = self.model(cells, masks)
                    
                    # 处理可能的tuple返回
                    if isinstance(output, tuple):
                        pred = output[0]
                    else:
                        pred = output
                    
                    predictions.extend(pred.cpu().numpy())
                except Exception as e:
                    print(f"    ⚠️ Prediction error: {e}", flush=True)
                    continue
        
        # 构建结果
        pred_dict = dict(zip(valid_donors, predictions))
        final_preds = [pred_dict.get(d, 50.0) for d in donor_ids]
        
        return np.array(final_preds)
    
    # ========== 缓存相关方法（复用DeepSetsTabularModelAttn的逻辑）==========
    
    def _get_weights_cache_key(self):
        import hashlib
        import json
        
        cache_params = {
            'hidden_dim': self.params.get('hidden_dim', 128),
            'n_layers': self.params.get('n_layers', 3),
            'dropout': self.params.get('dropout', 0.2),
            'lr': self.params.get('lr', 1e-3),
            'batch_size': self.params.get('batch_size', 4),
            'epochs': self.params.get('epochs', 100),
            'use_reconstruction': self.params.get('use_reconstruction', True),
            'lambda_recon': self.params.get('lambda_recon', 0.1),
            'max_cells': self.max_cells,
            'model_type': 'scNETTabularModel',
        }
        
        if hasattr(self.__class__, '_shared_train_cell_data'):
            cell_data = self.__class__._shared_train_cell_data
            n_donors = len(cell_data)
            n_genes = list(cell_data.values())[0].shape[1] if n_donors > 0 else 0
            cache_params['n_donors'] = n_donors
            cache_params['n_genes'] = n_genes
        
        cache_str = json.dumps(cache_params, sort_keys=True)
        cache_key = hashlib.md5(cache_str.encode()).hexdigest()
        
        return cache_key, cache_params
    
    def _get_weights_cache_path(self, cache_key):
        from pathlib import Path
        cache_dir = Path(self._weights_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"scnet_weights_{cache_key}.pth"
    
    def _save_weights_to_cache(self, cache_key, cache_params):
        if self.model is None:
            return False
        
        try:
            import time
            cache_path = self._get_weights_cache_path(cache_key)
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'scaler_mean': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
                'scaler_scale': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None,
                'max_cells': self.max_cells,
                'cache_params': cache_params,
                'timestamp': time.time(),
            }
            
            torch.save(checkpoint, cache_path)
            print(f"    💾 Saved scNET weights to cache: {cache_path}", flush=True)
            return True
        except Exception as e:
            print(f"    ⚠️ Failed to save weights: {e}", flush=True)
            return False
    
    def _load_weights_from_cache(self, cache_key):
        cache_path = self._get_weights_cache_path(cache_key)
        
        if not cache_path.exists():
            return False
        
        try:
            import time
            print(f"    📂 Loading scNET weights from cache: {cache_path}", flush=True)
            checkpoint = torch.load(cache_path, map_location=self.device, weights_only=False)
            
            # 验证参数
            cached_params = checkpoint.get('cache_params', {})
            current_key, current_params = self._get_weights_cache_key()
            
            key_params = ['hidden_dim', 'n_layers', 'dropout', 'epochs', 'n_donors', 'n_genes']
            for param in key_params:
                if cached_params.get(param) != current_params.get(param):
                    print(f"    ⚠️ Cache mismatch: {param}", flush=True)
                    return False
            
            # 加载scaler
            if checkpoint['scaler_mean'] is not None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                self.scaler.mean_ = checkpoint['scaler_mean']
                self.scaler.scale_ = checkpoint['scaler_scale']
                self.scaler.n_features_in_ = len(checkpoint['scaler_mean'])
            
            self.max_cells = checkpoint['max_cells']
            
            print(f"    ✅ Loaded cached scNET weights (saved at {time.ctime(checkpoint['timestamp'])})", flush=True)
            
            return checkpoint['model_state_dict']
        except Exception as e:
            print(f"    ⚠️ Failed to load cache: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False