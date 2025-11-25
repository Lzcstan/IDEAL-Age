import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LassoCV
from scipy.stats import linregress, spearmanr

from autogluon.tabular import TabularPredictor

def get_pseudobulk(adata, donor_list, use_raw=True):
    pseudo_df = pd.DataFrame(columns=adata.var_names)
    for donor in donor_list:
        if use_raw:
            select_data = adata[adata.obs["donor_id"] == donor, :].layers["counts"]
        else:
            select_data = adata[adata.obs["donor_id"] == donor, :].X
        pseudo_df.loc[donor] = select_data.mean(axis=0).tolist()[0]
    return pseudo_df


class BenchmarkItem:

    def __init__(self, predicted_age, actual_age):
        self.predicted_age = predicted_age
        self.actual_age = actual_age
        (
            self.slope,
            self.pcc,
            self.p_value_pearson,
            self.rho,
            self.p_value_spearman,
            self.mae,
            self.rae,
            self.mean_log2_ratio,
            self.std_dev_log2_ratio,
        ) = self.predict(predicted_age, actual_age)

    def predict(self, predicted_age, actual_age):
        # --- 1. 计算皮尔逊相关系数 (Pearson Correlation) ---
        slope, _, pcc, p_value_pearson, _ = linregress(predicted_age, actual_age)
        print("--- 皮尔逊相关性分析 ---")
        print(f"斜率: {slope:.3f}")
        print(f"皮尔逊相关系数 (PCC): {pcc:.3f}")
        print(f"P-value: {p_value_pearson:.3e}")

        # --- 2. 计算斯皮尔曼等级相关系数 (Spearman Correlation) ---
        # 注意：斯皮尔曼相关的"R-value"通常指的就是相关系数rho
        rho, p_value_spearman = spearmanr(predicted_age, actual_age)
        print("--- 斯皮尔曼相关性分析 ---")
        print(f"斯皮尔曼相关系数 (rho): {rho:.3f}")
        print(f"P-value: {p_value_spearman:.3e}")

        # --- 3. 计算平均绝对误差 (Mean Absolute Error - MAE) ---
        # MAE = (1/n) * Σ|actual - predicted|
        mae = np.mean(np.abs(actual_age - predicted_age))
        print("--- 误差计算 ---")
        print(f"平均绝对误差 (MAE): {mae:.3f}")

        # --- 4. 计算相对绝对误差 (Relative Absolute Error - RAE) ---
        # RAE = Σ|actual - predicted| / Σ|actual - mean(actual)|
        numerator = np.sum(np.abs(actual_age - predicted_age))
        denominator = np.sum(np.abs(actual_age - np.mean(actual_age)))
        rae = numerator / denominator

        print(f"相对绝对误差 (RAE): {rae:.3f}")

        # --- 5. 计算 log2(预测年龄 / 实际年龄) ---
        # 1. 首先计算比率
        ratio = predicted_age / actual_age
        # 2. 然后计算该比率的 log2 值
        log2_ratio = np.log2(ratio)
        # 3. 计算这些 log2 比率的平均值
        mean_log2_ratio = np.mean(log2_ratio)
        # 4. 计算这些 log2 比率的标准差
        std_dev_log2_ratio = np.std(log2_ratio)

        print("--- 对数比率分析 ---")
        # 打印所有样本的log2比率值，以便查看
        # print(f"每个样本的 Log2 比率: {np.round(log2_ratio, 4)}")
        print(f"Log2(预测年龄/实际年龄) 的平均值: {mean_log2_ratio:.4f}")
        print(f"Log2(预测年龄/实际年龄) 的标准差: {std_dev_log2_ratio:.4f}")

        return (
            slope,
            pcc,
            p_value_pearson,
            rho,
            p_value_spearman,
            mae,
            rae,
            mean_log2_ratio,
            std_dev_log2_ratio,
        )

    def scatter_plot(self, color, title, save_dir=None, filename=None):
        plt.figure(figsize=(6, 6))
        sns.regplot(
            x=self.actual_age,
            y=self.predicted_age,
            ci=95,
            line_kws={"color": color, "label": "Regression line"},
            scatter_kws={"color": color, "s": 10, "alpha": 0.6},
        )
        plt.plot(
            [self.actual_age.min(), self.actual_age.max()],
            [self.actual_age.min(), self.actual_age.max()],
            "--",
            color="gray",
            lw=2,
            alpha=0.6,
            label="True line",
        )
        plt.xlabel("True Age")
        plt.ylabel("Predicted Age")
        plt.title(title, fontsize=16)
        text = f"Slope = {self.slope:.3f}\n"
        text += f"Pearson PCC = {self.pcc:.3f}\n"
        text += f"Pearson P value = {self.p_value_pearson:.2e}\n"
        text += f"Spearman Rho = {self.rho:.3f}\n"
        text += f"Spearman P value = {self.p_value_spearman:.2e}"
        plt.text(
            0.05,
            0.95,
            text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
        )
        plt.legend(loc="lower right")
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=300)
            plt.savefig(os.path.join(save_dir, f"{filename}.pdf"), dpi=300)
        plt.show()

    def residual_plot(self, color, title, save_dir=None, filename=None):
        residual = self.actual_age - self.predicted_age
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(
            x=self.actual_age,
            y=residual,
            color=color,
            s=10,
            alpha=0.6,
        )
        ax.axhline(y=0, color="gray", linestyle="--", lw=2, alpha=0.6)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Actual Age")
        ax.set_ylabel("Residuals")
        text = f"MAD = {self.mae:.3f}\nRAE = {self.rae:.3f}\n"
        text += f"Mean Log2 Ratio = {self.mean_log2_ratio:.3f}\n"
        text += f"Std Dev Log2 Ratio = {self.std_dev_log2_ratio:.3f}"
        fig.text(
            1.02,
            0.5,
            text,
            ha="left",
            va="center",
            fontsize=10,
        )
        plt.subplots_adjust(right=0.8)
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=300)
            plt.savefig(os.path.join(save_dir, f"{filename}.pdf"), dpi=300)
        plt.show()

if __name__ == '__main__':
    os.chdir("/personal/ImmAge")
    print(os.getcwd())

    adata = sc.read_h5ad("h5ad/2024-A/protein_genes_4sets.h5ad")
    print(adata)

    print(adata.X[:10, :10])
    print(adata.layers["counts"][:10, :10])

    donor_info = pd.read_csv("info/donor_info_4set.csv", index_col=0)
    print(donor_info)

    print(adata.obs.columns)

    # raw + pseudobulk (LASSO)
    HCA_donors = donor_info[donor_info["dataset"] == "HCA"].index.tolist()
    HCA_pseudo = get_pseudobulk(adata, HCA_donors)
    HCA_age = donor_info.loc[HCA_donors, "age"]
    print(HCA_pseudo.shape)

    siAge_donors = donor_info[donor_info["dataset"] == "siAge"].index.tolist()
    siAge_pseudo = get_pseudobulk(adata, siAge_donors)
    siAge_age = donor_info.loc[siAge_donors, "age"]
    print(siAge_pseudo.shape)

    train_AIDA_donors = donor_info.loc[
        (donor_info["dataset"] == "AIDA") & donor_info["is_train"],
    ].index.tolist()
    train_AIDA_pseudo = get_pseudobulk(adata, train_AIDA_donors)
    train_AIDA_age = donor_info.loc[train_AIDA_donors, "age"]

    test_AIDA_donors = donor_info.loc[
        (donor_info["dataset"] == "AIDA") & ~donor_info["is_train"],
    ].index.tolist()
    test_AIDA_pseudo = get_pseudobulk(adata, test_AIDA_donors)
    test_AIDA_age = donor_info.loc[test_AIDA_donors, "age"]

    print(train_AIDA_pseudo.shape)
    print(test_AIDA_pseudo.shape)

    train_eQTL_donors = donor_info.loc[
        (donor_info["dataset"] == "eQTL") & donor_info["is_train"],
    ].index.tolist()
    train_eQTL_pseudo = get_pseudobulk(adata, train_eQTL_donors)
    train_eQTL_age = donor_info.loc[train_eQTL_donors, "age"]

    test_eQTL_donors = donor_info.loc[
        (donor_info["dataset"] == "eQTL") & ~donor_info["is_train"],
    ].index.tolist()
    test_eQTL_pseudo = get_pseudobulk(adata, test_eQTL_donors)
    test_eQTL_age = donor_info.loc[test_eQTL_donors, "age"]

    print(train_eQTL_pseudo.shape)
    print(test_eQTL_pseudo.shape)

    train_data = pd.concat([train_AIDA_pseudo, train_eQTL_pseudo], axis=0)
    train_age = donor_info.loc[train_data.index, "age"]
    print(train_data.shape)


    lasso_train = LassoCV(n_jobs=128)
    lasso_train.fit(train_data, train_age)



    lasso_AIDA = lasso_train.predict(test_AIDA_pseudo)
    lasso_AIDA_item = BenchmarkItem(lasso_AIDA, test_AIDA_age)
    lasso_AIDA_item.scatter_plot(color="#c57541", title="Lasso Regression - AIDA internal")
    lasso_AIDA_item.residual_plot(color="#c57541", title="Lasso Regression - AIDA internal")

    lasso_eQTL = lasso_train.predict(test_eQTL_pseudo)
    lasso_eQTL_item = BenchmarkItem(lasso_eQTL, test_eQTL_age)
    lasso_eQTL_item.scatter_plot(color="#777acc", title="Lasso Regression - eQTL internal")
    lasso_eQTL_item.residual_plot(color="#777acc", title="Lasso Regression - eQTL internal")

    lasso_HCA = lasso_train.predict(HCA_pseudo)
    lasso_HCA_item = BenchmarkItem(lasso_HCA, HCA_age)
    lasso_HCA_item.scatter_plot(color="#73a85d", title="Lasso Regression - HCA")
    lasso_HCA_item.residual_plot(color="#73a85d", title="Lasso Regression - HCA")

    lasso_siAge = lasso_train.predict(siAge_pseudo)
    lasso_siAge_item = BenchmarkItem(lasso_siAge, siAge_age)
    lasso_siAge_item.scatter_plot(color="#c45a95", title="Lasso Regression - siAge")
    lasso_siAge_item.residual_plot(color="#c45a95", title="Lasso Regression - siAge")

    # raw + pseudobulk (AutoGluon)
    train_data_age = pd.concat([train_data, train_age], axis=1)
    print(train_data_age)

    predictor = TabularPredictor(
        label="age",
        problem_type="regression",
        path="AutogluonModels/20250903_raw_pseudo",
    )
    predictor.fit(train_data_age)


    predictor = TabularPredictor(
        label="age",
        problem_type="regression",
        path="AutogluonModels/20250903_raw_pseudo",
    )
    predictor.fit(train_data_age)

    ag_AIDA = predictor.predict(test_AIDA_pseudo)
    ag_AIDA_item = BenchmarkItem(ag_AIDA, test_AIDA_age)
    ag_AIDA_item.scatter_plot(color="#c57541", title="AutoGluon - AIDA internal")
    ag_AIDA_item.residual_plot(color="#c57541", title="AutoGluon - AIDA internal")

    ag_eQTL = predictor.predict(test_eQTL_pseudo)
    ag_eQTL_item = BenchmarkItem(ag_eQTL, test_eQTL_age)
    ag_eQTL_item.scatter_plot(color="#777acc", title="AutoGluon - eQTL internal")
    ag_eQTL_item.residual_plot(color="#777acc", title="AutoGluon - eQTL internal")

    ag_HCA = predictor.predict(HCA_pseudo)
    ag_HCA_item = BenchmarkItem(ag_HCA, HCA_age)
    ag_HCA_item.scatter_plot(color="#73a85d", title="AutoGluon - HCA")
    ag_HCA_item.residual_plot(color="#73a85d", title="AutoGluon - HCA")

    ag_siAge = predictor.predict(siAge_pseudo)
    ag_siAge_item = BenchmarkItem(ag_siAge, siAge_age)
    ag_siAge_item.scatter_plot(color="#c45a95", title="AutoGluon - siAge")
    ag_siAge_item.residual_plot(color="#c45a95", title="AutoGluon - siAge")

    # scale+log1p + pseudobulk (LASSO)
    train_donors = train_data.index.values
    train_scale_pseudo = get_pseudobulk(adata, train_donors, use_raw=False)
    print(train_scale_pseudo)

    test_AIDA_scale_pseudo = get_pseudobulk(adata, test_AIDA_donors, use_raw=False)
    test_eQTL_scale_pseudo = get_pseudobulk(adata, test_eQTL_donors, use_raw=False)
    HCA_scale_pseudo = get_pseudobulk(adata, HCA_donors, use_raw=False)
    siAge_scale_pseudo = get_pseudobulk(adata, siAge_donors, use_raw=False)
    print(test_AIDA_scale_pseudo.shape)
    print(test_eQTL_scale_pseudo.shape)
    print(HCA_scale_pseudo.shape)
    print(siAge_scale_pseudo.shape)

    train_data_scale = pd.concat([train_scale_pseudo, train_age], axis=1)
    print(train_data_scale.shape)

    lasso_scale = LassoCV(n_jobs=128)
    lasso_scale.fit(train_scale_pseudo, train_age)

    lasso_AIDA = lasso_scale.predict(test_AIDA_scale_pseudo)
    lasso_AIDA_item = BenchmarkItem(lasso_AIDA, test_AIDA_age)
    lasso_AIDA_item.scatter_plot(color="#c57541", title="Lasso Regression - AIDA internal")
    lasso_AIDA_item.residual_plot(color="#c57541", title="Lasso Regression - AIDA internal")

    lasso_eQTL = lasso_scale.predict(test_eQTL_scale_pseudo)
    lasso_eQTL_item = BenchmarkItem(lasso_eQTL, test_eQTL_age)
    lasso_eQTL_item.scatter_plot(color="#777acc", title="Lasso Regression - eQTL internal")
    lasso_eQTL_item.residual_plot(color="#777acc", title="Lasso Regression - eQTL internal")

    lasso_HCA = lasso_scale.predict(HCA_pseudo)
    lasso_HCA_item = BenchmarkItem(lasso_HCA, HCA_age)
    lasso_HCA_item.scatter_plot(color="#73a85d", title="Lasso Regression - HCA")
    lasso_HCA_item.residual_plot(color="#73a85d", title="Lasso Regression - HCA")

    lasso_siAge = lasso_scale.predict(siAge_pseudo)
    lasso_siAge_item = BenchmarkItem(lasso_siAge, siAge_age)
    lasso_siAge_item.scatter_plot(color="#c45a95", title="Lasso Regression - siAge")
    lasso_siAge_item.residual_plot(color="#c45a95", title="Lasso Regression - siAge")

    # scale+log1p + pseudobulk (AutoGluon)
    scale_predictor = TabularPredictor(
        label="age",
        problem_type="regression",
        path="AutogluonModels/20250903_scale_pseudo",
    )
    scale_predictor.fit(train_data_scale)

    ag_AIDA = scale_predictor.predict(test_AIDA_scale_pseudo)
    ag_AIDA_item = BenchmarkItem(ag_AIDA, test_AIDA_age)
    ag_AIDA_item.scatter_plot(color="#c57541", title="AutoGluon - AIDA internal")
    ag_AIDA_item.residual_plot(color="#c57541", title="AutoGluon - AIDA internal")

    ag_eQTL = scale_predictor.predict(test_eQTL_scale_pseudo)
    ag_eQTL_item = BenchmarkItem(ag_eQTL, test_eQTL_age)
    ag_eQTL_item.scatter_plot(color="#777acc", title="AutoGluon - eQTL internal")
    ag_eQTL_item.residual_plot(color="#777acc", title="AutoGluon - eQTL internal")

    ag_HCA = scale_predictor.predict(HCA_scale_pseudo)
    ag_HCA_item = BenchmarkItem(ag_HCA, HCA_age)
    ag_HCA_item.scatter_plot(color="#73a85d", title="AutoGluon - HCA")
    ag_HCA_item.residual_plot(color="#73a85d", title="AutoGluon - HCA")

    ag_siAge = scale_predictor.predict(siAge_scale_pseudo)
    ag_siAge_item = BenchmarkItem(ag_siAge, siAge_age)
    ag_siAge_item.scatter_plot(color="#c45a95", title="AutoGluon - siAge")
    ag_siAge_item.residual_plot(color="#c45a95", title="AutoGluon - siAge")