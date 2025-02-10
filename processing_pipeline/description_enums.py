from enum import Enum


class ProcessType(Enum):
    """
    Enum representing different types of processes in the pipeline.
    """
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class ProcessPhase(Enum):
    """
    Enum representing different phases of a process in the pipeline.
    """
    INITIALIZATION = "initialization"
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class DataOrigin(Enum):
    """
    Enum representing the origin of data in the pipeline.
    """
    MODULE = "module"
    MODEL = "model"
    TRAINER = "trainer"
    CALCULATOR = "calculator"
    VISUALISATION = "visualisation"


class AbstractionLevel(Enum):
    """
    Enum representing different levels of abstraction in the pipeline.
    """
    INSTANCE = "instance"
    BATCH = "batch"
    EPOCH = "epoch"
    GENERAL = "general"


class Column(Enum):
    """
    Enum representing different columns used in data frames within the pipeline.
    """
    METRICS = "metrics"
    PARAMS = "params"
    MAX_EPOCH = "max_epoch"
    GLOBAL_STEP = "global_step"
    BATCH_IDX = "batch_idx"
    EPOCH = "epoch"
    PREDICTED_DISTRIBUTION = "predicted_distribution"
    TARGET_DISTRIBUTION = "target_distribution"
    DISTRIBUTION_COMPARISON = "distribution_comparison"
    TOTAL_ERROR = "total_error"
    PREDICTED = "predicted"
    TARGET = "target"
    LOSS = "loss"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    R2 = "R2"
    MSE = "mse"
    MAE = "mae"
    NMSE = "nmse"
    NRMSE = "nrmse"
    F1 = "f1"
    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
    CONFUSION_MATRIX = "confusion_matrix"
    CLASSIFICATION_REPORT = "classification_report"
    FEATURE_IMPORTANCE = "feature_importance"
    FEATURE_CORRELATION = "feature_correlation"
    FEATURE_DISTRIBUTION = "feature_distribution"
    FEATURE_SCATTER = "feature_scatter"
    FEATURE_PLOT = "feature_plot"
    FEATURE_HISTOGRAM = "feature_histogram"
    FEATURE_DENSITY = "feature_density"
    FEATURE_BOXPLOT = "feature_boxplot"
    FEATURE_VIOLIN = "feature_violin"
    FEATURE_SWARM = "feature_swarm"
    FEATURE_PAIRPLOT = "feature_pairplot"
    FEATURE_JOINTPLOT = "feature_jointplot"
    FEATURE_REGPLOT = "feature_regplot"
    FEATURE_LM_PLOT = "feature_lm_plot"
    FEATURE_RESIDUAL_PLOT = "feature_residual_plot"
    FEATURE_HEATMAP = "feature_heatmap"
    FEATURE_CLUSTERMAP = "feature_clustermap"
    FEATURE_COUNT_PLOT = "feature_count_plot"
    FEATURE_BAR_PLOT = "feature_bar_plot"
    FEATURE_VIOLIN_PLOT = "feature_violin_plot"
    FEATURE_BOX_PLOT = "feature_box_plot"
    FEATURE_SCATTER_PLOT = "feature_scatter_plot"
    FEATURE_LINE_PLOT = "feature_line_plot"
    FEATURE_AREA_PLOT = "feature_area_plot"
    FEATURE_DENSITY_PLOT = "feature_density_plot"
    FEATURE_HIST_PLOT = "feature_hist_plot"
    FEATURE_PIE_PLOT = "feature_pie_plot"
    FEATURE_DONUT_PLOT = "feature_donut_plot"
    FEATURE_CONTOUR_PLOT = "feature_contour_plot"
    FEATURE_CONTOURF_PLOT = "feature_contourf_plot"
    FEATURE_CONTOUR3D_PLOT = "feature_contour3d_plot"
    FEATURE_CONTOURF3D_PLOT = "feature_contourf3d_plot"
    FEATURE_SURFACE_PLOT = "feature_surface_plot"
