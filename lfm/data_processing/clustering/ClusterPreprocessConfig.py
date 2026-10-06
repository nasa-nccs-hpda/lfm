from dataclasses import dataclass


@dataclass
class ClusterPreprocessConfig:
    """Configuration for optional K-means preprocessing and feature creation."""

    clipPercentiles: tuple[float, float] | None = None
    gaussianSigma: float | None = None
    includeRaw: bool = True
    includeLocalMean: bool = False
    localMeanSize: int = 5
    includeLocalStd: bool = False
    localStdSize: int = 5
    includeGradientMagnitude: bool = False
    includeLaplacian: bool = False
    standardizeFeatures: bool = True
    medianFilterLabelsSize: int | None = None