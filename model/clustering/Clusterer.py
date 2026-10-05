from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from osgeo import gdal

from model.clustering.ClusterPreprocessConfig import ClusterPreprocessConfig


# ----------------------------------------------------------------------------
# Class Clusterer
# ----------------------------------------------------------------------------
class Clusterer(object):

    # ------------------------------------------------------------------------
    # getClusters
    # ------------------------------------------------------------------------
    @staticmethod
    def getClusters(
        bands: list,
        numClusters: int = 20,
        batchSize: int = 262144,
        maxIter: int = 20,
        tol: float = 1.0e-4,
        device: str = 'auto',
        showProgress: bool = True,
        randomState: int = 0,
        noDataValue=None,
        noDataLabel: int = 0,
        preprocessConfig: ClusterPreprocessConfig | None = None,
    ) -> np.ndarray:

        img = Clusterer._prepareFeatures(bands, noDataValue, preprocessConfig)
        flatImg = np.ascontiguousarray(
            img.reshape(-1, img.shape[-1]),
            dtype=np.float32,
        )

        finiteMask = np.isfinite(flatImg).all(axis=1)
        if not finiteMask.all():
            raise ValueError(
                'Input bands contain NaN or infinite values. Replace or mask '
                'invalid pixels before clustering.'
            )

        validMask = finiteMask
        if noDataValue is not None:
            source = np.moveaxis(bands, 0, -1).reshape(-1, len(bands))
            validMask = validMask & ~(source == noDataValue).any(axis=1)

        if not validMask.any():
            raise ValueError('No valid pixels are available for clustering.')

        img1d = flatImg[validMask]

        if img1d.shape[0] < numClusters:
            raise ValueError(
                f'Cannot create {numClusters} clusters from '
                f'{img1d.shape[0]} valid pixels.'
            )

        if device == 'auto':
            torchDevice = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            torchDevice = torch.device(device)

        rng = np.random.default_rng(randomState)
        if img1d.shape[1] == 1:
            quantiles = np.linspace(0.0, 1.0, numClusters + 2, dtype=np.float32)[1:-1]
            initialCentroids = np.quantile(img1d[:, 0], quantiles).reshape(-1, 1)
        else:
            initIdx = rng.choice(img1d.shape[0], size=numClusters, replace=False)
            initialCentroids = img1d[initIdx]

        centroids = torch.as_tensor(
            np.ascontiguousarray(initialCentroids, dtype=np.float32),
            dtype=torch.float32,
            device=torchDevice,
        )

        nPixels = img1d.shape[0]
        batchSize = max(1, int(batchSize))
        nBatches = (nPixels + batchSize - 1) // batchSize
        prevInertia = None

        fitProgress = None
        if showProgress:
            fitProgress = tqdm(
                total=maxIter * nBatches,
                desc='K-Means fit',
                unit='batch',
            )

        try:
            for _ in range(maxIter):
                sums = torch.zeros_like(centroids)
                counts = torch.zeros(numClusters, dtype=torch.long, device=torchDevice)
                inertia = 0.0

                for start in range(0, nPixels, batchSize):
                    end = min(start + batchSize, nPixels)
                    batch = torch.as_tensor(
                        img1d[start:end],
                        dtype=torch.float32,
                        device=torchDevice,
                    )
                    distances = torch.cdist(batch, centroids)
                    minDistances, labels = distances.min(dim=1)

                    sums.index_add_(0, labels, batch)
                    counts += torch.bincount(labels, minlength=numClusters)
                    inertia += float((minDistances ** 2).sum().item())

                    if fitProgress is not None:
                        fitProgress.update(1)

                nonEmpty = counts > 0
                centroids[nonEmpty] = sums[nonEmpty] / counts[nonEmpty].unsqueeze(1)

                empty = torch.where(~nonEmpty)[0]
                if len(empty) > 0:
                    replaceIdx = rng.choice(nPixels, size=len(empty), replace=False)
                    centroids[empty] = torch.as_tensor(
                        img1d[replaceIdx],
                        dtype=torch.float32,
                        device=torchDevice,
                    )

                if fitProgress is not None:
                    fitProgress.set_postfix(inertia=f'{inertia:.4g}')

                if prevInertia is not None:
                    denom = max(abs(prevInertia), 1.0)
                    if abs(prevInertia - inertia) / denom < tol:
                        break
                prevInertia = inertia
        finally:
            if fitProgress is not None:
                fitProgress.close()

        validLabels = np.empty(nPixels, dtype=np.int32)
        predictIter = range(0, nPixels, batchSize)
        if showProgress:
            predictIter = tqdm(predictIter, desc='K-Means predict', unit='batch')

        with torch.no_grad():
            for start in predictIter:
                end = min(start + batchSize, nPixels)
                batch = torch.as_tensor(
                    img1d[start:end],
                    dtype=torch.float32,
                    device=torchDevice,
                )
                labels = torch.cdist(batch, centroids).argmin(dim=1)
                validLabels[start:end] = labels.cpu().numpy()

        imgCl = np.full(flatImg.shape[0], noDataLabel, dtype=np.int32)
        if noDataValue is None:
            imgCl[validMask] = validLabels
        else:
            imgCl[validMask] = validLabels + 1
        imgCl = imgCl.reshape(img[:, :, 0].shape)

        if (
            preprocessConfig is not None
            and preprocessConfig.medianFilterLabelsSize is not None
            and preprocessConfig.medianFilterLabelsSize > 1
        ):
            imgCl = Clusterer._medianFilterLabels(
                imgCl,
                preprocessConfig.medianFilterLabelsSize,
            )
            imgCl[~validMask.reshape(imgCl.shape)] = noDataLabel

        return imgCl

    @staticmethod
    def _prepareFeatures(
        bands: list,
        noDataValue=None,
        config: ClusterPreprocessConfig | None = None,
    ) -> np.ndarray:
        img = np.moveaxis(bands, 0, -1).astype(np.float32, copy=False)
        if config is None:
            return img

        base = img[:, :, 0]
        validMask = np.isfinite(base)
        if noDataValue is not None:
            validMask &= base != noDataValue

        working = base.copy()
        if config.clipPercentiles is not None and validMask.any():
            lo, hi = np.percentile(working[validMask], config.clipPercentiles)
            working[validMask] = np.clip(working[validMask], lo, hi)

        if config.gaussianSigma is not None and config.gaussianSigma > 0:
            working = Clusterer._smooth(working, validMask, config.gaussianSigma)

        features = []
        if config.includeRaw:
            features.append(working)
        if config.includeLocalMean:
            features.append(Clusterer._boxSmooth(working, validMask, config.localMeanSize))
        if config.includeLocalStd:
            mean = Clusterer._boxSmooth(working, validMask, config.localStdSize)
            meanSq = Clusterer._boxSmooth(working * working, validMask, config.localStdSize)
            features.append(np.sqrt(np.maximum(meanSq - mean * mean, 0.0)))
        if config.includeGradientMagnitude:
            gy, gx = np.gradient(working)
            features.append(np.sqrt(gx * gx + gy * gy))
        if config.includeLaplacian:
            lap = np.zeros_like(working, dtype=np.float32)
            lap[1:-1, 1:-1] = (
                working[:-2, 1:-1]
                + working[2:, 1:-1]
                + working[1:-1, :-2]
                + working[1:-1, 2:]
                - 4.0 * working[1:-1, 1:-1]
            )
            features.append(lap)

        if not features:
            raise ValueError('At least one clustering feature must be enabled.')

        featureImg = np.stack(features, axis=-1).astype(np.float32, copy=False)
        if config.standardizeFeatures and validMask.any():
            for idx in range(featureImg.shape[-1]):
                layer = featureImg[:, :, idx]
                values = layer[validMask]
                mean = values.mean()
                std = values.std()
                if std > 0:
                    featureImg[:, :, idx] = (layer - mean) / std
                else:
                    featureImg[:, :, idx] = layer - mean

        return featureImg

    @staticmethod
    def _smooth(values: np.ndarray, validMask: np.ndarray, sigma: float) -> np.ndarray:
        try:
            from scipy.ndimage import gaussian_filter

            filled = np.where(validMask, values, 0.0)
            weights = validMask.astype(np.float32)
            smoothed = gaussian_filter(filled, sigma=sigma)
            smoothedWeights = gaussian_filter(weights, sigma=sigma)
            return np.divide(
                smoothed,
                smoothedWeights,
                out=np.zeros_like(smoothed),
                where=smoothedWeights > 0,
            )
        except ImportError:
            size = max(3, int(round(sigma * 2 + 1)))
            return Clusterer._boxSmooth(values, validMask, size)

    @staticmethod
    def _boxSmooth(values: np.ndarray, validMask: np.ndarray, size: int) -> np.ndarray:
        size = max(1, int(size))
        if size % 2 == 0:
            size += 1
        radius = size // 2
        paddedValues = np.pad(np.where(validMask, values, 0.0), radius, mode='edge')
        paddedWeights = np.pad(validMask.astype(np.float32), radius, mode='edge')
        valueIntegral = Clusterer._integralImage(paddedValues)
        weightIntegral = Clusterer._integralImage(paddedWeights)
        totals = Clusterer._windowSums(valueIntegral, size)
        weights = Clusterer._windowSums(weightIntegral, size)
        return np.divide(totals, weights, out=np.zeros_like(totals), where=weights > 0)

    @staticmethod
    def _integralImage(values: np.ndarray) -> np.ndarray:
        return np.pad(values, ((1, 0), (1, 0)), mode='constant').cumsum(0).cumsum(1)

    @staticmethod
    def _windowSums(integral: np.ndarray, size: int) -> np.ndarray:
        return (
            integral[size:, size:]
            - integral[:-size, size:]
            - integral[size:, :-size]
            + integral[:-size, :-size]
        )

    @staticmethod
    def _medianFilterLabels(labels: np.ndarray, size: int) -> np.ndarray:
        size = max(1, int(size))
        if size % 2 == 0:
            size += 1
        radius = size // 2
        padded = np.pad(labels, radius, mode='edge')
        windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
        return np.median(windows, axis=(-2, -1)).astype(labels.dtype, copy=False)

    # ------------------------------------------------------------------------
    # labelsToGeotiff
    # These renderers seem to write tiles or pyramids to disk.
    # To make rendering code easier, write the labels as a geotiff; then the
    # renderer will not need to do it.
    # ------------------------------------------------------------------------
    @staticmethod
    def labelsToGeotiff(referenceDs: gdal.Dataset,
                        labelsFile: Path,
                        labels: np.ndarray) -> gdal.Dataset:

        if labelsFile.exists():
            labelsFile.unlink()

        labelsDs = gdal.GetDriverByName('GTiff').Create( \
                        str(labelsFile),
                        xsize=referenceDs.RasterXSize,
                        ysize=referenceDs.RasterYSize,
                        eType=gdal.GDT_Float32,
                        options=['COMPRESS=LZW']
                   )

        labelsDs.SetSpatialRef(referenceDs.GetSpatialRef())
        labelsDs.SetGeoTransform(referenceDs.GetGeoTransform())
        outBand = labelsDs.GetRasterBand(1)
        outBand.WriteArray(labels)
        outBand.SetDescription('Cluster labels')
        outBand.ComputeStatistics(0)  # For min, max used in color table
        labelsDs.FlushCache()
        labelsDs.BuildOverviews()

        return labelsDs