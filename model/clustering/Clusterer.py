from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from osgeo import gdal


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
    ) -> np.ndarray:

        img = np.moveaxis(bands, 0, -1)
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
            validMask = validMask & ~(flatImg == noDataValue).any(axis=1)

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
        initIdx = rng.choice(img1d.shape[0], size=numClusters, replace=False)
        centroids = torch.as_tensor(
            img1d[initIdx],
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

        return imgCl
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