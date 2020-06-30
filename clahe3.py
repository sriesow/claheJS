import numpy as np
import cv2
import imageio
import math


def interpolationBody(src, dst, lut, tileSize, block):
    ind1_p = []
    ind2_p = []
    xa_p = []
    xa1_p = []

    tileSize_width, tileSize_height = tileSize
    inv_tw = 1 / tileSize_width
    lut_step = 256
    for i in range(src.shape[1]):
        txf = i*inv_tw - 0.5
        tx1 = math.floor(txf)
        tx2 = tx1 + 1

        xa_p.append(float("%0.1f" % (txf - tx1)))
        xa1_p.append(1 - xa_p[i])

        tx1 = max(tx1, 0)
        tx2 = min(tx2, block - 1)
        ind1_p.append(tx1 * lut_step)
        ind2_p.append(tx2 * lut_step)

    inv_th = 1 / tileSize_height
    for i in range(src.shape[0]):
        # print('lut', lut[i])
        tyf = i*inv_th - 0.5
        ty1 = math.floor(tyf)
        ty2 = ty1 + 1

        ya = float("%0.1f" % (tyf - ty1))
        ya1 = 1 - ya

        ty1 = max(ty1, 0)
        ty2 = min(ty2, block - 1)

        lutBlock1 = ty1 * block
        lutBlock2 = ty2 * block

        for j in range(src.shape[1]):
            srcVal = src[i][j]
            # print('val', j, i, srcVal)
            ind1 = ind1_p[j] + srcVal
            ind2 = ind2_p[j] + srcVal
            # print('stats',lutIndex1, lutIndex2, ind1, ind2)
            # print('val', lut[lutIndex1][ind1], lut[lutIndex1][ind2], lut[lutIndex2][ind1], lut[lutIndex2][ind2])

            lutBlock1Residue = int(ind1 / 256)
            lutBlock2Residue = int(ind2 / 256)

            lutIndex1 = ind1 % 256
            lutIndex2 = ind2 % 256
            # print(lut[lutBlock1+lutBlock1Residue])
            # print(lut[lutBlock1+lutBlock1Residue][lutIndex1], lut[lutBlock1+lutBlock2Residue][lutIndex2], lut[lutBlock2+lutBlock1Residue][lutIndex1], lut[lutBlock2+lutBlock2Residue][lutIndex2])
            result = (lut[lutBlock1+lutBlock1Residue][lutIndex1] * xa1_p[j] +
                      lut[lutBlock1+lutBlock2Residue][lutIndex2] * xa_p[j]) * ya1 + (lut[lutBlock2+lutBlock1Residue][lutIndex1] * xa1_p[j] + lut[lutBlock2+lutBlock2Residue][lutIndex2] * xa_p[j]) * ya
            # print('result' , i, j, result)
            # print("\n")
            final_val = int(255 if result > 255 else (
                0 if result < 0 else result))
            dst[i][j] = final_val
        # ind1_p[i] = ty1 * src.shape[1]
        # ind2_p[i] = ty2 * src.shape[1]
    return dst


def claheGO(src, _step=2, cut=3.0):
    CLAHE_GO = np.copy(src)
    # CLAHE_GO = CLAHE_GO.astype(np.float16)
    block = _step
    height = src.shape[0]
    width = src.shape[1]
    width_block = int(width/block)
    height_block = int(height/block)
    histSize = 256
    tileSize = (width_block, height_block)
    total = width_block * height_block
    lutScale = float((histSize - 1) / total)
    lut = np.zeros((block*block, 256), dtype=np.uint8)
    LIMIT = int(cut * total / histSize)
    LIMIT = max(LIMIT, 1)

    for block_count in range(block*block):
        ty = int(block_count / block)
        tx = block_count % block

        start_x = tx * width_block
        start_y = ty * height_block
        end_x = start_x + width_block
        end_y = start_y + height_block

        tileHist = [0 for i in range(histSize)]

        for i in range(start_y, end_y):
            fillCount = 0
            while fillCount <= width_block-4:
                tileHist[src[i][fillCount+start_x]] += 1
                tileHist[src[i][fillCount+start_x+1]] += 1
                tileHist[src[i][fillCount+start_x+2]] += 1
                tileHist[src[i][fillCount+start_x+3]] += 1
                fillCount += 4
            while fillCount < width_block:
                tileHist[src[i][fillCount+start_x]] += 1
                fillCount += 1
        if (LIMIT > 0):
            clipped = 0
            for i in range(histSize):
                if(tileHist[i] > LIMIT):
                    clipped += tileHist[i] - LIMIT
                    tileHist[i] = LIMIT
            redistBatch = int(clipped/histSize)
            residual = int(clipped - redistBatch * histSize)
            for i in range(histSize):
                tileHist[i] += redistBatch
            if residual != 0:
                residualStep = max(int(histSize/residual), 1)
                i = 0
                while i < histSize and residual > 0:
                    tileHist[i] += 1
                    i += residualStep
                    residual -= 1
        sum = 0
        for k in range(histSize):
            sum += tileHist[k]
            value = sum*lutScale
            final_val = int(255 if value > 255 else (
                0 if value < 0 else round(value)))
            lut[block_count][k] = final_val
    CLAHE_GO = interpolationBody(
        src, CLAHE_GO, lut, (width_block, height_block), block)
    return CLAHE_GO


def doClahe(x, step):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(step, step))
    cl = clahe.apply(x.astype(np.uint8))
    converted = claheGO(x.astype(np.uint8), _step=step)
    print("Stats(min, max, sum, mean)")
    diff = cl-converted
    print(np.min(diff), np.max(diff), np.count_nonzero(diff), np.mean(diff))
    return cl, converted


# x = np.random.randint(low=0, high=255, size=(224, 224))
# doClahe(x)
# # x = np.array([72, 136, 128,  59, 223,  85, 252,  75,
# #               246,  84, 111, 107, 249, 245, 229,  94])
# # x = np.reshape(x, (4, 4))
# print("source")
# print(x.astype(np.uint8))
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
# cl = clahe.apply(x.astype(np.uint8))
# print("clahe")
# print(cl)
# converted = claheGO(x.astype(np.uint8))
# print("converted")
# print(converted)
# print("Stats(min, max, sum, mean)")
# diff = cl-converted
# print(np.min(diff), np.max(diff), np.sum(diff), np.mean(diff))
