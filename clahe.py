import numpy as np
import cv2
import imageio

def claheGO(src, _step=2):
    CLAHE_GO = np.copy(src)
    # CLAHE_GO = CLAHE_GO.astype(np.float16)
    block = _step
    width = src.shape[0]
    height = src.shape[1]
    width_block = width/block
    height_block = height/block

    tmp2 = np.zeros((block*block, 256), dtype=np.float)
    C2 = np.zeros((block*block, 256), dtype=np.float16)
    total = width_block * height_block
    lutScale = float((255) / total);
    for i in range(block):
        for j in range(block):
            start_x = i*width_block
            end_x = start_x + width_block
            start_y = j*height_block
            end_y = start_y + height_block
            num = i+block*j
            for ii in range(int(start_x), int(end_x)):
                for jj in range(int(start_y), int(end_y)):
                    index = src[jj][ii]
                    tmp2[num][index] += 1

            average = total / 256
            LIMIT = 3.0 * average
            LIMIT = max(LIMIT, 1)
            steal = 0
            for k in range(256):
                if tmp2[num][k] > LIMIT:
                    steal += tmp2[num][k] - LIMIT
                    tmp2[num][k] = LIMIT

            bonus = steal/256
            residual = steal - bonus * 256
            for k in range(256):
                tmp2[num][k] += bonus

            if residual != 0:
                print(residual)
                residualStep = max(256 / residual, 1)
                i = 0
                while i < 256 and residual > 0:
                     tmp2[num][i] += 1
                     i += residualStep
                     residual-=1
            sum = 0
            # opencv methpd
            # for k in range(256):
            #     sum += tmp2[num][k]
            #     value = sum*lutScale
            #     final_val = int(255 if value > 255 else (0 if value < 0 else value))
            #     C2[num][k] = final_val
            # print("opencv", C2[num].astype(np.uint8))
            # alternate method
            for k in range(256):
                if k == 0:
                    C2[num][k] = 1.0 * tmp2[num][k] / total
                else:
                    C2[num][k] = C2[num][k-1] + 1.0 * tmp2[num][k] / total
            # print("altern", (C2[num]*255).astype(np.uint8))
    for i in range(width):
        for j in range(height):
            if (i <= width_block/2 and j <= height_block/2):
                num = 0
                CLAHE_GO[j][i] = (int)(C2[num][CLAHE_GO[j][i]] * 255)
            elif(i <= width_block/2 and j >= ((block-1)*height_block + height_block/2)):
                num = block*(block-1)
                CLAHE_GO[j][i] = (int)(C2[num][CLAHE_GO[j][i]] * 255)
            elif(i >= ((block-1)*width_block+width_block/2) and j <= height_block/2):
                num = block-1
                CLAHE_GO[j][i] = (int)(C2[num][CLAHE_GO[j][i]] * 255)
            elif(i >= ((block-1)*width_block+width_block/2) and j >= ((block-1)*height_block + height_block/2)):
                num = block*block-1
                CLAHE_GO[j][i] = (int)(C2[num][CLAHE_GO[j][i]] * 255)
            elif(i <= width_block/2):
                num_i = 0
                num_j = int((j - height_block/2)/height_block)
                num1 = int(num_j*block + num_i)
                num2 = int(num1 + block)
                p = (j - (num_j*height_block+height_block/2))/(1.0*height_block)
                q = 1-p
                CLAHE_GO[j][i] = (int)(
                    (q*C2[num1][CLAHE_GO[j][i]] + p*C2[num2][CLAHE_GO[j][i]]) * 255)
            elif(i >= ((block-1)*width_block+width_block/2)):
                num_i = block-1
                num_j = int((j - height_block/2)/height_block)
                num1 = int(num_j*block + num_i)
                num2 = int(num1 + block)
                p = (j - (num_j*height_block+height_block/2))/(1.0*height_block)
                q = 1-p
                CLAHE_GO[j][i] = (int)(
                    (q*C2[num1][CLAHE_GO[j][i]] + p*C2[num2][CLAHE_GO[j][i]]) * 255)
            elif(j <= height_block/2):
                num_i = int((i - width_block/2)/width_block)
                num_j = 0
                num1 = int(num_j*block + num_i)
                num2 = int(num1 + 1)
                p = (i - (num_i*width_block+width_block/2))/(1.0*width_block)
                q = 1-p
                CLAHE_GO[j][i] = (int)(
                    (q*C2[num1][CLAHE_GO[j][i]] + p*C2[num2][CLAHE_GO[j][i]]) * 255)
            elif(j >= ((block-1)*height_block + height_block/2)):
                num_i = int((i - width_block/2)/width_block)
                num_j = block-1
                num1 = num_j*block + num_i
                num2 = num1 + 1
                p = (i - (num_i*width_block+width_block/2))/(1.0*width_block)
                q = 1-p
                CLAHE_GO[j][i] = (int)(
                    (q*C2[num1][CLAHE_GO[j][i]] + p*C2[num2][CLAHE_GO[j][i]]) * 255)
            else:
                num_i = int((i - width_block/2)/width_block)
                num_j = int((j - height_block/2)/height_block)
                num1 = num_j*block + num_i
                num2 = num1 + 1
                num3 = num1 + block
                num4 = num2 + block
                u = (i - (num_i*width_block+width_block/2))/(1.0*width_block)
                v = (j - (num_j*height_block+height_block/2))/(1.0*height_block)
                CLAHE_GO[j][i] = (int)((u*v*C2[num4][CLAHE_GO[j][i]] +
                                        (1-v)*(1-u)*C2[num1][CLAHE_GO[j][i]] +
                                        u*(1-v)*C2[num2][CLAHE_GO[j][i]] +
                                        v*(1-u)*C2[num3][CLAHE_GO[j][i]]) * 255)
            # CLAHE_GO[j][i] = CLAHE_GO[j][i] + \
            #     (CLAHE_GO[j][i] << 8) + (CLAHE_GO[j][i] << 16)
    return CLAHE_GO


# img_path = "/home/raam/Downloads/sample_fp/0.5686317086219788 - original.png"
# lab_data = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2HLS)
# h, l, s = cv2.split(lab_data)
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# cl = clahe.apply(l)
# # print(cl)
# limg = cv2.merge((h,cl,s))
# final_clahe_hls = cv2.cvtColor(limg, cv2.COLOR_HLS2RGB)
# imageio.imsave('hls_try.jpg', final_clahe_hls)
# converted = claheGO(l)
# limg = cv2.merge((h,converted,s))
# final_clahe_hls = cv2.cvtColor(limg, cv2.COLOR_HLS2RGB)
# imageio.imsave('hls_try_1.jpg', final_clahe_hls)

# x = np.random.randint(low=0, high=255, size=(224,224))
x = np.array([72, 136, 128,  59, 223,  85, 252,  75, 246,  84, 111, 107, 249, 245, 229,  94])
x = np.reshape(x, (4,4))
print("source")
print(x.astype(np.uint8))
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2, 2))
cl = clahe.apply(x.astype(np.uint8))
print("clahe")
print(cl)
converted = claheGO(x.astype(np.uint8))
print("converted")
print(converted)
print("Stats(min, max, mean)")
diff = cl-converted
print(np.min(diff), np.max(diff), np.mean(diff))