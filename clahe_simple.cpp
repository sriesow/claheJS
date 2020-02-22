
Mat claheGO(Mat src,int _step = 8)
{
    Mat CLAHE_GO = src.clone();
    int block = _step;//pblock
    int width = src.cols;
    int height= src.rows;
    int width_block = width/block; //每个小格子的长和宽
    int height_block = height/block;
    //存储各个直方图  
    int tmp2[8*8][256] ={0};
    float C2[8*8][256] = {0.0};
    //分块
    int total = width_block * height_block; 
    for (int i=0;i<block;i++)
    {
        for (int j=0;j<block;j++)
        {
            int start_x = i*width_block;
            int end_x = start_x + width_block;
            int start_y = j*height_block;
            int end_y = start_y + height_block;
            int num = i+block*j;  
            //遍历小块,计算直方图
            for(int ii = start_x ; ii < end_x ; ii++)  
            {  
                for(int jj = start_y ; jj < end_y ; jj++)  
                {  
                    int index =src.at<uchar>(jj,ii);
                    tmp2[num][index]++;  
                }  
            } 
            //裁剪和增加操作，也就是clahe中的cl部分
            //这里的参数 对应《Gem》上面 fCliplimit  = 4  , uiNrBins  = 255
            int average = width_block * height_block / 255;  
            //关于参数如何选择，需要进行讨论。不同的结果进行讨论
            //关于全局的时候，这里的这个cl如何算，需要进行讨论 
            int LIMIT = 40 * average;  
            int steal = 0;  
            for(int k = 0 ; k < 256 ; k++)  
            {  
                if(tmp2[num][k] > LIMIT){  
                    steal += tmp2[num][k] - LIMIT;  
                    tmp2[num][k] = LIMIT;  
                }  
            }  
            int bonus = steal/256;  
            //hand out the steals averagely  
            for(int k = 0 ; k < 256 ; k++)  
            {  
                tmp2[num][k] += bonus;  
            }  
            //计算累积分布直方图  
            for(int k = 0 ; k < 256 ; k++)  
            {  
                if( k == 0)  
                    C2[num][k] = 1.0f * tmp2[num][k] / total;  
                else  
                    C2[num][k] = C2[num][k-1] + 1.0f * tmp2[num][k] / total;  
            }  
        }
    }
    //计算变换后的像素值  
    //根据像素点的位置，选择不同的计算方法  
    for(int  i = 0 ; i < width; i++)  
    {  
        for(int j = 0 ; j < height; j++)  
        {  
            //four coners  
            if(i <= width_block/2 && j <= height_block/2)  
            {  
                int num = 0;  
                CLAHE_GO.at<uchar>(j,i) = (int)(C2[num][CLAHE_GO.at<uchar>(j,i)] * 255);  
            }else if(i <= width_block/2 && j >= ((block-1)*height_block + height_block/2)){  
                int num = block*(block-1);  
                CLAHE_GO.at<uchar>(j,i) = (int)(C2[num][CLAHE_GO.at<uchar>(j,i)] * 255);  
            }else if(i >= ((block-1)*width_block+width_block/2) && j <= height_block/2){  
                int num = block-1;  
                CLAHE_GO.at<uchar>(j,i) = (int)(C2[num][CLAHE_GO.at<uchar>(j,i)] * 255);  
            }else if(i >= ((block-1)*width_block+width_block/2) && j >= ((block-1)*height_block + height_block/2)){  
                int num = block*block-1;  
                CLAHE_GO.at<uchar>(j,i) = (int)(C2[num][CLAHE_GO.at<uchar>(j,i)] * 255);  
            }  
            //four edges except coners  
            else if( i <= width_block/2 )  
            {  
                //线性插值  
                int num_i = 0;  
                int num_j = (j - height_block/2)/height_block;  
                int num1 = num_j*block + num_i;  
                int num2 = num1 + block;  
                float p =  (j - (num_j*height_block+height_block/2))/(1.0f*height_block);  
                float q = 1-p;  
                CLAHE_GO.at<uchar>(j,i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j,i)]+ p*C2[num2][CLAHE_GO.at<uchar>(j,i)])* 255);  
            }else if( i >= ((block-1)*width_block+width_block/2)){  
                //线性插值  
                int num_i = block-1;  
                int num_j = (j - height_block/2)/height_block;  
                int num1 = num_j*block + num_i;  
                int num2 = num1 + block;  
                float p =  (j - (num_j*height_block+height_block/2))/(1.0f*height_block);  
                float q = 1-p;  
                CLAHE_GO.at<uchar>(j,i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j,i)]+ p*C2[num2][CLAHE_GO.at<uchar>(j,i)])* 255);  
            }else if( j <= height_block/2 ){  
                //线性插值  
                int num_i = (i - width_block/2)/width_block;  
                int num_j = 0;  
                int num1 = num_j*block + num_i;  
                int num2 = num1 + 1;  
                float p =  (i - (num_i*width_block+width_block/2))/(1.0f*width_block);  
                float q = 1-p;  
                CLAHE_GO.at<uchar>(j,i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j,i)]+ p*C2[num2][CLAHE_GO.at<uchar>(j,i)])* 255);  
            }else if( j >= ((block-1)*height_block + height_block/2) ){  
                //线性插值  
                int num_i = (i - width_block/2)/width_block;  
                int num_j = block-1;  
                int num1 = num_j*block + num_i;  
                int num2 = num1 + 1;  
                float p =  (i - (num_i*width_block+width_block/2))/(1.0f*width_block);  
                float q = 1-p;  
                CLAHE_GO.at<uchar>(j,i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j,i)]+ p*C2[num2][CLAHE_GO.at<uchar>(j,i)])* 255);  
            }  
            //双线性插值
            else{  
                int num_i = (i - width_block/2)/width_block;  
                int num_j = (j - height_block/2)/height_block;  
                int num1 = num_j*block + num_i;  
                int num2 = num1 + 1;  
                int num3 = num1 + block;  
                int num4 = num2 + block;  
                float u = (i - (num_i*width_block+width_block/2))/(1.0f*width_block);  
                float v = (j - (num_j*height_block+height_block/2))/(1.0f*height_block);  
                CLAHE_GO.at<uchar>(j,i) = (int)((u*v*C2[num4][CLAHE_GO.at<uchar>(j,i)] +   
                    (1-v)*(1-u)*C2[num1][CLAHE_GO.at<uchar>(j,i)] +  
                    u*(1-v)*C2[num2][CLAHE_GO.at<uchar>(j,i)] +  
                    v*(1-u)*C2[num3][CLAHE_GO.at<uchar>(j,i)]) * 255);  
            }  
            //smooth
            CLAHE_GO.at<uchar>(j,i) = CLAHE_GO.at<uchar>(j,i) + (CLAHE_GO.at<uchar>(j,i) << 8) + (CLAHE_GO.at<uchar>(j,i) << 16);         
        }  
    }  
  return CLAHE_GO;
}