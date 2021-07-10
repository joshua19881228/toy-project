#include<stdlib.h>
#include<memory.h>
#include<math.h>
#include "face_align.h"

#define MAX(a,b)   (a)>(b)?(a):(b)

CFaceAlign::CFaceAlign()
{
    memcpy(face_3D_pt_mean_model, face_3D_pt_mean_model_const, front_face_3D_pt_num * sizeof(Facial_3D_pt));
    memset(face_2D_pt_mean_model, 0, front_face_2D_pt_num * sizeof(Facial_2D_pt));//2D平锟斤拷锟斤拷13锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷
    memset(face_2D_pt_regression_mean_model, 0, front_face_2D_pt_num * sizeof(Facial_2D_pt));//SDM锟截癸拷锟斤拷13锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷
    norm_texture_wd_ht = 64;//2D平锟斤拷锟斤拷锟竭达拷
    norm_regression_wd_ht = 80;//SDM锟截癸拷锟斤拷锟斤拷锟斤拷锟竭达拷
    affine_norm_wd_ht = 128;//2D转锟斤拷锟斤拷锟斤拷原始锟斤拷锟斤拷锟竭达拷
    final_norm_wd_ht = 128;//2D转锟斤拷锟斤拷锟斤拷要锟矫碉拷锟斤拷全锟斤拷锟斤拷锟斤拷锟竭寸（一锟斤拷锟斤拷2D转锟斤拷锟斤拷锟斤拷原始锟斤拷锟斤拷锟竭达拷锟较凤拷锟缴ｏ拷
    projection_focus_factor = 1.875;//3D-2D投影锟斤拷锟斤拷
    //int select_pt_num_front[front_face_3D_pt_num]={0,1,2,3,4,5,6,7,8,9,10,11,12};
    mean_texture_projection_move_z = -6.0;//3D-2D投影锟斤拷锟斤拷
    std_mean_focus = 0.0;//3D-2D投影锟斤拷锟斤拷

    Projection_From_3D_To_2D();
}

void CFaceAlign::Projection_From_3D_To_2D()
//锟斤拷锟斤拷13锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷3D-2D投影锟斤拷锟斤拷face_3D_pt_mean_model锟矫碉拷face_2D_pt_mean_model锟斤拷face_2D_pt_regression_mean_model
{
    std_mean_focus = -norm_texture_wd_ht * projection_focus_factor;
    for (int j = 0; j < front_face_2D_pt_num; j++)
    {
        face_2D_pt_mean_model[j].x = std_mean_focus * (face_3D_pt_mean_model[j].x / (face_3D_pt_mean_model[j].z + mean_texture_projection_move_z)) + norm_texture_wd_ht / 2;
        face_2D_pt_mean_model[j].y = -std_mean_focus * (face_3D_pt_mean_model[j].y / (face_3D_pt_mean_model[j].z + mean_texture_projection_move_z)) + norm_texture_wd_ht / 2;

        face_2D_pt_regression_mean_model[j].x = face_2D_pt_mean_model[j].x - norm_texture_wd_ht / 2 + norm_regression_wd_ht / 2;
        face_2D_pt_regression_mean_model[j].y = face_2D_pt_mean_model[j].y - norm_texture_wd_ht / 2 + norm_regression_wd_ht / 2;
    }
}

void CFaceAlign::Affine_transformation_2D(Facial_2D_pt face_key_2D_getpt[],
    unsigned char* get_norm_image, unsigned char* image, int ht, int wd)
    //锟斤拷锟捷硷拷锟解到锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷2D锟斤拷锟斤拷校锟斤拷
    //face_key_2D_getpt:[input], 源图片锟叫硷拷锟解到锟斤拷13锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷
    //get_norm_image:[output],锟斤拷锟节达拷锟斤拷目锟斤拷图片锟斤拷转锟斤拷锟斤拷锟斤拷图片锟斤拷锟斤拷锟节达拷buffer,
    //image:[input],源图片锟斤拷锟斤拷
    //ht:[input],源图片锟斤拷锟斤拷
    //wd:[input],源图片锟斤拷锟斤拷
{
    float pt1_x[front_face_2D_pt_num], pt1_y[front_face_2D_pt_num];
    float pt2_x[front_face_2D_pt_num], pt2_y[front_face_2D_pt_num];
    for (int j = 0; j < front_face_2D_pt_num; j++)
    {
        pt1_x[j] = ((float)affine_norm_wd_ht) / norm_regression_wd_ht * (face_2D_pt_regression_mean_model[j].x - norm_regression_wd_ht / 2) + affine_norm_wd_ht / 2 + (final_norm_wd_ht - affine_norm_wd_ht) / 2;
        pt1_y[j] = ((float)affine_norm_wd_ht) / norm_regression_wd_ht * (face_2D_pt_regression_mean_model[j].y - norm_regression_wd_ht / 2) + affine_norm_wd_ht / 2 + (final_norm_wd_ht - affine_norm_wd_ht) / 2;

        //pt1_x[j] = ((float)affine_norm_wd_ht) / norm_texture_wd_ht*(face_2D_pt_mean_model[j].x - norm_texture_wd_ht / 2) + affine_norm_wd_ht / 2 + (final_norm_wd_ht - affine_norm_wd_ht) / 2;
        //pt1_y[j] = ((float)affine_norm_wd_ht) / norm_texture_wd_ht*(face_2D_pt_mean_model[j].y - norm_texture_wd_ht / 2) + affine_norm_wd_ht / 2 + (final_norm_wd_ht - affine_norm_wd_ht) / 2;

        pt2_x[j] = face_key_2D_getpt[j].x;
        pt2_y[j] = face_key_2D_getpt[j].y;
    }

    float rot_s_x_n, rot_s_y_n, move_x_n, move_y_n;
    ClaAffineTransfromCoeff_float(pt2_x, pt2_y, pt1_x, pt1_y, front_face_2D_pt_num, rot_s_x_n, rot_s_y_n, move_x_n, move_y_n);
    ImageAffineTransform_Sam_Bilinear(rot_s_x_n, rot_s_y_n, move_x_n, move_y_n, get_norm_image, final_norm_wd_ht, final_norm_wd_ht, image, ht, wd);//final_norm_wd_ht,final_norm_wd_ht应为锟斤拷final_norm_ht锟斤拷锟斤拷为final_norm_wd

}
void CFaceAlign::ClaAffineTransfromCoeff_float(float* pt1_x, float* pt1_y, float* pt2_x, float* pt2_y,
    int npt, float& rot_s_x, float& rot_s_y, float& move_x, float& move_y)
    //锟斤拷锟斤拷锟斤拷小锟斤拷锟斤拷锟斤拷锟斤拷affine锟戒换锟斤拷锟斤拷锟侥硷拷锟斤拷
    //前4锟斤拷锟斤拷锟斤拷为锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷4锟斤拷锟斤拷锟斤拷为锟斤拷锟斤拷锟斤拷锟斤拷
{
    float* X, * A, * B;
    float* temp, * TA;
    int nDim = 4, nrow = npt * 2;
    int i, ii;
    int n1, n2;

    X = (float*)malloc(sizeof(float) * nDim);
    A = (float*)malloc(sizeof(float) * npt * nDim * 2);
    TA = (float*)malloc(sizeof(float) * npt * nDim * 2);
    B = (float*)malloc(sizeof(float) * npt * 2);
    temp = (float*)malloc(sizeof(float) * nDim * nDim);

    for (i = 0; i < npt; ++i)
    {
        ii = (i << 1);
        n1 = ii * nDim;
        n2 = (ii + 1) * nDim;
        B[ii] = pt1_x[i];
        B[ii + 1] = pt1_y[i];
        A[n1] = pt2_x[i];
        A[n1 + 1] = -pt2_y[i];
        A[n1 + 2] = 1;
        A[n1 + 3] = 0;
        A[n2] = pt2_y[i];
        A[n2 + 1] = pt2_x[i];
        A[n2 + 2] = 0;
        A[n2 + 3] = 1;
    }

    MatrixTranspose(A, nrow, nDim, TA);
    MatrixMulti(TA, nDim, nrow, A, nrow, nDim, temp);
    MatrixInverse(temp, nDim, nDim);
    MatrixMulti(TA, nDim, nrow, B, nrow, 1, A);
    MatrixMulti(temp, nDim, nDim, A, nDim, 1, X);

    rot_s_x = X[0];
    rot_s_y = X[1];
    move_x = X[2];
    move_y = X[3];

    free(TA);
    free(A);
    free(B);
    free(temp);
    free(X);
}

void CFaceAlign::ImageAffineTransform_Sam_Bilinear(float rot_s_x, float rot_s_y, float move_x, float move_y,
    unsigned char* image, int ht, int wd, unsigned char* ori_image, int oriht, int oriwd)
    //锟斤拷锟斤拷affine锟戒换
    //前4锟斤拷锟斤拷锟斤拷:[input],锟戒换锟斤拷锟斤拷
    //image:[output],锟斤拷锟节达拷锟斤拷目锟斤拷图片锟斤拷锟节达拷buffer
    //ht,wd:[input],目锟斤拷图片锟斤拷转锟斤拷锟斤拷锟斤拷图片锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷
    //ori_image:[input],源图片锟斤拷锟斤拷
    //oriht, oriwd:[input],源图片锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷
{
    int i, j;
    float x1, y1;
    float* rx, * ry;

    int max_ht_wd = MAX(ht, wd) + 1;
    float tx1, ty1;
    rx = (float*)malloc(sizeof(float) * max_ht_wd);
    ry = (float*)malloc(sizeof(float) * max_ht_wd);

    for (i = 0; i < max_ht_wd; ++i)
        rx[i] = rot_s_x * i;
    if (rot_s_y == 0)
        memset(ry, 0, sizeof(int) * max_ht_wd);
    else
    {
        for (i = 0; i < max_ht_wd; ++i)
            ry[i] = rot_s_y * i;
    }

    for (i = 0; i < ht; ++i)
    {
        tx1 = -ry[i] + move_x;
        ty1 = rx[i] + move_y;
        for (j = 0; j < wd; ++j)
        {
            x1 = rx[j] + tx1;
            y1 = ry[j] + ty1;
            image[i * wd + j] = 0;
            if (x1 < 0 || y1 < 0 || x1 >= oriwd - 1 || y1 >= oriht - 1)
            {
                continue;
            }
            int x_int = int(x1);
            int y_int = int(y1);
            float x_tail = x1 - x_int;
            float y_tail = y1 - y_int;
            int x_round = x_int + 1;
            int y_round = y_int + 1;
            float pixel1 = ori_image[y_int * oriwd + x_int] * (1 - x_tail) + ori_image[y_int * oriwd + x_round] * x_tail;
            float pixel2 = ori_image[y_round * oriwd + x_int] * (1 - x_tail) + ori_image[y_round * oriwd + x_round] * x_tail;
            image[i * wd + j] = int(pixel1 * (1 - y_tail) + pixel2 * y_tail + 0.5);
        }
    }
    free(rx);
    free(ry);
}

bool CFaceAlign::MatrixTranspose(float* m1, int row1, int col1, float* m2)
//锟斤拷锟斤拷转锟斤拷,m1->m2
//m2为锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷为锟斤拷锟斤拷
{
    int i, j;
    if (m2 == NULL)
    {
        float* m3;
        m3 = (float*)malloc(sizeof(float) * row1 * col1);
        for (i = 0; i < col1; ++i)
            for (j = 0; j < row1; ++j)
            {
                m3[i * row1 + j] = m1[j * col1 + i];
            }
        for (i = 0; i < row1; ++i)
            for (j = 0; j < col1; ++j)
                m1[i * col1 + j] = m3[j * col1 + i];
        free(m3);
    }
    else
    {
        for (i = 0; i < col1; ++i)
            for (j = 0; j < row1; ++j)
            {
                m2[i * row1 + j] = m1[j * col1 + i];
            }
    }
    return true;
}

bool CFaceAlign::MatrixMulti(float* m1, int row1, int col1, float* m2, int row2, int col2, float* m3)
//锟斤拷锟斤拷锟斤拷锟斤拷,m1*m2->m3
//m3为锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷为锟斤拷锟斤拷
{
    int i, j, k;
    for (i = 0; i < row1; ++i)
        for (j = 0; j < col2; ++j)
        {
            float sum = 0;
            for (k = 0; k < col1; ++k)
                sum += m1[i * col1 + k] * m2[k * col2 + j];
            m3[i * col2 + j] = sum;
        }
    return true;
}

bool CFaceAlign::MatrixInverse(float* m1, int row1, int col1)
//锟斤拷锟斤拷锟斤拷锟斤拷,m1->m1
{
    int i, j, k;
    float div, temp;
    float* out;
    int* is, * js;
    if (row1 != col1)
        return false;
    out = (float*)malloc(sizeof(float) * row1 * col1);
    is = (int*)malloc(sizeof(int) * row1);
    js = (int*)malloc(sizeof(int) * row1);
    for (i = 0; i < row1; ++i)
    {
        is[i] = i;
        js[i] = i;
    }
    //start from first column to next
    for (k = 0; k < row1; ++k)
    {
        div = 0;
        for (i = k; i < row1; ++i)
            for (j = k; j < row1; ++j)
            {
                if (fabs(m1[i * col1 + j]) > div)
                {
                    div = fabs(m1[i * col1 + j]);
                    is[k] = i;
                    js[k] = j;
                }
            }
        if (fabs(div) < 1e-40)
        {
            free(out);
            free(is);
            free(js);
            return false;
        }
        if (is[k] != k)
        {
            for (j = 0; j < row1; ++j)
            {
                temp = m1[k * col1 + j];
                m1[k * col1 + j] = m1[is[k] * col1 + j];
                m1[is[k] * col1 + j] = temp;
            }
        }
        if (js[k] != k)
        {
            for (i = 0; i < row1; ++i)
            {
                temp = m1[i * col1 + k];
                m1[i * col1 + k] = m1[i * col1 + js[k]];
                m1[i * col1 + js[k]] = temp;
            }
        }
        m1[k * col1 + k] = 1 / m1[k * col1 + k];
        for (j = 0; j < row1; ++j)
        {
            if (j != k)
                m1[k * col1 + j] = m1[k * col1 + j] * m1[k * col1 + k];
        }
        for (i = 0; i < row1; ++i)
        {
            if (i != k)
            {
                for (j = 0; j < row1; ++j)
                {
                    if (j != k)
                        m1[i * col1 + j] -= m1[i * col1 + k] * m1[k * col1 + j];
                }
            }
        }
        for (i = 0; i < row1; ++i)
        {
            if (i != k)
                m1[i * col1 + k] = -m1[i * col1 + k] * m1[k * col1 + k];
        }
    }
    for (k = row1 - 1; k >= 0; --k)
    {
        for (j = 0; j < row1; ++j)
            if (js[k] != k)
            {
                temp = m1[k * col1 + j];
                m1[k * col1 + j] = m1[js[k] * col1 + j];
                m1[js[k] * col1 + j] = temp;
            }
        for (i = 0; i < row1; ++i)
            if (is[k] != k)
            {
                temp = m1[i * col1 + k];
                m1[i * col1 + k] = m1[i * col1 + is[k]];
                m1[i * col1 + is[k]] = temp;
            }
    }
    free(is);
    free(js);
    free(out);
    return true;
}

//void CFaceAlign::cvtLandmark2Align(Face a_face, Facial_2D_pt face_key_2D_getpt[], int w, int h, bool is_roi)
//{
//	if (is_roi)
//	{
//		face_key_2D_getpt[0].x = (a_face.landmarks.points[39].x - a_face.x)*w;
//		face_key_2D_getpt[0].y = (a_face.landmarks.points[39].y - a_face.y)*h;
//
//		face_key_2D_getpt[1].x = (a_face.landmarks.points[36].x - a_face.x)*w;
//		face_key_2D_getpt[1].y = (a_face.landmarks.points[36].y - a_face.y)*h;
//
//		face_key_2D_getpt[2].x = (a_face.landmarks.points[42].x - a_face.x)*w;
//		face_key_2D_getpt[2].y = (a_face.landmarks.points[42].y - a_face.y)*h;
//
//		face_key_2D_getpt[3].x = (a_face.landmarks.points[45].x - a_face.x)*w;
//		face_key_2D_getpt[3].y = (a_face.landmarks.points[45].y - a_face.y)*h;
//
//		face_key_2D_getpt[4].x = (a_face.landmarks.points[27].x - a_face.x)*w;
//		face_key_2D_getpt[4].y = (a_face.landmarks.points[27].y - a_face.y)*h;
//
//		face_key_2D_getpt[5].x = (a_face.landmarks.points[30].x - a_face.x)*w;
//		face_key_2D_getpt[5].y = (a_face.landmarks.points[30].y - a_face.y)*h;
//
//		face_key_2D_getpt[6].x = ((a_face.landmarks.points[28].x + a_face.landmarks.points[29].x) / 2 - a_face.x)*w;
//		face_key_2D_getpt[6].y = ((a_face.landmarks.points[28].y + a_face.landmarks.points[29].y) / 2 - a_face.y)*h;
//
//		face_key_2D_getpt[7].x = (a_face.landmarks.points[33].x - a_face.x)*w;
//		face_key_2D_getpt[7].y = (a_face.landmarks.points[33].y - a_face.y)*h;
//
//		face_key_2D_getpt[8].x = (a_face.landmarks.points[31].x - a_face.x)*w;
//		face_key_2D_getpt[8].y = (a_face.landmarks.points[31].y - a_face.y)*h;
//
//		face_key_2D_getpt[9].x = (a_face.landmarks.points[35].x - a_face.x)*w;
//		face_key_2D_getpt[9].y = (a_face.landmarks.points[35].y - a_face.y)*h;
//
//		face_key_2D_getpt[10].x = (a_face.landmarks.points[48].x - a_face.x)*w;
//		face_key_2D_getpt[10].y = (a_face.landmarks.points[48].y - a_face.y)*h;
//
//		face_key_2D_getpt[11].x = (a_face.landmarks.points[54].x - a_face.x)*w;
//		face_key_2D_getpt[11].y = (a_face.landmarks.points[54].y - a_face.y)*h;
//
//		face_key_2D_getpt[12].x = (a_face.landmarks.points[51].x - a_face.x)*w;
//		face_key_2D_getpt[12].y = (a_face.landmarks.points[51].y - a_face.y)*h;
//	}
//	else
//	{
//		face_key_2D_getpt[0].x = a_face.landmarks.points[39].x*w;
//		face_key_2D_getpt[0].y = a_face.landmarks.points[39].y*h;
//
//		face_key_2D_getpt[1].x = a_face.landmarks.points[36].x*w;
//		face_key_2D_getpt[1].y = a_face.landmarks.points[36].y*h;
//
//		face_key_2D_getpt[2].x = a_face.landmarks.points[42].x*w;
//		face_key_2D_getpt[2].y = a_face.landmarks.points[42].y*h;
//
//		face_key_2D_getpt[3].x = a_face.landmarks.points[45].x*w;
//		face_key_2D_getpt[3].y = a_face.landmarks.points[45].y*h;
//
//		face_key_2D_getpt[4].x = a_face.landmarks.points[27].x*w;
//		face_key_2D_getpt[4].y = a_face.landmarks.points[27].y*h;
//
//		face_key_2D_getpt[5].x = a_face.landmarks.points[30].x*w;
//		face_key_2D_getpt[5].y = a_face.landmarks.points[30].y*h;
//
//		face_key_2D_getpt[6].x = (a_face.landmarks.points[28].x + a_face.landmarks.points[29].x) / 2 * w;
//		face_key_2D_getpt[6].y = (a_face.landmarks.points[28].y + a_face.landmarks.points[29].y) / 2 * h;
//
//		face_key_2D_getpt[7].x = a_face.landmarks.points[33].x*w;
//		face_key_2D_getpt[7].y = a_face.landmarks.points[33].y*h;
//
//		face_key_2D_getpt[8].x = a_face.landmarks.points[31].x*w;
//		face_key_2D_getpt[8].y = a_face.landmarks.points[31].y*h;
//
//		face_key_2D_getpt[9].x = a_face.landmarks.points[35].x*w;
//		face_key_2D_getpt[9].y = a_face.landmarks.points[35].y*h;
//
//		face_key_2D_getpt[10].x = a_face.landmarks.points[48].x*w;
//		face_key_2D_getpt[10].y = a_face.landmarks.points[48].y*h;
//
//		face_key_2D_getpt[11].x = a_face.landmarks.points[54].x*w;
//		face_key_2D_getpt[11].y = a_face.landmarks.points[54].y*h;
//
//		face_key_2D_getpt[12].x = a_face.landmarks.points[51].x*w;
//		face_key_2D_getpt[12].y = a_face.landmarks.points[51].y*h;
//	}
//}

//cv::Mat CFaceAlign::FaceAlign(Facial_2D_pt keyPoints[13], cv::Mat src, int crop_width, int crop_height)
//{
//	vector<cv::Mat> channels;
//	cv::Mat dst_img[3];
//	cv::Mat tmp[3];
//	cv::Mat mergeImage;//锟斤拷锟斤拷目锟斤拷图片锟斤拷锟斤拷
//
//	cv::split(src, channels);//锟斤拷源图片锟街斤拷为RGB锟斤拷锟斤拷通锟斤拷
//	for (int i = 0; i<3; i++)
//	{
//		tmp[i] = cv::Mat(crop_height, crop_width, CV_8UC1);
//		dst_img[i] = cv::Mat(tmp[i]);
//	}
//
//	for (int i = 0; i < 3; i++)
//	{
//		//锟斤拷每锟斤拷通锟斤拷锟街憋拷锟斤拷锟斤拷affine转锟斤拷
//		Affine_transformation_2D(keyPoints, 
//			(unsigned char*)dst_img[i].data, 
//			(unsigned char*)channels.at(i).data, 
//			channels.at(i).rows, 
//			channels.at(i).cols);
//	}
//	//锟斤拷affine转锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷通锟斤拷锟斤拷锟叫合诧拷锟矫碉拷目锟斤拷图片锟斤拷锟斤拷锟斤拷
//	merge(dst_img, 3, mergeImage);
//
//	return mergeImage;
//}

//void CFaceAlign::CFaceAlign(Face& a_face, 
//	cv::Mat src,
//	int crop_width,
//	int crop_height)
//{
//	Facial_2D_pt keyPoints[13];
//	cvtLandmark2Align(a_face, keyPoints, src.cols, src.rows);
//	cv::Mat mergeImage = CFaceAlign(keyPoints, src, crop_width, crop_height);
//	mergeImage.copyTo(a_face.align_face_attr);
//	//cv::imshow("a_face.align_face", a_face.align_face);
//	//cv::waitKey(0);
//}

//void CFaceAlign::faceAlign5Points(Face& a_face, 
//	cv::Mat src, 
//	int crop_width, 
//	int crop_height, 
//	float eye_ratio, 
//	float mouse_ratio)
//{
//	cv::Point2f left_eye = cv::Point2f(
//		a_face.landmarks_5.points[0].x*src.cols,
//		a_face.landmarks_5.points[0].y*src.rows);
//	cv::Point2f right_eye = cv::Point2f(
//		a_face.landmarks_5.points[1].x*src.cols,
//		a_face.landmarks_5.points[1].y*src.rows);
//	cv::Point2f left_mouth = cv::Point2f(
//		a_face.landmarks_5.points[3].x*src.cols,
//		a_face.landmarks_5.points[3].y*src.rows);
//	cv::Point2f right_mouth = cv::Point2f(
//		a_face.landmarks_5.points[4].x*src.cols,
//		a_face.landmarks_5.points[4].y*src.rows);
//	cv::Point2f forehead = cv::Point2f(
//		(left_eye.x + right_eye.x) / 2.,
//		(left_eye.y + right_eye.y) / 2.);
//	cv::Point2f mouth = cv::Point2f(
//		(left_mouth.x + right_mouth.x) / 2.,
//		(left_mouth.y + right_mouth.y) / 2.);
//
//	float distance = sqrtf((mouth.x - forehead.x)*(mouth.x - forehead.x)
//		+ (mouth.y - forehead.y)*(mouth.y - forehead.y));
//	float angel = atan2f(mouth.x - forehead.x, mouth.y - forehead.y);
//	cv::Point2f ver = cv::Point2f(mouth.x + distance*cosf(angel), mouth.y - distance*sinf(angel));
//	//cout << distance << " " << cosf(angel);
//	//cout << forehead << mouth << ver << endl;
//	//cv::imwrite("affine.png", src);
//	//exit(0);
//	cv::Point2f dst_forehead = cv::Point2f(crop_width*0.5, crop_height*eye_ratio);
//	cv::Point2f dst_mouth = cv::Point2f(crop_width*0.5, crop_height*mouse_ratio);
//
//	float dst_distance = sqrtf((dst_mouth.x - dst_forehead.x)*(dst_mouth.x - dst_forehead.x)
//		+ (dst_mouth.y - dst_forehead.y)*(dst_mouth.y - dst_forehead.y));
//	float dst_angel = atan2f(dst_mouth.x - dst_forehead.x, dst_mouth.y - dst_forehead.y);
//	cv::Point2f dst_ver = cv::Point2f(dst_mouth.x + dst_distance*cosf(dst_angel), 
//		dst_mouth.y - dst_distance*sinf(dst_angel));
//
//	cv::Point2f dst_points[] = { dst_forehead,
//		dst_mouth,
//		dst_ver };
//	cv::Point2f src_points[] = { forehead, mouth, ver };
//	cv::Mat transform = cv::getAffineTransform(src_points, dst_points);
//
//	cv::warpAffine(src, a_face.align_face_attr, transform, cv::Size(crop_width, crop_height));
//
//	//cv::imshow("a_face.align_face", a_face.align_face);
//	//cv::waitKey(0);
//}
//
//void CFaceAlign::faceAlign5Points(Face& a_face, cv::Mat src, int crop_length)
//{
//	if (crop_length == 128)
//	{
//		cv::Point2f left_eye = cv::Point2f(
//			a_face.landmarks_5.points[0].x*src.cols,
//			a_face.landmarks_5.points[0].y*src.rows);
//		cv::Point2f right_eye = cv::Point2f(
//			a_face.landmarks_5.points[1].x*src.cols,
//			a_face.landmarks_5.points[1].y*src.rows);
//		cv::Point2f left_mouth = cv::Point2f(
//			a_face.landmarks_5.points[3].x*src.cols,
//			a_face.landmarks_5.points[3].y*src.rows);
//		cv::Point2f right_mouth = cv::Point2f(
//			a_face.landmarks_5.points[4].x*src.cols,
//			a_face.landmarks_5.points[4].y*src.rows);
//		cv::Point2f forehead = cv::Point2f(
//			(left_eye.x + right_eye.x) / 2.,
//			(left_eye.y + right_eye.y) / 2.);
//		cv::Point2f mouth = cv::Point2f(
//			(left_mouth.x + right_mouth.x) / 2.,
//			(left_mouth.y + right_mouth.y) / 2.);
//
//		float distance = sqrtf((mouth.x - forehead.x)*(mouth.x - forehead.x)
//			+ (mouth.y - forehead.y)*(mouth.y - forehead.y));
//		float angel = atan2f(mouth.x - forehead.x, mouth.y - forehead.y);
//		cv::Point2f ver = cv::Point2f(mouth.x + distance*cosf(angel), mouth.y - distance*sinf(angel));
//		//cout << distance << " " << cosf(angel);
//		//cout << forehead << mouth << ver << endl;
//		//cv::imwrite("affine.png", src);
//		//exit(0);
//
//		cv::Point2f dst_points[] = { cv::Point2f(64, 40), cv::Point2f(64, 80), cv::Point2f(104, 80) };
//		cv::Point2f src_points[] = { forehead, mouth, ver };
//		cv::Mat transform = cv::getAffineTransform(src_points, dst_points);
//
//		cv::warpAffine(src, a_face.align_face_attr, transform, cv::Size(crop_length, crop_length));
//	}
//	else if (crop_length==140)
//	{
//		cv::Point2f left_eye = cv::Point2f(
//			a_face.landmarks_5.points[0].x*src.cols,
//			a_face.landmarks_5.points[0].y*src.rows);
//		cv::Point2f right_eye = cv::Point2f(
//			a_face.landmarks_5.points[1].x*src.cols,
//			a_face.landmarks_5.points[1].y*src.rows);
//		cv::Point2f left_mouth = cv::Point2f(
//			a_face.landmarks_5.points[3].x*src.cols,
//			a_face.landmarks_5.points[3].y*src.rows);
//		cv::Point2f right_mouth = cv::Point2f(
//			a_face.landmarks_5.points[4].x*src.cols,
//			a_face.landmarks_5.points[4].y*src.rows);
//		cv::Point2f forehead = cv::Point2f(
//			(left_eye.x + right_eye.x) / 2.,
//			(left_eye.y + right_eye.y) / 2.);
//		cv::Point2f mouth = cv::Point2f(
//			(left_mouth.x + right_mouth.x) / 2.,
//			(left_mouth.y + right_mouth.y) / 2.);
//
//		float distance = sqrtf((mouth.x - forehead.x)*(mouth.x - forehead.x)
//			+ (mouth.y - forehead.y)*(mouth.y - forehead.y));
//		float angel = atan2f(mouth.x - forehead.x, mouth.y - forehead.y);
//		cv::Point2f ver = cv::Point2f(mouth.x + distance*cosf(angel), mouth.y - distance*sinf(angel));
//		//cout << distance << " " << cosf(angel);
//		//cout << forehead << mouth << ver << endl;
//		//cv::imwrite("affine.png", src);
//		//exit(0);
//
//		cv::Point2f dst_points[] = { cv::Point2f(70, 56), cv::Point2f(70, 101), cv::Point2f(115, 101) };
//		cv::Point2f src_points[] = { forehead, mouth, ver };
//		cv::Mat transform = cv::getAffineTransform(src_points, dst_points);
//
//		cv::warpAffine(src, a_face.align_face_attr, transform, cv::Size(crop_length, crop_length));
//	}
//}

void CFaceAlign::faceAlign5Points(DPFaceStruct& a_face, cv::Mat src, int crop_length)
{
    if (crop_length == 140)
    {
        cv::Point2f left_eye = cv::Point2f(
            a_face.landmarks_5[0][0] * src.cols,
            a_face.landmarks_5[0][1] * src.rows);
        cv::Point2f right_eye = cv::Point2f(
            a_face.landmarks_5[1][0] * src.cols,
            a_face.landmarks_5[1][1] * src.rows);
        cv::Point2f left_mouth = cv::Point2f(
            a_face.landmarks_5[3][0] * src.cols,
            a_face.landmarks_5[3][1] * src.rows);
        cv::Point2f right_mouth = cv::Point2f(
            a_face.landmarks_5[4][0] * src.cols,
            a_face.landmarks_5[4][1] * src.rows);
        cv::Point2f forehead = cv::Point2f(
            (left_eye.x + right_eye.x) / 2.,
            (left_eye.y + right_eye.y) / 2.);
        cv::Point2f mouth = cv::Point2f(
            (left_mouth.x + right_mouth.x) / 2.,
            (left_mouth.y + right_mouth.y) / 2.);

        float distance = sqrtf((mouth.x - forehead.x) * (mouth.x - forehead.x)
            + (mouth.y - forehead.y) * (mouth.y - forehead.y));
        float angel = atan2f(mouth.x - forehead.x, mouth.y - forehead.y);
        cv::Point2f ver = cv::Point2f(mouth.x + distance * cosf(angel), mouth.y - distance * sinf(angel));
        cv::Point2f dst_points[] = { cv::Point2f(70, 56), cv::Point2f(70, 101), cv::Point2f(115, 101) };
        cv::Point2f src_points[] = { forehead, mouth, ver };
        cv::Mat transform = cv::getAffineTransform(src_points, dst_points);
        cv::warpAffine(src, a_face.aligned, transform, cv::Size(crop_length, crop_length));
    }
    else if (crop_length == 128)
    {
        cv::Point2f left_eye = cv::Point2f(
            a_face.landmarks_5[0][0] * src.cols,
            a_face.landmarks_5[0][1] * src.rows);
        cv::Point2f right_eye = cv::Point2f(
            a_face.landmarks_5[1][0] * src.cols,
            a_face.landmarks_5[1][1] * src.rows);
        cv::Point2f left_mouth = cv::Point2f(
            a_face.landmarks_5[3][0] * src.cols,
            a_face.landmarks_5[3][1] * src.rows);
        cv::Point2f right_mouth = cv::Point2f(
            a_face.landmarks_5[4][0] * src.cols,
            a_face.landmarks_5[4][1] * src.rows);
        cv::Point2f forehead = cv::Point2f(
            (left_eye.x + right_eye.x) / 2.,
            (left_eye.y + right_eye.y) / 2.);
        cv::Point2f mouth = cv::Point2f(
            (left_mouth.x + right_mouth.x) / 2.,
            (left_mouth.y + right_mouth.y) / 2.);

        float distance = sqrtf((mouth.x - forehead.x) * (mouth.x - forehead.x)
            + (mouth.y - forehead.y) * (mouth.y - forehead.y));
        float angel = atan2f(mouth.x - forehead.x, mouth.y - forehead.y);
        cv::Point2f ver = cv::Point2f(mouth.x + distance * cosf(angel), mouth.y - distance * sinf(angel));
        cv::Point2f dst_points[] = { cv::Point2f(64, 50), cv::Point2f(64, 95), cv::Point2f(109, 95) };
        cv::Point2f src_points[] = { forehead, mouth, ver };
        cv::Mat transform = cv::getAffineTransform(src_points, dst_points);
        cv::warpAffine(src, a_face.aligned, transform, cv::Size(crop_length, crop_length));
    }
}

bool CFaceAlign::CropImage_112x96(const cv::Mat& img, const float* facial5point, cv::Mat& crop)
{
    cv::Size designed_size(96, 112);
    float coord5point[10] =
    {
        30.2946, 51.6963,
        65.5318, 51.5014,
        48.0252, 71.7366,
        33.5493, 92.3655,
        62.7299, 92.2041
    };

    cv::Mat transform;
    //clock_t t1 = clock();
    _findSimilarity(5, facial5point, coord5point, transform);
    //clock_t t2 = clock();
    cv::warpAffine(img, crop, transform, designed_size);
    //clock_t t3 = clock();
    //printf("findtrans:%.3f, warp:%.3f\n", 0.001*(t2 - t1), 0.001*(t3 - t2));
    return true;
}

bool CFaceAlign::CropImage_112x112(const cv::Mat& img, const float* facial5point, cv::Mat& crop)
{
    cv::Size designed_size(112, 112);
    float coord5point[10] =
    {
        30.2946 + 8, 51.6963,
        65.5318 + 8, 51.5014,
        48.0252 + 8, 71.7366,
        33.5493 + 8, 92.3655,
        62.7299 + 8, 92.2041
    };

    cv::Mat transform;
    //clock_t t1 = clock();
    _findSimilarity(5, facial5point, coord5point, transform);
    //clock_t t2 = clock();
    cv::warpAffine(img, crop, transform, designed_size);
    //clock_t t3 = clock();
    //printf("findtrans:%.3f, warp:%.3f\n", 0.001*(t2 - t1), 0.001*(t3 - t2));
    return true;
}

bool CFaceAlign::CropImage_120x120(const cv::Mat& img, const float* facial5point, cv::Mat& crop)
{
    cv::Size designed_size(120, 120);
    float coord5point[10] =
    {
        (30.2946 + 8) * 1.07142857, 51.6963 * 1.07142857,
        (65.5318 + 8) * 1.07142857, 51.5014 * 1.07142857,
        (48.0252 + 8) * 1.07142857, 71.7366 * 1.07142857,
        (33.5493 + 8) * 1.07142857, 92.3655 * 1.07142857,
        (62.7299 + 8) * 1.07142857, 92.2041 * 1.07142857
    };

    cv::Mat transform;
    //clock_t t1 = clock();
    _findSimilarity(5, facial5point, coord5point, transform);
    //clock_t t2 = clock();
    cv::warpAffine(img, crop, transform, designed_size);
    //clock_t t3 = clock();
    //printf("findtrans:%.3f, warp:%.3f\n", 0.001*(t2 - t1), 0.001*(t3 - t2));
    return true;
}

bool CFaceAlign::CropImage_140x140(const cv::Mat& img, const float* facial5point, cv::Mat& crop)
{
    cv::Size designed_size(140, 140);
    float coord5point[10] =
    {
        (30.2946 + 8) * 1.25, 51.6963 * 1.25,
        (65.5318 + 8) * 1.25, 51.5014 * 1.25,
        (48.0252 + 8) * 1.25, 71.7366 * 1.25,
        (33.5493 + 8) * 1.25, 92.3655 * 1.25,
        (62.7299 + 8) * 1.25, 92.2041 * 1.25
    };

    cv::Mat transform;
    //clock_t t1 = clock();
    _findSimilarity(5, facial5point, coord5point, transform);
    //clock_t t2 = clock();
    cv::warpAffine(img, crop, transform, designed_size);
    //clock_t t3 = clock();
    //printf("findtrans:%.3f, warp:%.3f\n", 0.001*(t2 - t1), 0.001*(t3 - t2));
    return true;
}

bool CFaceAlign::CropImage_160x160(const cv::Mat& img, const float* facial5point, cv::Mat& crop)
{
    cv::Size designed_size(160, 160);
    float coord5point[10] =
    {
        30.2946 + 32, 51.6963 + 24,
        65.5318 + 32, 51.5014 + 24,
        48.0252 + 32, 71.7366 + 24,
        33.5493 + 32, 92.3655 + 24,
        62.7299 + 32, 92.2041 + 24
    };

    cv::Mat transform;
    //clock_t t1 = clock();
    _findSimilarity(5, facial5point, coord5point, transform);
    //clock_t t2 = clock();
    cv::warpAffine(img, crop, transform, designed_size);
    //clock_t t3 = clock();
    //printf("findtrans:%.3f, warp:%.3f\n", 0.001*(t2 - t1), 0.001*(t3 - t2));
    return true;
}

void CFaceAlign::_findNonreflectiveSimilarity(int nPts, const float* uv, const float* xy, cv::Mat& transform)
{
    /*
    %
    % For a nonreflective similarity :
    %
    % let sc = s*cos(theta)
    % let ss = s*sin(theta)
    %
    %				  [sc -ss
    %[u v] = [x y 1] * ss  sc
    %                  tx  ty]
    %
    % There are 4 unknowns: sc, ss, tx, ty.
    %
    % Another way to write this is :
    %
    % u = [x y 1 0] * [sc
    %                  ss
    %                  tx
    %                  ty]
    %
    % v = [y -x 0 1] * [sc
    %                   ss
    %                   tx
    %                   ty]
    %
    % With 2 or more correspondence points we can combine the u equations and
    % the v equations for one linear system to solve for sc, ss, tx, ty.
    %
    %[u1] = [x1  y1  1  0] * [sc]
    %[u2]   [x2  y2  1  0]   [ss]
    %[...]  [...]            [tx]
    %[un]   [xn  yn  1  0]   [ty]
    %[v1]   [y1 -x1  0  1]
    %[v2]   [y2 -x2  0  1]
    %[...]  [...]
    %[vn]   [yn - xn  0  1]
    %
    % Or rewriting the above matrix equation :
    % U = X * r, where r = [sc ss tx ty]'
    % so r = X\U.
    %


    x = xy(:, 1);
    y = xy(:, 2);
    X = [x   y  ones(M, 1)   zeros(M, 1);
    y  -x  zeros(M, 1)  ones(M, 1)];

    u = uv(:, 1);
    v = uv(:, 2);
    U = [u; v];

    % We know that X * r = U
    if rank(X) >= 2 * K
    r = X \ U;
    else
    error(message('images:cp2tform:twoUniquePointsReq'))
    end

    sc = r(1);
    ss = r(2);
    tx = r(3);
    ty = r(4);

    Tinv = [sc -ss 0;
    ss  sc 0;
    tx  ty 1];

    T = inv(Tinv);
    T(:, 3) = [0 0 1]';

    trans = maketform('affine', T);
    */

    int type = CV_32FC1;
    //if (_strcmpi(typeid(float).name(), "double") == 0)
    //    type = CV_64FC1;
    //int type = CV_64FC1;
    //using TmpType = double;
    using TmpType = float;
    cv::Mat X(nPts * 2, 4, type);
    cv::Mat U(nPts * 2, 1, type);
    for (int i = 0; i < nPts; i++)
    {
        X.ptr<TmpType>(i)[0] = xy[i * 2 + 0];
        X.ptr<TmpType>(i)[1] = xy[i * 2 + 1];
        X.ptr<TmpType>(i)[2] = 1;
        X.ptr<TmpType>(i)[3] = 0;
        X.ptr<TmpType>(i + nPts)[0] = xy[i * 2 + 1];
        X.ptr<TmpType>(i + nPts)[1] = -xy[i * 2 + 0];
        X.ptr<TmpType>(i + nPts)[2] = 0;
        X.ptr<TmpType>(i + nPts)[3] = 1;
        U.ptr<TmpType>(i)[0] = uv[i * 2 + 0];
        U.ptr<TmpType>(i + nPts)[0] = uv[i * 2 + 1];
    }
    cv::Mat r(4, 1, type);
    if (!cv::solve(X, U, r, cv::DECOMP_SVD))
    {
        //std::cout << "failed to solve\n";
        return;
    }
    //printf("solve:%.3f\n", t2 - t1);
    TmpType sc = r.ptr<TmpType>(0)[0];
    TmpType ss = r.ptr<TmpType>(1)[0];
    TmpType tx = r.ptr<TmpType>(2)[0];
    TmpType ty = r.ptr<TmpType>(3)[0];

    TmpType Tinv[9] =
    {
        sc, -ss, 0,
        ss, sc, 0,
        tx, ty, 1
    };

    /*for (int i = 0; i < 3; i++)
    {
    for (int j = 0; j < 3; j++)
    {
    printf("%12.5f", Tinv[i * 3 + j]);
    }
    printf("\n");
    }*/
    cv::Mat Tinv_mat(3, 3, type), T_mat(3, 3, type);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            Tinv_mat.ptr<TmpType>(i)[j] = Tinv[i * 3 + j];
    }
    transform = Tinv_mat;

}

void CFaceAlign::_findSimilarity(int nPts, const float* uv, const float* xy, cv::Mat& transform)
{
    /*
    function [trans, output] = findSimilarity(uv,xy,options)
    %
    % The similarities are a superset of the nonreflective similarities as they may
    % also include reflection.
    %
    % let sc = s*cos(theta)
    % let ss = s*sin(theta)
    %
    %                   [ sc -ss
    % [u v] = [x y 1] *   ss  sc
    %                     tx  ty]
    %
    %          OR
    %
    %                   [ sc  ss
    % [u v] = [x y 1] *   ss -sc
    %                     tx  ty]
    %
    % Algorithm:
    % 1) Solve for trans1, a nonreflective similarity.
    % 2) Reflect the xy data across the Y-axis,
    %    and solve for trans2r, also a nonreflective similarity.
    % 3) Transform trans2r to trans2, undoing the reflection done in step 2.
    % 4) Use TFORMFWD to transform uv using both trans1 and trans2,
    %    and compare the results, returning the transformation corresponding
    %    to the smaller L2 norm.

    % Need to reset options.K to prepare for calls to findNonreflectiveSimilarity.
    % This is safe because we already checked that there are enough point pairs.
    options.K = 2;

    % Solve for trans1
    [trans1, output] = findNonreflectiveSimilarity(uv,xy,options);


    % Solve for trans2

    % manually reflect the xy data across the Y-axis
    xyR = xy;
    xyR(:,1) = -1*xyR(:,1);

    trans2r  = findNonreflectiveSimilarity(uv,xyR,options);

    % manually reflect the tform to undo the reflection done on xyR
    TreflectY = [-1  0  0;
    0  1  0;
    0  0  1];
    trans2 = maketform('affine', trans2r.tdata.T * TreflectY);


    % Figure out if trans1 or trans2 is better
    xy1 = tformfwd(trans1,uv);
    norm1 = norm(xy1-xy);

    xy2 = tformfwd(trans2,uv);
    norm2 = norm(xy2-xy);

    if norm1 <= norm2
    trans = trans1;
    else
    trans = trans2;
    end
    */

    int type = CV_32FC1;
    //if (_strcmpi(typeid(float).name(), "double") == 0)
    //    type = CV_64FC1;

    //int type = CV_64FC1;
    //using TmpType = double;
    using TmpType = float;
    cv::Mat transform1, transform2R, transform2;
    //clock_t t1 = clock();
    _findNonreflectiveSimilarity(nPts, uv, xy, transform1);
    //clock_t t2 = clock();
    /*for (int i = 0; i < 3; i++)
    {
    for (int j = 0; j < 3; j++)
    {
    printf("%12.5f", transform1.ptr<float>(i)[j]);
    }
    printf("\n");
    }*/
    float* xyR = new float[nPts * 2];
    for (int i = 0; i < nPts; i++)
    {
        xyR[i * 2 + 0] = -xy[i * 2 + 0];
        xyR[i * 2 + 1] = xy[i * 2 + 1];
    }
    //clock_t t3 = clock();
    _findNonreflectiveSimilarity(nPts, uv, xyR, transform2R);
    //clock_t t4 = clock();
    /*for (int i = 0; i < 3; i++)
    {
    for (int j = 0; j < 3; j++)
    {
    printf("%12.5f", transform2R.ptr<float>(i)[j]);
    }
    printf("\n");
    }*/

    const TmpType TreflectY[9] =
    {
        -1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };
    cv::Mat TreflectY_mat(3, 3, type);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            TreflectY_mat.ptr<TmpType>(i)[j] = TreflectY[i * 3 + j];
    }
    transform2 = transform2R * TreflectY_mat;

    /*for (int i = 0; i < 3; i++)
    {
    for (int j = 0; j < 3; j++)
    {
    printf("%12.5f", transform2.ptr<float>(i)[j]);
    }
    printf("\n");
    }*/

    //forward transform
    TmpType norm1 = 0, norm2 = 0;
    for (int p = 0; p < nPts; p++)
    {
        TmpType uv1_x = transform1.ptr<TmpType>(0)[0] * xy[p * 2 + 0] + transform1.ptr<TmpType>(1)[0] * xy[p * 2 + 1] + transform1.ptr<TmpType>(2)[0];
        TmpType uv1_y = transform1.ptr<TmpType>(0)[1] * xy[p * 2 + 0] + transform1.ptr<TmpType>(1)[1] * xy[p * 2 + 1] + transform1.ptr<TmpType>(2)[1];
        TmpType uv2_x = transform2.ptr<TmpType>(0)[0] * xy[p * 2 + 0] + transform2.ptr<TmpType>(1)[0] * xy[p * 2 + 1] + transform2.ptr<TmpType>(2)[0];
        TmpType uv2_y = transform2.ptr<TmpType>(0)[1] * xy[p * 2 + 0] + transform2.ptr<TmpType>(1)[1] * xy[p * 2 + 1] + transform2.ptr<TmpType>(2)[1];

        norm1 += (uv[p * 2 + 0] - uv1_x) * (uv[p * 2 + 0] - uv1_x) + (uv[p * 2 + 1] - uv1_y) * (uv[p * 2 + 1] - uv1_y);
        norm2 += (uv[p * 2 + 0] - uv2_x) * (uv[p * 2 + 0] - uv2_x) + (uv[p * 2 + 1] - uv2_y) * (uv[p * 2 + 1] - uv2_y);
    }

    //clock_t t5 = clock();
    cv::Mat tmp;
    if (norm1 < norm2)
        cv::invert(transform1, tmp, cv::DECOMP_SVD);
    else
        cv::invert(transform2, tmp, cv::DECOMP_SVD);
    //clock_t t6 = clock();

    cv::Mat trans(2, 3, type);
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            trans.ptr<TmpType>(i)[j] = tmp.ptr<TmpType>(j)[i];
            //printf("%f ", trans.ptr<TmpType>(i)[j]);
        }
        //printf("\n");
    }

    transform = trans;
    //printf("%f,%f,%f\n", 0.001*(t2 - t1), 0.001*(t4 - t3), 0.001*(t6 - t5));
}

void CFaceAlign::faceAlign5PointsLS(DPFaceStruct& a_face,
    cv::Mat src,
    int crop_length)
{
    float landmark5[10];
    for (int l = 0; l < LANDMARK_5_LEN; ++l)
    {
        landmark5[l * 2] = a_face.landmarks_5[l][0] * src.cols;
        landmark5[l * 2 + 1] = a_face.landmarks_5[l][1] * src.rows;
    }
    if (crop_length == 112)
        CropImage_112x112(src, landmark5, a_face.aligned);
    else if (crop_length == 120)
        CropImage_120x120(src, landmark5, a_face.aligned);
    else if (crop_length == 140)
        CropImage_140x140(src, landmark5, a_face.aligned);
    else if (crop_length == 160)
        CropImage_160x160(src, landmark5, a_face.aligned);
    else
        CropImage_140x140(src, landmark5, a_face.aligned);
}
