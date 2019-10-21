#include <vector>
#include <queue>
#include <math.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main()  
{
    Mat img_0 = imread( "HDR_0.jpg" );
    Mat img_m2 = imread( "HDR_-2.jpg" );
    Mat img_p2 = imread( "HDR_+2.jpg" );
    
//    imshow( "img_0", img_0 );
//    imshow( "img_-2", img_m2 );
//    imshow( "img_+2", img_p2 );
    
        // Gamma correction
    Mat img_0_Gamma  = img_0.clone();
    Mat img_m2_Gamma = img_m2.clone();
    Mat img_p2_Gamma = img_p2.clone();
    float Gamma = 2.2f;
    for ( int x = 0; x < img_0.cols; x ++ )
    {
        for ( int y = 0; y < img_0.rows; y ++ )
        {
            img_0_Gamma.at< Vec3b >(y, x).val[0] = uchar( pow( float(img_0.at< Vec3b >(y, x).val[0]) / 255, Gamma ) * 255 );
            img_0_Gamma.at< Vec3b >(y, x).val[1] = uchar( pow( float(img_0.at< Vec3b >(y, x).val[1]) / 255, Gamma ) * 255 );
            img_0_Gamma.at< Vec3b >(y, x).val[2] = uchar( pow( float(img_0.at< Vec3b >(y, x).val[2]) / 255, Gamma ) * 255 );
            
            img_m2_Gamma.at< Vec3b >(y, x).val[0] = uchar( pow( float(img_m2.at< Vec3b >(y, x).val[0]) / 255, Gamma ) * 255 );
            img_m2_Gamma.at< Vec3b >(y, x).val[1] = uchar( pow( float(img_m2.at< Vec3b >(y, x).val[1]) / 255, Gamma ) * 255 );
            img_m2_Gamma.at< Vec3b >(y, x).val[2] = uchar( pow( float(img_m2.at< Vec3b >(y, x).val[2]) / 255, Gamma ) * 255 );
            
            img_p2_Gamma.at< Vec3b >(y, x).val[0] = uchar( pow( float(img_p2.at< Vec3b >(y, x).val[0]) / 255, Gamma ) * 255 );
            img_p2_Gamma.at< Vec3b >(y, x).val[1] = uchar( pow( float(img_p2.at< Vec3b >(y, x).val[1]) / 255, Gamma ) * 255 );
            img_p2_Gamma.at< Vec3b >(y, x).val[2] = uchar( pow( float(img_p2.at< Vec3b >(y, x).val[2]) / 255, Gamma ) * 255 );
        }
    }
    imshow( "img_0_Gamma", img_0_Gamma );
    imshow( "img_-2_Gamma", img_m2_Gamma );
    imshow( "img_+2_Gamma", img_p2_Gamma );
    
        // Nornalizetion
    Mat img_0_Norm  = Mat::zeros( img_0_Gamma.size(), CV_32FC3 );
    Mat img_m2_Norm = Mat::zeros( img_m2_Gamma.size(), CV_32FC3 );
    Mat img_p2_Norm = Mat::zeros( img_p2_Gamma.size(), CV_32FC3 );
    for ( int x = 0; x < img_0.cols; x ++ )
    {
        for ( int y = 0; y < img_0.rows; y ++ )
        {
            img_0_Norm.at< Vec3f >(y, x).val[0] = img_0_Gamma.at< Vec3b >(y, x).val[0] / 255.0f;
            img_0_Norm.at< Vec3f >(y, x).val[1] = img_0_Gamma.at< Vec3b >(y, x).val[1] / 255.0f;
            img_0_Norm.at< Vec3f >(y, x).val[2] = img_0_Gamma.at< Vec3b >(y, x).val[2] / 255.0f;
            
            img_m2_Norm.at< Vec3f >(y, x).val[0] = img_m2_Gamma.at< Vec3b >(y, x).val[0] / 255.0f;
            img_m2_Norm.at< Vec3f >(y, x).val[1] = img_m2_Gamma.at< Vec3b >(y, x).val[1] / 255.0f;
            img_m2_Norm.at< Vec3f >(y, x).val[2] = img_m2_Gamma.at< Vec3b >(y, x).val[2] / 255.0f;
            
            img_p2_Norm.at< Vec3f >(y, x).val[0] = img_p2_Gamma.at< Vec3b >(y, x).val[0] / 255.0f;
            img_p2_Norm.at< Vec3f >(y, x).val[1] = img_p2_Gamma.at< Vec3b >(y, x).val[1] / 255.0f;
            img_p2_Norm.at< Vec3f >(y, x).val[2] = img_p2_Gamma.at< Vec3b >(y, x).val[2] / 255.0f;
        }
    }
    
        // Averaging
    Mat img_Resault = Mat::zeros( img_0.size(), CV_8UC3 );
    for ( int x = 0; x < img_0.cols; x ++ )
    {
        for ( int y = 0; y < img_0.rows; y ++ )
        {
            img_Resault.at< Vec3b >(y, x).val[0] = uchar( ( img_0_Norm.at< Vec3f >(y, x).val[0] +
                                                            img_m2_Norm.at< Vec3f >(y, x).val[0] + 
                                                            img_p2_Norm.at< Vec3f >(y, x).val[0] ) / 3.0f * 255.0f );
            img_Resault.at< Vec3b >(y, x).val[1] = uchar( ( img_0_Norm.at< Vec3f >(y, x).val[1] +
                                                            img_m2_Norm.at< Vec3f >(y, x).val[1] + 
                                                            img_p2_Norm.at< Vec3f >(y, x).val[1] ) / 3.0f * 255.0f );
            img_Resault.at< Vec3b >(y, x).val[2] = uchar( ( img_0_Norm.at< Vec3f >(y, x).val[2] +
                                                            img_m2_Norm.at< Vec3f >(y, x).val[2] + 
                                                            img_p2_Norm.at< Vec3f >(y, x).val[2] ) / 3.0f * 255.0f );
        }
    }
    imshow( "img_Resault", img_Resault );
    imwrite( "img_Resault.png", img_Resault );
    
    waitKey(0);
    return 0;
}
