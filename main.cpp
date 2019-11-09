#include <vector>
#include <queue>
#include <math.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>

using namespace std;
using namespace cv;

void minTenPersent( Mat &src )
{
    cvtColor( src, src, COLOR_BGR2HSV );
    
    Mat channels[3];
    split( src, channels );
    double min_V = 0, max_V = 0;
    minMaxLoc( channels[2], &min_V, &max_V );
    cout << "Max_V: " << max_V << endl;
    double ten_percent = double( max_V * 0.1 );
    cout << "10%: " << ten_percent << endl;
    for ( int i = 0; i < int( src.total() ); i++ )
    {
        int temp = int( src.at< Vec3b >(i).val[2] - ten_percent );
        if ( temp > 0 )
            src.at< Vec3b >(i).val[2] = uchar( src.at< Vec3b >(i).val[2] - ten_percent );
    }
    
    cvtColor( src, src, COLOR_HSV2BGR );
}

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
    
        // -10%
//    minTenPersent( img_0_Gamma );
//    minTenPersent( img_m2_Gamma );
//    minTenPersent( img_p2_Gamma );
    
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
    Mat img_Resault = Mat::zeros( img_0.size(), CV_8UC3 );  // CV_8UC3
    for ( int x = 0; x < img_0.cols; x ++ )
    {
        for ( int y = 0; y < img_0.rows; y ++ )
        {
            img_Resault.at< Vec3b >(y, x).val[0] = uchar( ( img_0_Norm.at< Vec3f >(y, x).val[0] +
                                                            img_m2_Norm.at< Vec3f >(y, x).val[0] + 
                                                            img_p2_Norm.at< Vec3f >(y, x).val[0] ) / 3.0f * 255.0f);
            img_Resault.at< Vec3b >(y, x).val[1] = uchar( ( img_0_Norm.at< Vec3f >(y, x).val[1] +
                                                            img_m2_Norm.at< Vec3f >(y, x).val[1] + 
                                                            img_p2_Norm.at< Vec3f >(y, x).val[1] ) / 3.0f * 255.0f);
            img_Resault.at< Vec3b >(y, x).val[2] = uchar( ( img_0_Norm.at< Vec3f >(y, x).val[2] +
                                                            img_m2_Norm.at< Vec3f >(y, x).val[2] + 
                                                            img_p2_Norm.at< Vec3f >(y, x).val[2] ) / 3.0f * 255.0f);
        }
    }
    
    imshow( "img_Resault", img_Resault );
    imwrite( "img_Resault.png", img_Resault );
    
    vector< Mat > images;
    images.push_back( img_0 );
    images.push_back( img_m2 );
    images.push_back( img_p2 );
    
    vector< float > times;
    float center = 4.2f;
    float epsilo = 3.f; 
    times.push_back( center );
    times.push_back( center - epsilo );
    times.push_back( center + epsilo );
    
    
    Mat responseDebevec;
    Ptr< CalibrateDebevec > calibrateDebevec = createCalibrateDebevec();
    calibrateDebevec->process( images, responseDebevec, times );
    
    Mat hdrDebevec;
    Ptr< MergeDebevec > mergeDebevec = createMergeDebevec();
    mergeDebevec->process( images, hdrDebevec, times, responseDebevec );
    imshow( "HDR", hdrDebevec );
    imwrite( "img_Resault_HDR.png", hdrDebevec * 255 );
    
    
    waitKey(0);
    return 0;
}
