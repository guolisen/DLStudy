#include <stdio.h>
#include <iostream>
#include "MNIST.h"
//#include <Windows.h>

#include "SoftmaxRegression.h"

int main()
{
    srand((int)time(NULL));
    mnist::MNIST m("train_images.dat", "train_labels.dat");
    
    boost::optional<mnist::LABLE_DATA_VEC> lbVec = m.GetLableList(5000, 5300);
    boost::optional<mnist::IMAGE_DATA_VEC> imVec = m.GetImageList(5000, 5300);

    //m.PrintVec(imVec);

    //mnist::LABLE_DATA lable = m.GetLable(98);
    //unsigned char t = *lable;
    //bool r = m.CheckResult(98,3);

    softmax::SoftmaxRegression soft(10, 28*28);
    softmax::SoftmaxRegression softold(10, 28*28);
    double old_rate = 0.0;
    for(int i =0; i<10; ++i)
    {
        boost::optional<mnist::LABLE_DATA_VEC> lbVecTest = m.GetLableList(0 + i*2000, i*2000 + 2000);
        boost::optional<mnist::IMAGE_DATA_VEC> imVecTest = m.GetImageList(0 + i*2000, i*2000 + 2000);

        std::cout << "StudyTimes: " << std::endl;

        soft.Exercise(*lbVecTest, *imVecTest);

        mnist::LABLE_DATA_VEC::iterator iterYTest = lbVec->begin();
        mnist::IMAGE_DATA_VEC::iterator iterXTest = imVec->begin();
        int len = 28*28;
        double correct = 0;
        double total = lbVec->size();

        std::cout << "Test: " << std::endl;
        for(; (iterYTest != lbVec->end()) && (iterXTest != imVec->end()); ++iterYTest, ++iterXTest)
        {
            mnist::IMAGE_DATA xData = *iterXTest;
            softmax::Vector xi(len, 0.0);
            for(int j = 0; j < len; ++j)
                xi[j] = ((xData.get())[j] / 255.0);

            //std::cout << xi << std::endl;
            double res = soft.SoftMaxHx(xi);
            mnist::LABLE_DATA yi = *iterYTest;

            //printf("T: %d H: %f\n", *yi, res);
            if(softmax::EQ_DOUBLE(*yi, res))
                ++correct;
        }
        double rate = correct/total;
        std::cout << "Correct: " << rate << std::endl;
        if(old_rate > rate)
            break;
        old_rate = rate;
        softold = soft;
    }

    for(int i =0; i<10; ++i)
    {
        boost::optional<mnist::LABLE_DATA_VEC> lbVecTest = m.GetLableList(500 + i*1500, i*1500 + 2000);
        boost::optional<mnist::IMAGE_DATA_VEC> imVecTest = m.GetImageList(500 + i*1500, i*1500 + 2000);

        mnist::LABLE_DATA_VEC::iterator iterYTest = lbVecTest->begin();
        mnist::IMAGE_DATA_VEC::iterator iterXTest = imVecTest->begin();
        int len = 28*28;
        double correct = 0;
        double total = lbVecTest->size();

        std::cout << "!!!Test: " << std::endl;
        for(; (iterYTest != lbVecTest->end()) && (iterXTest != imVecTest->end()); ++iterYTest, ++iterXTest)
        {
            mnist::IMAGE_DATA xData = *iterXTest;
            softmax::Vector xi(len, 0.0);
            for(int j = 0; j < len; ++j)
                xi[j] = ((xData.get())[j] / 255.0);

            //std::cout << xi << std::endl;
            double res = softold.SoftMaxHx(xi);
            mnist::LABLE_DATA yi = *iterYTest;

            //printf("T: %d H: %f\n", *yi, res);
            if(softmax::EQ_DOUBLE(*yi, res))
                ++correct;
        }
        double rate = correct/total;
        std::cout << "!Correct: " << rate << std::endl;
    }




    return 0;
}